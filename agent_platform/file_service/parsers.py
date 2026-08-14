"""
文件解析器 — 将真实业务文件（Excel/PDF/Word）解析为 HTML 预览

支持格式:
  - .xlsx  : openpyxl
  - .xls   : xlrd
  - .pdf   : pdfplumber / PyPDF2 / fitz (PyMuPDF) 按可用性降级
  - .docx  : python-docx
  - .doc   : 老格式，仅提取基础信息（无法内嵌预览则提示下载）

所有解析器返回 HTML 片段，前端内嵌展示。
依赖库不可用时优雅降级（提示下载原文件）。
"""

import html
import logging
import os

logger = logging.getLogger(__name__)


def _esc(text: str) -> str:
    """HTML 转义"""
    return html.escape(text or "")


def _fmt_cell(value) -> str:
    """格式化单元格值"""
    if value is None:
        return ""
    return _esc(str(value))


# ============================================================
# Excel 解析
# ============================================================
def parse_excel(path: str, max_rows: int = 200, max_cols: int = 40) -> str:
    """解析 Excel (.xlsx/.xls) 为 HTML 表格"""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".xlsx":
            return _parse_xlsx(path, max_rows, max_cols)
        else:
            return _parse_xls(path, max_rows, max_cols)
    except Exception as e:
        logger.warning("Excel 解析失败 %s: %s", path, e)
        return _fallback_html("Excel 解析失败", str(e))


def _parse_xlsx(path: str, max_rows: int, max_cols: int) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    sheets = wb.sheetnames[:5]  # 最多展示 5 个 sheet
    html_parts = []
    for sheet_name in sheets:
        ws = wb[sheet_name]
        html_parts.append(f'<h4 class="fp-sheet">📄 {_esc(sheet_name)}</h4>')
        html_parts.append('<table class="fp-table"><tbody>')
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            if row_count >= max_rows:
                html_parts.append(
                    f'<tr><td colspan="{max_cols}" class="fp-more">… 表格过长，仅预览前 {max_rows} 行</td></tr>'
                )
                break
            cells = list(row)[:max_cols]
            # 用 <th> 突出前 2 行（表头）
            tag = "th" if row_count < 2 else "td"
            html_parts.append(
                "<tr>" + "".join(f"<{tag}>{_fmt_cell(c)}</{tag}>" for c in cells) + "</tr>"
            )
            row_count += 1
        html_parts.append("</tbody></table>")
        if row_count == 0:
            html_parts.append('<p class="fp-empty">（空表）</p>')
    wb.close()
    return "".join(html_parts)


def _parse_xls(path: str, max_rows: int, max_cols: int) -> str:
    import xlrd

    wb = xlrd.open_workbook(path, on_demand=True)
    html_parts = []
    for sheet_name in wb.sheet_names()[:5]:
        ws = wb.sheet_by_name(sheet_name)
        html_parts.append(f'<h4 class="fp-sheet">📄 {_esc(sheet_name)}</h4>')
        html_parts.append('<table class="fp-table"><tbody>')
        for r in range(min(ws.nrows, max_rows)):
            tag = "th" if r < 2 else "td"
            cells = [ws.cell_value(r, c) for c in range(min(ws.ncols, max_cols))]
            html_parts.append(
                "<tr>" + "".join(f"<{tag}>{_fmt_cell(c)}</{tag}>" for c in cells) + "</tr>"
            )
        html_parts.append("</tbody></table>")
    return "".join(html_parts)


# ============================================================
# PDF 解析
# ============================================================
def parse_pdf(path: str, max_chars: int = 20000) -> str:
    """解析 PDF 为 HTML 文本"""
    try:
        text = _extract_pdf_text(path)
        if not text.strip():
            return _fallback_html("PDF 无文本内容", "该 PDF 可能为扫描件，无法提取文本，请下载原文件查看。")
        clipped = text[:max_chars]
        html_parts = ['<div class="fp-pdf">']
        for para in clipped.split("\n"):
            p = para.strip()
            if p:
                html_parts.append(f"<p>{_esc(p)}</p>")
        if len(text) > max_chars:
            html_parts.append('<p class="fp-more">… 内容过长，仅预览前 ' + str(max_chars) + " 字符</p>")
        html_parts.append("</div>")
        return "".join(html_parts)
    except Exception as e:
        logger.warning("PDF 解析失败 %s: %s", path, e)
        return _fallback_html("PDF 解析失败", str(e))


def _extract_pdf_text(path: str) -> str:
    # 优先 pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages[:20]:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)
    except ImportError:
        pass
    # 其次 PyMuPDF
    try:
        import fitz

        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        pass
    # 最后 PyPDF2
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages[:20]]
    return "\n".join(pages)


# ============================================================
# Word 解析
# ============================================================
def parse_docx(path: str, max_chars: int = 20000) -> str:
    """解析 .docx 为 HTML 文本"""
    try:
        from docx import Document

        doc = Document(path)
        html_parts = ['<div class="fp-word">']
        char_count = 0
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            html_parts.append(f"<p>{_esc(text)}</p>")
            char_count += len(text)
            if char_count >= max_chars:
                html_parts.append('<p class="fp-more">… 内容过长，仅预览部分</p>')
                break
        # 表格
        for table in doc.tables[:10]:
            html_parts.append('<table class="fp-table"><tbody>')
            for ri, row in enumerate(table.rows[:50]):
                tag = "th" if ri == 0 else "td"
                html_parts.append(
                    "<tr>" + "".join(f"<{tag}>{_fmt_cell(c.text)}</{tag}>" for c in row.cells[:20]) + "</tr>"
                )
            html_parts.append("</tbody></table>")
        html_parts.append("</div>")
        return "".join(html_parts)
    except Exception as e:
        logger.warning("DOCX 解析失败 %s: %s", path, e)
        return _fallback_html("Word 解析失败", str(e))


def parse_doc(path: str) -> str:
    """解析老版 .doc：尝试文本提取，失败则提示下载"""
    try:
        # 尝试用 antiword（Linux 工具）
        import subprocess

        result = subprocess.run(
            ["antiword", path], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            html_parts = ['<div class="fp-word">']
            for para in result.stdout.split("\n"):
                if para.strip():
                    html_parts.append(f"<p>{_esc(para)}</p>")
            html_parts.append("</div>")
            return "".join(html_parts)
    except Exception:
        pass
    return _fallback_html(
        "老版 Word (.doc)",
        "该文件为旧版 Word 格式，当前环境无法内嵌预览。请下载原文件查看。",
    )


# ============================================================
# 统一入口 + 降级
# ============================================================
def parse_file(path: str) -> str:
    """按扩展名分发解析"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx" or ext == ".xls":
        return parse_excel(path)
    elif ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".docx":
        return parse_docx(path)
    elif ext == ".doc":
        return parse_doc(path)
    elif ext in (".txt", ".md", ".csv", ".json", ".jsonl"):
        return _parse_text(path)
    else:
        return _fallback_html("不支持预览", f"文件格式 {ext or '(无)'} 暂不支持内嵌预览，请下载原文件。")


def _parse_text(path: str, max_chars: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars)
        html_parts = ['<pre class="fp-text">']
        html_parts.append(_esc(text))
        html_parts.append("</pre>")
        return "".join(html_parts)
    except Exception as e:
        return _fallback_html("文本解析失败", str(e))


def _fallback_html(title: str, reason: str) -> str:
    """降级提示"""
    return (
        f'<div class="fp-fallback"><h4>⚠️ {_esc(title)}</h4>'
        f"<p>{_esc(reason)}</p>"
        '<p class="fp-dl-hint">可点击右上角「下载原文件」查看完整内容。</p></div>'
    )
