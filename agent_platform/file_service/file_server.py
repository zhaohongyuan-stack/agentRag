"""
文件服务 — 真实业务文件浏览/预览/下载/上传/解析嵌入 API

提供:
  GET  /api/files/tree           — 目录树（递归，用于制度文库左侧导航）
  GET  /api/files/list?dir=      — 指定目录的文件与子目录列表
  GET  /api/files/preview?path=  — 解析文件内容返回 HTML（内嵌预览）
  GET  /api/files/download?path= — 下载原文件
  GET  /api/files/search?q=      — 按文件名搜索
  POST /api/files/upload         — 上传文档到指定目录（base64 编码）
  POST /api/files/mkdir          — 创建文件夹（树状结构维护）
  POST /api/files/parse          — 触发自研解析器解析并嵌入到检索服务

安全: 所有 path 必须解析后位于 FILES_ROOT 之内，防止路径穿越。
"""

import base64
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import parsers

logger = logging.getLogger(__name__)

# 文件根目录（由环境变量配置，默认 ./data_files）
FILES_ROOT = Path(os.environ.get("FILES_ROOT", "/opt/ace-rag/data_files")).resolve()

# 常见业务文件扩展名（参与搜索/列表）
FILE_EXTS = {".xls", ".xlsx", ".pdf", ".doc", ".docx", ".txt", ".csv", ".md", ".json", ".jsonl"}

# 自研解析器脚本路径（解析 data_files 下的文档并生成 _chunks.jsonl）
PARSER_SCRIPT = Path(os.environ.get("PARSER_SCRIPT", "/opt/ace-rag/run_parser.py"))

# 检索服务 reload 接口（解析完成后触发索引重建）
RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "")

router = APIRouter(prefix="/api/files", tags=["files"])


def _safe_path(rel: str) -> Path:
    """将相对路径解析为 FILES_ROOT 内绝对路径，防路径穿越"""
    rel = unquote(rel or "").lstrip("/\\")
    target = (FILES_ROOT / rel).resolve()
    if not target.is_relative_to(FILES_ROOT):
        raise HTTPException(status_code=400, detail="非法的文件路径")
    return target


def _file_info(p: Path) -> Dict[str, Any]:
    stat = p.stat()
    return {
        "name": p.name,
        "path": str(p.relative_to(FILES_ROOT)).replace("\\", "/"),
        "size": stat.st_size,
        "size_human": _human_size(stat.st_size),
        "ext": p.suffix.lower(),
        "modified": stat.st_mtime,
    }


def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


@router.get("/tree")
def file_tree():
    """返回目录树（层级结构）"""
    def build(path: Path) -> Dict[str, Any]:
        node = {"name": path.name if path != FILES_ROOT else "文件库", "path": _rel(path), "dirs": [], "files": []}
        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return node
        for e in entries:
            if e.is_dir():
                node["dirs"].append(build(e))
            elif e.suffix.lower() in FILE_EXTS:
                node["files"].append(_file_info(e))
        return node

    return build(FILES_ROOT)


def _rel(path: Path) -> str:
    if path == FILES_ROOT:
        return ""
    return str(path.relative_to(FILES_ROOT)).replace("\\", "/")


@router.get("/list")
def file_list(dir: str = Query("", description="相对目录路径")):
    """列出指定目录的文件与子目录"""
    target = _safe_path(dir)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="目录不存在")
    dirs, files = [], []
    try:
        entries = sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        entries = []
    for e in entries:
        if e.is_dir():
            dirs.append({"name": e.name, "path": _rel(e), "count": _count_files(e)})
        elif e.suffix.lower() in FILE_EXTS:
            files.append(_file_info(e))
    return {"dir": _rel(target), "dirs": dirs, "files": files}


def _count_files(path: Path) -> int:
    return sum(1 for _ in path.rglob("*") if _.is_file() and _.suffix.lower() in FILE_EXTS)


@router.get("/preview", response_class=HTMLResponse)
def file_preview(path: str = Query(..., description="相对文件路径")):
    """解析文件内容返回 HTML 预览"""
    target = _safe_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    html_body = parsers.parse_file(str(target))
    title = _esc(target.name)
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>{title}</title><style>
body{{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:#f4f7fb;color:#1a202c;margin:0;padding:16px;font-size:13px}}
.fp-sheet{{color:#1a3a5c;font-weight:800;margin:10px 0 6px;border-bottom:2px solid #c9a961;padding-bottom:4px}}
.fp-table{{border-collapse:collapse;width:100%;margin-bottom:14px;background:#fff;border:1px solid #e2e8f0;font-size:12px}}
.fp-table th{{background:#1a3a5c;color:#fff;padding:6px 8px;text-align:left;white-space:nowrap;border:1px solid #2c5282}}
.fp-table td{{padding:5px 8px;border:1px solid #e2e8f0;white-space:nowrap}}
.fp-table tr:nth-child(even) td{{background:#f8fafc}}
.fp-more{{color:#a0aec0;font-style:italic;margin:6px 0}}
.fp-empty{{color:#a0aec0}}
.fp-pdf p,.fp-word p{{margin:6px 0;line-height:1.8;text-align:justify}}
.fp-text{{background:#fff;padding:12px;border:1px solid #e2e8f0;border-radius:6px;white-space:pre-wrap;font-size:12px;line-height:1.7}}
.fp-fallback{{background:#fffaf0;border:1px solid #e8dcc0;border-radius:8px;padding:20px;text-align:center}}
.fp-fallback h4{{color:#c9a961;margin:0 0 8px}}
.fp-fallback p{{color:#4a5568}}
.fp-dl-hint{{font-size:12px;color:#a0aec0}}
</style></head><body>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
  <div style="font-weight:800;color:#1a3a5c;font-size:14px">{title}</div>
  <div style="font-size:12px;color:#a0aec0">{_human_size(target.stat().st_size)}</div>
</div>
{html_body}
</body></html>"""


def _esc(text: str) -> str:
    import html as _h
    return _h.escape(str(text or ""))


@router.get("/download")
def file_download(path: str = Query(..., description="相对文件路径")):
    """下载原文件"""
    target = _safe_path(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=str(target),
        filename=target.name,
        media_type="application/octet-stream",
    )


@router.get("/search")
def file_search(q: str = Query("", description="搜索关键词")):
    """按文件名递归搜索"""
    q = (q or "").strip().lower()
    if not q:
        return {"items": []}
    results = []
    for p in FILES_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in FILE_EXTS and q in p.name.lower():
            results.append(_file_info(p))
            if len(results) >= 100:
                break
    return {"items": results}


# ============================================================
# 文档上传 — 业务人员在此放入文档，构成制度文库树状结构
# ============================================================
class FileUploadRequest(BaseModel):
    """文件上传请求（base64 编码，无需 python-multipart）"""
    filename: str                          # 文件名（含扩展名，不含路径）
    content: str                           # base64 编码的文件内容
    dir: str = ""                          # 目标子目录（相对 FILES_ROOT，不存在时自动创建）
    overwrite: bool = False                # 同名文件是否覆盖（默认拒绝，返回 409）


@router.post("/upload")
def file_upload(body: FileUploadRequest):
    """
    上传文档到制度文库指定目录

    业务流程:
      1. 校验文件名安全性（防路径穿越）
      2. 解析目标目录，不存在则自动创建（维护树状结构）
      3. 同名文件且 overwrite=False 时返回 409
      4. 写入文件，返回相对路径供后续 parse 调用
    """
    filename = (body.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=422, detail="文件名不能为空")

    # 文件名安全检查：不能包含路径分隔符或 ..
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="文件名包含非法字符（不允许路径分隔符）")

    ext = Path(filename).suffix.lower()
    if ext not in FILE_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的文件格式 {ext or '(无)'}，仅支持 {', '.join(sorted(FILE_EXTS))}",
        )

    # 解析目标目录（自动创建子目录，维护树状结构）
    target_dir = _safe_path(body.dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_dir.is_dir():
        raise HTTPException(status_code=500, detail="目标目录创建失败")

    target_file = target_dir / filename
    # 二次校验：最终路径必须仍在 FILES_ROOT 内
    if not target_file.resolve().is_relative_to(FILES_ROOT):
        raise HTTPException(status_code=400, detail="非法的目标路径")

    if target_file.exists() and not body.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"文件 {filename} 已存在，如需覆盖请设置 overwrite=true",
        )

    try:
        file_data = base64.b64decode(body.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件内容解码失败: {e}")

    try:
        with open(target_file, "wb") as f:
            f.write(file_data)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    rel_path = str(target_file.relative_to(FILES_ROOT)).replace("\\", "/")
    logger.info("[Files] 文件上传成功: %s (%d bytes, dir=%s)",
                filename, len(file_data), body.dir or "(root)")
    print(f"[Files] ✅ 上传成功: {rel_path} ({_human_size(len(file_data))})")

    return {
        "message": f"文件 {filename} 上传成功",
        "filename": filename,
        "path": rel_path,
        "size": len(file_data),
        "size_human": _human_size(len(file_data)),
        "ext": ext,
        "hint": "文件已加入制度文库，可点击「解析嵌入」将其纳入检索",
    }


# ============================================================
# 创建文件夹 — 维护制度文库树状结构
# ============================================================
class MkdirRequest(BaseModel):
    """创建文件夹请求"""
    dir: str                 # 父目录（相对 FILES_ROOT，空表示根目录）
    name: str                # 新文件夹名


@router.post("/mkdir")
def file_mkdir(body: MkdirRequest):
    """
    在制度文库中创建文件夹（支持多级嵌套）

    用于业务人员按机构/业务线/年度等维度组织文档树。
    """
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="文件夹名不能为空")

    # 文件夹名安全检查
    if "/" in name or "\\" in name or ".." in name or name in (".", ""):
        raise HTTPException(status_code=400, detail="文件夹名包含非法字符")

    parent = _safe_path(body.dir)
    if not parent.exists():
        raise HTTPException(status_code=404, detail=f"父目录不存在: {body.dir or '(root)'}")

    target = parent / name
    if not target.resolve().is_relative_to(FILES_ROOT):
        raise HTTPException(status_code=400, detail="非法的目标路径")

    if target.exists():
        raise HTTPException(status_code=409, detail=f"文件夹 {name} 已存在")

    try:
        target.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"文件夹创建失败: {e}")

    rel_path = str(target.relative_to(FILES_ROOT)).replace("\\", "/")
    logger.info("[Files] 文件夹创建成功: %s (parent=%s)", rel_path, body.dir or "(root)")
    print(f"[Files] 📁 新建文件夹: {rel_path}")

    return {
        "message": f"文件夹 {name} 创建成功",
        "name": name,
        "path": rel_path,
    }


# ============================================================
# 触发解析嵌入 — 自研解析器 → chunks.jsonl → 检索服务 reload
# ============================================================
class ParseRequest(BaseModel):
    """解析嵌入请求"""
    path: Optional[str] = None       # 指定文件相对路径；None 则解析整个文库
    force: bool = False              # 是否强制重新解析（覆盖已有 chunks）


@router.post("/parse")
def file_parse(body: ParseRequest):
    """
    触发自研解析器解析文档并嵌入到检索服务

    流程:
      1. 调用 PARSER_SCRIPT 解析指定文件/目录 → 生成 *_chunks.jsonl
         （解析器在 chunk metadata 中写入 source_path 字段，记录原始文件相对路径）
      2. 调用检索服务 /api/v1/reload 接口重建索引
      3. 返回解析结果摘要

    若 PARSER_SCRIPT 不存在或解析失败，返回明确错误（不静默吞掉）。
    """
    # 解析目标路径
    if body.path:
        target = _safe_path(body.path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"文件不存在: {body.path}")
        target_arg = str(target)
    else:
        target = FILES_ROOT
        target_arg = str(FILES_ROOT)

    print(f"[Files] 🔍 开始解析嵌入: {body.path or '(整个文库)'} (force={body.force})")

    # ── Step 1: 调用自研解析器 ──
    parse_result = {"chunks_generated": 0, "output_files": []}
    if PARSER_SCRIPT.exists():
        cmd = [
            "python3", str(PARSER_SCRIPT),
            "--input", target_arg,
            "--output-dir", str(FILES_ROOT / ".parsed"),
        ]
        if body.force:
            cmd.append("--force")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,   # 单文件解析上限 5 分钟
                cwd=str(FILES_ROOT.parent),
            )
            if proc.returncode != 0:
                logger.warning("[Files] 解析器返回非零退出码: %s", proc.stderr[-500:])
                print(f"[Files] ⚠️ 解析器警告: {proc.stderr[-200:]}")
            else:
                print(f"[Files] ✅ 解析器执行完成")
            # 收集生成的 chunks 文件
            parsed_dir = FILES_ROOT / ".parsed"
            if parsed_dir.exists():
                parse_result["output_files"] = [
                    str(p.relative_to(FILES_ROOT)).replace("\\", "/")
                    for p in parsed_dir.rglob("*_chunks.jsonl")
                ]
                parse_result["chunks_generated"] = len(parse_result["output_files"])
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="解析器执行超时（>5分钟），请检查文件大小或分批解析")
        except FileNotFoundError:
            logger.warning("[Files] python3 不可用，尝试 python")
            # 兜底：尝试 python
            cmd[0] = "python"
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                print(f"[Files] ✅ 解析器执行完成 (python)")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"解析器执行失败: {e}")
    else:
        # 解析器脚本不存在时，提示用户但不报错（允许仅上传不解析）
        logger.warning("[Files] 解析器脚本不存在: %s", PARSER_SCRIPT)
        print(f"[Files] ⚠️ 解析器脚本不存在: {PARSER_SCRIPT}，跳过解析步骤")
        parse_result["warning"] = f"解析器脚本不存在 ({PARSER_SCRIPT})，请配置后重试"

    # ── Step 2: 触发检索服务索引重建 ──
    reload_result = {"ok": False, "detail": ""}
    if RETRIEVAL_SERVICE_URL:
        try:
            from urllib.request import urlopen, Request
            import json as _json
            req = Request(
                f"{RETRIEVAL_SERVICE_URL.rstrip('/')}/api/v1/reload",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=_json.dumps({"force": body.force}).encode(),
            )
            with urlopen(req, timeout=60) as resp:
                reload_result = _json.loads(resp.read().decode())
                reload_result["ok"] = True
                print(f"[Files] ✅ 检索服务索引重建完成")
        except Exception as e:
            reload_result["detail"] = str(e)
            logger.warning("[Files] 检索服务 reload 失败: %s", e)
            print(f"[Files] ⚠️ 检索服务 reload 失败: {e}")
    else:
        reload_result["detail"] = "RETRIEVAL_SERVICE_URL 未配置"
        print(f"[Files] ⚠️ RETRIEVAL_SERVICE_URL 未配置，跳过索引重建")

    logger.info("[Files] 解析嵌入完成: path=%s, chunks=%d, reload_ok=%s",
                body.path or "(all)", parse_result["chunks_generated"], reload_result["ok"])

    return {
        "message": "解析嵌入流程已完成",
        "target": body.path or "(整个文库)",
        "parse": parse_result,
        "reload": reload_result,
        "hint": "现在可以在问答中检索到新文档的内容，且引用可溯源到原始文件",
    }


def _register_router(app) -> None:
    """供 server.py 调用，注册路由"""
    app.include_router(router)
