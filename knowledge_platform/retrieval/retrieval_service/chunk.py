"""
数据结构 + JSON Chunk 加载

支持的 JSON 格式:
  - .jsonl 文件（每行一个 JSON chunk，推荐）
  - .json 数组文件（整个文件是 chunk 数组）
  - 目录（递归扫描 .jsonl 和 .json 文件）
"""

import json
from pathlib import Path
from typing import List, Optional, Union


# ============================================================
# 数据结构
# ============================================================
class Chunk:
    """文档片段（clause / subclause / item / glossary / body / table / table_section / excel_sheet / excel_range）"""

    __slots__ = (
        "chunk_id", "chunk_type", "content", "hierarchy_path",
        "source_file", "doc_id", "doc_name", "doc_title",
        "metadata",
    )

    def __init__(self,
                 chunk_id: str = "",
                 chunk_type: str = "clause",
                 content: str = "",
                 hierarchy_path: str = "",
                 source_file: str = "",
                 doc_id: str = "",
                 doc_name: str = "",
                 doc_title: str = "",
                 metadata: Optional[dict] = None):
        self.chunk_id = chunk_id
        self.chunk_type = chunk_type
        self.content = content
        self.hierarchy_path = hierarchy_path
        self.source_file = source_file
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.doc_title = doc_title
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Chunk({self.chunk_type}: {self.chunk_id[:40]}...)"


# ============================================================
# 索引文本构造
# ============================================================
def build_index_text(raw: dict) -> str:
    """
    构造检索用文本。
    v1.2 约定: content 字段已含精简上下文前插（最近2级父级 + 原文），
    直接使用 content 作为检索输入，不再重复拼接 hierarchy_path。
    """
    return raw.get("content", "") or raw.get("hierarchy_path", "") or ""


# ============================================================
# 元数据展平
# ============================================================
# ── 统一 schema 中已定义的 metadata 字段名（用于区分标准字段 vs 格式专属字段）──
_STANDARD_META_KEYS = frozenset({
    # 文档级（约定 3.1）
    "doc_name", "doc_id", "source_url", "sha256", "source_title", "column",
    "parse_timestamp", "parser_version", "parser_type",
    # 结构级（约定 3.2）
    "attachment_no", "applicable_scope", "parent_section",
    "chapter_number", "clause_number", "subclause_number",
    "capital_tool_level", "context_chunk_id",
    "glossary_term", "glossary_definition", "glossary_term_number",
    # 语义级（约定 3.3）
    "normative_level", "numeric_conditions", "keywords",
    "cross_attachment_refs", "cross_table_refs",
    # 表格专属（约定 3.4）
    "table_name", "table_full_name", "row_count", "col_count",
    "merge_info", "cross_refs", "table_section_name",
    "sheet_name",
    # 别名字段（需归一化处理，不作为最终 key）
    "file_type",  # Excel 解析器用的替代名 → 映射为 parser_type
    # docx 解析器额外产出的辅助字段
    "block_title",
})


def flatten_metadata(raw: dict) -> dict:
    """
    将嵌套的 metadata 展开为顶层字段，方便过滤。
    对齐 v1.2 chunk 约定规范。

    支持 Word/PDF/Excel 三种解析器产出：
      - 字段名归一化（如 Excel 的 file_type → parser_type）
      - 格式专属字段保留在 _extra 中，不丢弃
    """
    meta = raw.get("metadata", {})
    index_text = build_index_text(raw)

    # ── 字段名归一化 ──
    parser_type = meta.get("parser_type") or meta.get("file_type") or ""
    parser_version = meta.get("parser_version", "")
    parse_timestamp = meta.get("parse_timestamp", "")

    # ── 收集非标准字段（各格式专属），不丢弃 ──
    _extra = {
        k: v for k, v in meta.items()
        if k not in _STANDARD_META_KEYS
    }

    return {
        "chunk_id":       raw.get("chunk_id", ""),
        "chunk_type":     raw.get("chunk_type", "clause"),
        "content":        index_text,
        "hierarchy_path": raw.get("hierarchy_path", ""),
        "parent_chunk_id": raw.get("parent_chunk_id") or meta.get("parent_chunk_id", ""),
        "source_file":    meta.get("source_title") or meta.get("doc_name", ""),
        "doc_id":         str(meta.get("doc_id", "")),
        "doc_name":       meta.get("doc_name", ""),
        "doc_title":      meta.get("source_title") or meta.get("doc_name", ""),
        "_meta": {
            # ── 检索相关（顶层冗余到 metadata，方便统一取用）──
            "content":          index_text,
            "content_raw":      raw.get("content_raw", "") or raw.get("content", "") or "",
            "content_markdown": raw.get("content_markdown") or "",
            "content_json":     raw.get("content_json") or {},
            "evidence_snippet": raw.get("evidence_snippet", ""),
            "sub_chunks":       raw.get("sub_chunks", []) or [],
            # ── 文档级元数据（约定 3.1）──
            "parser_type":      parser_type,
            "parser_version":   parser_version,
            "parse_timestamp":  parse_timestamp,
            "source_url":       meta.get("source_url", ""),
            "sha256":           meta.get("sha256", ""),
            "column":           meta.get("column", ""),
            # ── 结构级元数据（约定 3.2）──
            "attachment_no":        meta.get("attachment_no", ""),
            "applicable_scope":     meta.get("applicable_scope", "未指定"),
            "parent_section":       meta.get("parent_section", ""),
            "chapter_number":       meta.get("chapter_number", ""),
            "clause_number":        meta.get("clause_number", ""),
            "subclause_number":     meta.get("subclause_number", ""),
            "capital_tool_level":   meta.get("capital_tool_level", ""),
            "context_chunk_id":     meta.get("context_chunk_id", ""),
            # ── 术语（约定 3.2，仅 glossary 类型）──
            "glossary_term":        meta.get("glossary_term", ""),
            "glossary_definition":  meta.get("glossary_definition", ""),
            "glossary_term_number": meta.get("glossary_term_number", ""),
            # ── 语义级元数据（约定 3.3）──
            "normative_level":        meta.get("normative_level", "neutral"),
            "numeric_conditions":     meta.get("numeric_conditions", []),
            "keywords":               meta.get("keywords", []),
            "cross_attachment_refs":  meta.get("cross_attachment_refs", []),
            "cross_table_refs":       meta.get("cross_table_refs", []),
            # ── 表格专属元数据（约定 3.4）──
            "table_name":         meta.get("table_name", ""),
            "table_full_name":    meta.get("table_full_name", ""),
            "table_section_name": meta.get("table_section_name", ""),
            "sheet_name":         meta.get("sheet_name", ""),
            "row_count":          meta.get("row_count", 0),
            "col_count":          meta.get("col_count", 0),
            "merge_info":         meta.get("merge_info", []),
            "cross_refs":         meta.get("cross_refs", []),
            # ── 格式专属字段（Excel: range/unit/metric_group, Docx: block_title 等）──
            "_extra": _extra,
        },
    }


# ============================================================
# 加载入口
# ============================================================
def load_json_chunks(source: Union[str, Path]) -> List[Chunk]:
    """
    加载 JSON chunk 文件。

    支持三种格式：
      - .jsonl 文件（每行一个 JSON chunk，推荐）
      - .json 数组文件（整个文件是 chunk 数组）
      - 目录（递归扫描 .jsonl 和 .json 文件）

    返回 Chunk 列表。
    """
    source_path = Path(source)
    chunks: List[Chunk] = []

    if source_path.is_dir():
        files = sorted(source_path.rglob("*_chunks.jsonl")) + sorted(source_path.rglob("*_chunks.json"))
        # 排除隐藏目录（如 .cache），避免加载导出/缓存文件
        files = [f for f in files if not any(p.startswith('.') for p in f.parts)]
        for fpath in files:
            _load_file(fpath, chunks)
    else:
        _load_file(source_path, chunks)

    print(f"  [加载] 共 {len(chunks)} 个 chunks")
    return chunks


def _load_file(fpath: Path, chunks: List[Chunk]):
    """加载单个 .jsonl 或 .json 文件"""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return

        if fpath.suffix.lower() == ".jsonl":
            _load_jsonl(content, chunks)
        else:
            _load_json(content, chunks)
    except Exception as e:
        print(f"  [跳过] 无法加载 {fpath.name}: {e}")


def _load_jsonl(content: str, chunks: List[Chunk]):
    """解析 JSON Lines 格式"""
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            _append_chunk(obj, chunks)
        except json.JSONDecodeError:
            continue


def _load_json(content: str, chunks: List[Chunk]):
    """解析 JSON 数组或 {"chunks": [...]} 格式"""
    obj = json.loads(content)
    items = obj if isinstance(obj, list) else obj.get("chunks", [obj])
    if not isinstance(items, list):
        items = [items]
    for item in items:
        if not isinstance(item, dict):
            continue
        _append_chunk(item, chunks)


def _append_chunk(raw: dict, chunks: List[Chunk]):
    """将一条原始记录转为 Chunk 并追加到列表"""
    flat = flatten_metadata(raw)
    chunks.append(Chunk(
        chunk_id=flat["chunk_id"],
        chunk_type=flat["chunk_type"],
        content=flat["content"],
        hierarchy_path=flat["hierarchy_path"],
        source_file=flat["source_file"],
        doc_id=flat["doc_id"],
        doc_name=flat["doc_name"],
        doc_title=flat["doc_title"],
        metadata=flat["_meta"],
    ))


# ============================================================
# 共享常量 — chunk_type → emoji 图标，统一用于 LLM 展示
# ============================================================
CHUNK_TYPE_ICONS = {
    "clause":         "📜",   # 法规条款
    "subclause":      "📑",   # 子条款
    "item":           "🔹",   # 款/项
    "glossary":       "📖",   # 术语定义
    "body":           "📄",   # 正文
    "table":          "📊",   # 表格结构
    "cell_fact":      "📈",   # 数据单元格
    "note":           "📝",   # 注释/报表说明
    "report_summary": "📋",   # 报告摘要
    "table_section":  "📑",   # 表格章节
    "excel_sheet":    "📋",   # Excel 工作表
    "excel_range":    "🔢",   # Excel 区域
}
