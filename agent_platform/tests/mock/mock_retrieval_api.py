"""
Mock Retrieval API — B组独立开发用的Mock检索服务

基于 FastAPI 实现，复刻A组的5个HTTP接口（路径和参数完全一致）：
  1. POST /api/v1/search          — 统一检索入口
  2. GET  /api/v1/chunks/{chunk_id} — 按ID取chunk
  3. GET  /api/v1/documents/{doc_id} — 按ID取文档
  4. GET  /api/v1/documents        — 列出文档
  5. POST /api/v1/chunks/search    — 多字段组合查chunk

使用 DataLoader 加载 contracts/examples/ 下的样例数据，
使用 ScenarioRouter 根据请求的 scenario 字段或 query 内容路由到不同场景。

启动方式：
    python -m agent_platform.tests.mock.mock_retrieval_api
    或设置环境变量 MOCK_PORT=8002 python -m agent_platform.tests.mock.mock_retrieval_api

默认端口: 8001（通过环境变量 MOCK_PORT 配置）
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .data_loader import DataLoader
from .scenario_router import MockTimeoutError, ScenarioRouter

# ============================================================
# 请求体模型（与A组 server.py 完全一致，额外增加 Mock 专用 scenario 字段）
# ============================================================
class SearchRequest(BaseModel):
    """
    统一检索请求体

    与A组 server.py 的 SearchRequest 字段完全一致，
    额外增加 scenario 字段用于 Mock 场景路由（A组无此字段，默认 None 不影响兼容性）。
    """

    query: str
    strategy: str = "hybrid"
    top_k: int = 10
    filters: dict = {}
    expand_context: bool = False
    # Mock 专用：场景标识，用于 ScenarioRouter 路由（A组无此字段）
    scenario: Optional[str] = None


class ChunkSearchRequest(BaseModel):
    """多字段组合查询请求体"""

    doc_id: Optional[str] = None
    chunk_type: Optional[str] = None
    table_name: Optional[str] = None
    clause_number: Optional[str] = None
    chapter_number: Optional[str] = None
    limit: int = 20


# ============================================================
# 内置 Mock 文档数据
# ============================================================
MOCK_DOCUMENTS: List[dict] = [
    {
        "doc_id": "doc-400",
        "doc_name": "商业银行资本管理办法",
        "doc_title": "商业银行资本管理办法",
        "parser_type": "pdf_clause",
        "source_file": "商业银行资本管理办法.pdf",
        "parse_timestamp": "2025-06-15T08:30:00",
        "attachment_no": "正文",
        "applicable_scope": "全部",
    },
    {
        "doc_id": "doc-001",
        "doc_name": "2026年银行业总资产、总负债（月度）",
        "doc_title": "2026年银行业总资产、总负债（月度）",
        "parser_type": "excel",
        "source_file": "001_2026年银行业总资产、总负债（月度）.xls",
        "parse_timestamp": "2026-07-22T00:37:02",
        "attachment_no": "附件1",
        "applicable_scope": "全部",
    },
]

# ============================================================
# 内置 Mock Chunk 数据
# ============================================================
MOCK_CHUNKS: List[dict] = [
    # ── clause 类型 ──
    {
        "chunk_id": "400_c43",
        "doc_id": "doc-400",
        "chunk_type": "clause",
        "content": (
            "第四十三条 商业银行各级资本充足率不得低于如下最低要求："
            "（一）核心一级资本充足率不得低于5%。"
            "（二）一级资本充足率不得低于6%。"
            "（三）资本充足率不得低于8%。"
        ),
        "hierarchy_path": "商业银行资本管理办法 > 第七章 资本充足率监管要求 > 第四十三条",
        "parent_chunk_id": "400_ch7",
        "prev_chunk_id": "400_c42",
        "next_chunk_id": "400_c44",
        "chapter_number": "第七章",
        "clause_number": "第四十三条",
        "subclause_number": "",
        "applicable_scope": "全部",
        "normative_level": "obligatory",
        "capital_tool_level": "",
        "table_name": "",
        "table_section_name": "",
        "sheet_name": "",
        "glossary_term": "",
        "keywords": ["资本充足率", "核心一级资本充足率", "一级资本充足率", "最低要求", "5%", "6%", "8%"],
        "evidence_snippet": "核心一级资本充足率不得低于5%；一级资本充足率不得低于6%；资本充足率不得低于8%。",
        "content_raw": (
            "第四十三条　商业银行各级资本充足率不得低于如下最低要求：\n"
            "（一）核心一级资本充足率不得低于5%。\n"
            "（二）一级资本充足率不得低于6%。\n"
            "（三）资本充足率不得低于8%。"
        ),
        "sub_chunks": [],
        "metadata": {
            "parser_type": "pdf_clause",
            "parser_version": "1.2.0",
            "parse_timestamp": "2025-06-15T08:30:00Z",
            "source_url": "https://www.nfra.gov.cn/...",
            "sha256": "a3f5c9e1b7d2...",
            "attachment_no": "正文",
            "applicable_scope": "全部",
            "chapter_number": "第七章",
            "clause_number": "第四十三条",
            "normative_level": "obligatory",
            "numeric_conditions": [
                {"metric": "核心一级资本充足率", "operator": ">=", "value": 5, "unit": "%"},
                {"metric": "一级资本充足率", "operator": ">=", "value": 6, "unit": "%"},
                {"metric": "资本充足率", "operator": ">=", "value": 8, "unit": "%"},
            ],
            "keywords": ["资本充足率", "核心一级资本充足率", "一级资本充足率", "最低要求", "5%", "6%", "8%"],
            "cross_attachment_refs": [],
            "cross_table_refs": [],
        },
    },
    {
        "chunk_id": "400_c44",
        "doc_id": "doc-400",
        "chunk_type": "clause",
        "content": (
            "第四十四条 商业银行应当在最低资本要求的基础上计提储备资本，"
            "储备资本要求为风险加权资产的2.5%。"
            "在最低资本要求和储备资本要求之上计提逆周期资本，"
            "逆周期资本要求为风险加权资产的0至2.5%。"
        ),
        "hierarchy_path": "商业银行资本管理办法 > 第七章 资本充足率监管要求 > 第四十四条",
        "parent_chunk_id": "400_ch7",
        "prev_chunk_id": "400_c43",
        "next_chunk_id": "400_c45",
        "chapter_number": "第七章",
        "clause_number": "第四十四条",
        "subclause_number": "",
        "applicable_scope": "全部",
        "normative_level": "obligatory",
        "capital_tool_level": "",
        "table_name": "",
        "table_section_name": "",
        "sheet_name": "",
        "glossary_term": "",
        "keywords": ["储备资本", "逆周期资本", "2.5%", "风险加权资产"],
        "evidence_snippet": "储备资本要求为风险加权资产的2.5%；逆周期资本要求为风险加权资产的0至2.5%。",
        "content_raw": (
            "第四十四条　商业银行应当在最低资本要求的基础上计提储备资本，"
            "储备资本要求为风险加权资产的2.5%。"
        ),
        "sub_chunks": [],
        "metadata": {
            "parser_type": "pdf_clause",
            "parser_version": "1.2.0",
            "parse_timestamp": "2025-06-15T08:30:00Z",
            "source_url": "https://www.nfra.gov.cn/...",
            "sha256": "b4e6d0f2c8e3...",
            "attachment_no": "正文",
            "applicable_scope": "全部",
            "chapter_number": "第七章",
            "clause_number": "第四十四条",
            "normative_level": "obligatory",
            "numeric_conditions": [
                {"metric": "储备资本", "operator": "==", "value": 2.5, "unit": "%"},
                {"metric": "逆周期资本", "operator": "range", "value_min": 0, "value_max": 2.5, "unit": "%"},
            ],
            "keywords": ["储备资本", "逆周期资本", "2.5%", "风险加权资产"],
            "cross_attachment_refs": [],
            "cross_table_refs": [],
        },
    },
    {
        "chunk_id": "400_c45",
        "doc_id": "doc-400",
        "chunk_type": "clause",
        "content": (
            "第四十五条 系统重要性银行应当计提附加资本，"
            "国内系统重要性银行附加资本要求由中国人民银行、国家金融监督管理总局确定。"
        ),
        "hierarchy_path": "商业银行资本管理办法 > 第七章 资本充足率监管要求 > 第四十五条",
        "parent_chunk_id": "400_ch7",
        "prev_chunk_id": "400_c44",
        "next_chunk_id": "400_c46",
        "chapter_number": "第七章",
        "clause_number": "第四十五条",
        "subclause_number": "",
        "applicable_scope": "系统重要性银行",
        "normative_level": "obligatory",
        "capital_tool_level": "",
        "table_name": "",
        "table_section_name": "",
        "sheet_name": "",
        "glossary_term": "",
        "keywords": ["系统重要性银行", "附加资本", "中国人民银行", "国家金融监督管理总局"],
        "evidence_snippet": "系统重要性银行应当计提附加资本。",
        "content_raw": (
            "第四十五条　系统重要性银行应当计提附加资本，"
            "国内系统重要性银行附加资本要求由中国人民银行、国家金融监督管理总局确定。"
        ),
        "sub_chunks": [],
        "metadata": {
            "parser_type": "pdf_clause",
            "parser_version": "1.2.0",
            "parse_timestamp": "2025-06-15T08:30:00Z",
            "source_url": "https://www.nfra.gov.cn/...",
            "sha256": "c5f7e1a3d9b4...",
            "attachment_no": "正文",
            "applicable_scope": "系统重要性银行",
            "chapter_number": "第七章",
            "clause_number": "第四十五条",
            "normative_level": "obligatory",
            "numeric_conditions": [],
            "keywords": ["系统重要性银行", "附加资本", "中国人民银行", "国家金融监督管理总局"],
            "cross_attachment_refs": ["附件8"],
            "cross_table_refs": [],
        },
    },
    # ── cell_fact 类型 ──
    {
        "chunk_id": "001_t1_r1",
        "doc_id": "doc-001",
        "chunk_type": "cell_fact",
        "content": "2026年1月 银行业金融机构总资产：412.56万亿元，同比增长7.8%",
        "hierarchy_path": "2026年银行业总资产、总负债（月度） > 表1 银行业金融机构 > 第1行",
        "parent_chunk_id": "001_t1_header",
        "prev_chunk_id": "",
        "next_chunk_id": "001_t1_r2",
        "chapter_number": "",
        "clause_number": "",
        "subclause_number": "",
        "applicable_scope": "全部",
        "normative_level": "neutral",
        "capital_tool_level": "",
        "table_name": "1. 银行业金融机构",
        "table_section_name": "总资产",
        "sheet_name": "Sheet1",
        "glossary_term": "",
        "keywords": ["银行业", "总资产", "2026年1月", "412.56万亿"],
        "evidence_snippet": "2026年1月银行业总资产412.56万亿元",
        "content_raw": "2026年1月 银行业金融机构总资产：412.56万亿元，同比增长7.8%",
        "sub_chunks": [],
        "metadata": {
            "parser_type": "excel",
            "parser_version": "1.1.0",
            "parse_timestamp": "2026-07-22T00:37:02",
            "source_url": "",
            "sha256": "d6f8e2b4a0c5...",
            "attachment_no": "附件1",
            "applicable_scope": "全部",
            "table_name": "1. 银行业金融机构",
            "table_full_name": "表1 银行业金融机构总资产、总负债（月度）",
            "table_section_name": "总资产",
            "sheet_name": "Sheet1",
            "row_count": 12,
            "col_count": 4,
            "merge_info": [],
            "cross_refs": [],
            "numeric_conditions": [
                {"metric": "银行业总资产", "operator": "==", "value": 412.56, "unit": "万亿元"},
            ],
            "keywords": ["银行业", "总资产", "2026年1月", "412.56万亿"],
            "cross_attachment_refs": [],
            "cross_table_refs": [],
        },
    },
    {
        "chunk_id": "001_t1_r2",
        "doc_id": "doc-001",
        "chunk_type": "cell_fact",
        "content": "2026年1月 银行业金融机构总负债：378.92万亿元，同比增长7.5%",
        "hierarchy_path": "2026年银行业总资产、总负债（月度） > 表1 银行业金融机构 > 第2行",
        "parent_chunk_id": "001_t1_header",
        "prev_chunk_id": "001_t1_r1",
        "next_chunk_id": "001_t1_r3",
        "chapter_number": "",
        "clause_number": "",
        "subclause_number": "",
        "applicable_scope": "全部",
        "normative_level": "neutral",
        "capital_tool_level": "",
        "table_name": "1. 银行业金融机构",
        "table_section_name": "总负债",
        "sheet_name": "Sheet1",
        "glossary_term": "",
        "keywords": ["银行业", "总负债", "2026年1月", "378.92万亿"],
        "evidence_snippet": "2026年1月银行业总负债378.92万亿元",
        "content_raw": "2026年1月 银行业金融机构总负债：378.92万亿元，同比增长7.5%",
        "sub_chunks": [],
        "metadata": {
            "parser_type": "excel",
            "parser_version": "1.1.0",
            "parse_timestamp": "2026-07-22T00:37:02",
            "source_url": "",
            "sha256": "e7f9e3c5b1d6...",
            "attachment_no": "附件1",
            "applicable_scope": "全部",
            "table_name": "1. 银行业金融机构",
            "table_full_name": "表1 银行业金融机构总资产、总负债（月度）",
            "table_section_name": "总负债",
            "sheet_name": "Sheet1",
            "row_count": 12,
            "col_count": 4,
            "merge_info": [],
            "cross_refs": [],
            "numeric_conditions": [
                {"metric": "银行业总负债", "operator": "==", "value": 378.92, "unit": "万亿元"},
            ],
            "keywords": ["银行业", "总负债", "2026年1月", "378.92万亿"],
            "cross_attachment_refs": [],
            "cross_table_refs": [],
        },
    },
    {
        "chunk_id": "001_t1_r3",
        "doc_id": "doc-001",
        "chunk_type": "cell_fact",
        "content": "2026年2月 银行业金融机构总资产：418.23万亿元，同比增长8.1%",
        "hierarchy_path": "2026年银行业总资产、总负债（月度） > 表1 银行业金融机构 > 第3行",
        "parent_chunk_id": "001_t1_header",
        "prev_chunk_id": "001_t1_r2",
        "next_chunk_id": "",
        "chapter_number": "",
        "clause_number": "",
        "subclause_number": "",
        "applicable_scope": "全部",
        "normative_level": "neutral",
        "capital_tool_level": "",
        "table_name": "1. 银行业金融机构",
        "table_section_name": "总资产",
        "sheet_name": "Sheet1",
        "glossary_term": "",
        "keywords": ["银行业", "总资产", "2026年2月", "418.23万亿"],
        "evidence_snippet": "2026年2月银行业总资产418.23万亿元",
        "content_raw": "2026年2月 银行业金融机构总资产：418.23万亿元，同比增长8.1%",
        "sub_chunks": [],
        "metadata": {
            "parser_type": "excel",
            "parser_version": "1.1.0",
            "parse_timestamp": "2026-07-22T00:37:02",
            "source_url": "",
            "sha256": "f8a0f4d6c2e7...",
            "attachment_no": "附件1",
            "applicable_scope": "全部",
            "table_name": "1. 银行业金融机构",
            "table_full_name": "表1 银行业金融机构总资产、总负债（月度）",
            "table_section_name": "总资产",
            "sheet_name": "Sheet1",
            "row_count": 12,
            "col_count": 4,
            "merge_info": [],
            "cross_refs": [],
            "numeric_conditions": [
                {"metric": "银行业总资产", "operator": "==", "value": 418.23, "unit": "万亿元"},
            ],
            "keywords": ["银行业", "总资产", "2026年2月", "418.23万亿"],
            "cross_attachment_refs": [],
            "cross_table_refs": [],
        },
    },
    # ── glossary 类型（补充）──
    {
        "chunk_id": "400_g1",
        "doc_id": "doc-400",
        "chunk_type": "glossary",
        "content": (
            "系统重要性银行：指因规模较大、结构和业务复杂度较高、与其他金融机构关联性较强，"
            "一旦发生重大风险事件而无法持续经营，可能对金融体系和实体经济产生重大不利影响的银行业金融机构。"
        ),
        "hierarchy_path": "商业银行资本管理办法 > 术语定义 > 系统重要性银行",
        "parent_chunk_id": "400_glossary_section",
        "prev_chunk_id": "",
        "next_chunk_id": "",
        "chapter_number": "",
        "clause_number": "",
        "subclause_number": "",
        "applicable_scope": "全部",
        "normative_level": "definitional",
        "capital_tool_level": "",
        "table_name": "",
        "table_section_name": "",
        "sheet_name": "",
        "glossary_term": "系统重要性银行",
        "keywords": ["系统重要性银行", "定义", "金融稳定"],
        "evidence_snippet": "系统重要性银行定义：规模较大、结构复杂、关联性强的银行业金融机构",
        "content_raw": (
            "系统重要性银行：指因规模较大、结构和业务复杂度较高、与其他金融机构关联性较强，"
            "一旦发生重大风险事件而无法持续经营，可能对金融体系和实体经济产生重大不利影响的银行业金融机构。"
        ),
        "sub_chunks": [],
        "metadata": {
            "parser_type": "pdf_clause",
            "parser_version": "1.2.0",
            "parse_timestamp": "2025-06-15T08:30:00Z",
            "source_url": "https://www.nfra.gov.cn/...",
            "sha256": "g9b1a5e7d3f8...",
            "attachment_no": "正文",
            "applicable_scope": "全部",
            "glossary_term": "系统重要性银行",
            "glossary_definition": "指因规模较大、结构和业务复杂度较高...",
            "glossary_term_number": "1",
            "normative_level": "definitional",
            "numeric_conditions": [],
            "keywords": ["系统重要性银行", "定义", "金融稳定"],
            "cross_attachment_refs": [],
            "cross_table_refs": [],
        },
    },
]


# ============================================================
# 内部工具：chunk DB格式 → RetrievalHit dict 格式
# ============================================================
def _build_doc_lookup() -> Dict[str, dict]:
    """构建 doc_id → document 的查找表"""
    return {doc["doc_id"]: doc for doc in MOCK_DOCUMENTS}


_DOC_LOOKUP = _build_doc_lookup()


def _chunk_to_hit(chunk: dict, rank: int = 1, score: float = 0.9) -> dict:
    """
    将 chunk（DB行格式）转换为 RetrievalHit dict 格式

    对齐 A组 retrieval_request.py 中 RetrievalHit.to_dict() 的输出结构。

    Args:
        chunk: chunk 字典（DB 行格式）
        rank: 排名
        score: 综合得分

    Returns:
        RetrievalHit dict
    """
    doc = _DOC_LOOKUP.get(chunk.get("doc_id", ""), {})
    doc_name = doc.get("doc_name", "")
    doc_title = doc.get("doc_title", "")
    source_file = doc.get("source_file", "")

    # 构建 citation（LLM 引用格式）
    citation_parts = [f"《{doc_name}》"] if doc_name else []
    clause = chunk.get("clause_number", "")
    if clause:
        citation_parts.append(clause)
    citation = " ".join(citation_parts)

    # 构建 scores_detail
    scores_detail = {
        "bm25": round(score * 12.0, 4),
        "dense": round(score * 0.9, 4),
        "rrf": round(score * 0.03, 6),
    }

    # 构建 trace
    trace = {
        "strategy": "hybrid",
        "bm25_rank": rank,
        "bm25_score": scores_detail["bm25"],
        "dense_rank": rank,
        "dense_score": scores_detail["dense"],
        "rrf_score": scores_detail["rrf"],
        "filters_applied": {},
    }

    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "chunk_type": chunk.get("chunk_type", "clause"),
        "doc_id": chunk.get("doc_id", ""),
        "doc_name": doc_name,
        "doc_title": doc_title,
        "hierarchy_path": chunk.get("hierarchy_path", ""),
        "source": source_file,
        "citation": citation,
        "content": chunk.get("content", ""),
        "content_raw": chunk.get("content_raw", ""),
        "evidence_snippet": chunk.get("evidence_snippet", ""),
        "score": score,
        "scores_detail": scores_detail,
        "rank": rank,
        "matched_by": ["bm25", "dense"],
        "trace": trace,
        "context": None,
        "metadata": chunk.get("metadata", {}),
    }


def _build_default_hits() -> List[dict]:
    """从内置 Mock chunks 构建默认检索结果（RetrievalHit 格式）"""
    hits = []
    for i, chunk in enumerate(MOCK_CHUNKS, 1):
        score = 0.95 - (i - 1) * 0.08
        hits.append(_chunk_to_hit(chunk, rank=i, score=score))
    return hits


# ============================================================
# 初始化 DataLoader 和 ScenarioRouter
# ============================================================
_data_loader = DataLoader()
_default_hits = _build_default_hits()
_scenario_router = ScenarioRouter(
    data_loader=_data_loader,
    default_hits=_default_hits,
)


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="Mock Retrieval API",
    description="B组独立开发用的Mock检索服务，复刻A组5个HTTP接口",
    version="1.0-mock",
)


@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "mock-retrieval-api", "version": "1.0-mock"}


# ============================================================
# 接口一：POST /api/v1/search — 统一检索入口
# ============================================================
@app.post("/api/v1/search")
def search(req: SearchRequest):
    """
    统一检索入口，返回 RetrievalHit 列表

    使用 ScenarioRouter 根据请求的 scenario 字段或 query 内容路由：
      - normal 场景：返回默认检索结果
      - empty 场景：返回空数组 []
      - timeout 场景：返回 504 状态码
      - version_conflict 场景：返回带版本冲突标记的结果
      - partial_failure 场景：返回带部分失败标记的结果
    """
    request_dict = req.model_dump()
    try:
        hits = _scenario_router.route(request_dict)
        return hits
    except MockTimeoutError as e:
        return JSONResponse(
            status_code=504,
            content={
                "error": "timeout",
                "message": str(e),
                "detail": "请求匹配到 timeout 场景，Mock 检索服务模拟超时",
                "scenario": "timeout",
                "timestamp": time.time(),
            },
        )


# ============================================================
# 接口二：GET /api/v1/chunks/{chunk_id} — 按ID取chunk
# ============================================================
@app.get("/api/v1/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    """按 chunk_id 查单条 chunk，找不到返回空对象 {}"""
    for chunk in MOCK_CHUNKS:
        if chunk["chunk_id"] == chunk_id:
            return chunk
    return {}


# ============================================================
# 接口三：GET /api/v1/documents/{doc_id} — 按ID取文档
# ============================================================
@app.get("/api/v1/documents/{doc_id}")
def get_document(doc_id: str):
    """按 doc_id 查文档元信息，找不到返回空对象 {}"""
    for doc in MOCK_DOCUMENTS:
        if doc["doc_id"] == doc_id:
            return doc
    return {}


# ============================================================
# 接口四：GET /api/v1/documents — 列出文档
# ============================================================
@app.get("/api/v1/documents")
def list_documents(limit: int = Query(default=20, ge=1, le=200)):
    """列出文档，支持 limit 参数限制返回条数（1~200）"""
    docs = copy.deepcopy(MOCK_DOCUMENTS)
    return docs[:limit]


# ============================================================
# 接口五：POST /api/v1/chunks/search — 多字段组合查chunk
# ============================================================
@app.post("/api/v1/chunks/search")
def search_chunks(req: ChunkSearchRequest):
    """
    多字段组合查 chunk，所有条件 AND 关联

    支持的过滤字段：doc_id, chunk_type, table_name, clause_number, chapter_number
    """
    results = []
    for chunk in MOCK_CHUNKS:
        # 逐字段 AND 过滤
        if req.doc_id is not None and chunk.get("doc_id") != req.doc_id:
            continue
        if req.chunk_type is not None and chunk.get("chunk_type") != req.chunk_type:
            continue
        if req.table_name is not None and chunk.get("table_name") != req.table_name:
            continue
        if req.clause_number is not None and chunk.get("clause_number") != req.clause_number:
            continue
        if req.chapter_number is not None and chunk.get("chapter_number") != req.chapter_number:
            continue
        results.append(chunk)

    return results[:req.limit]


# ============================================================
# 直接启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MOCK_PORT", 8001))
    host = os.environ.get("MOCK_HOST", "127.0.0.1")
    print("=" * 60)
    print(f"  Mock Retrieval API 启动中...")
    print(f"  地址: http://{host}:{port}")
    print(f"  文档: http://{host}:{port}/docs")
    print(f"  健康检查: http://{host}:{port}/health")
    print(f"  内置文档数: {len(MOCK_DOCUMENTS)}")
    print(f"  内置Chunk数: {len(MOCK_CHUNKS)}")
    print(f"  DataLoader: {_data_loader}")
    print(f"  ScenarioRouter: {_scenario_router}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
