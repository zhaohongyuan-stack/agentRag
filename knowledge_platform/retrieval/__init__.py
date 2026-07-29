"""
retrieval_service — 法规知识库统一检索服务

A 组交付物：稳定、可过滤、可追溯、可评测的检索能力层。
A 组只负责"提供能力"，不负责判断某个用户问题应该调用哪些通道。
路由策略属于 B 组。

============================================================
快速开始
============================================================

    from retrieval_service import RetrievalAPI

    # 方式一：从 JSON 文件加载（构建全部 7 路索引 + 入库）
    api = RetrievalAPI()
    api.load("regulatory_docs/")

    # 混合检索（便捷入口）
    results = api.search("核心一级资本合格标准", top_k=5)

    # 单路检索 — 每个通道独立可调用
    api.lexical.search("不良贷款率", top_k=10)
    api.dense.search("不良贷款率", top_k=10)
    api.exact.search("第十二条", top_k=5)

    # 上下文 & 表格
    ctx = api.neighborhood.get_context("chunk_0042")
    rows = api.table.find_rows("KM1", "核心一级")

    # === 评测 ===
    from retrieval_service import RetrievalEval
    evaluator = RetrievalEval()
    evaluator.load_golden_set("golden_set.jsonl")
    report = evaluator.evaluate(api.search)

============================================================
模块清单（13 个模块，4 层架构）
============================================================

迁移层：
    migration        — JSON ↔ 数据库 导入/导出

数据结构层：
    chunk            — Chunk 数据结构 + JSON 加载

数据层：
    retrieval_db     — SQLite 数据库（documents + chunks 表，CRUD）

检索层（7 大独立通道）：
    exact_retriever       — 精确检索：文号、条款号、附件号、表名、指标名
    lexical_retriever     — BM25 检索：监管原文关键词和规范用语
    dense_retriever       — Dense 检索：自然语言和原文表达差异
    metadata_retriever    — 元数据过滤：主体、时间、版本、范围、文件
    table_retriever       — 表格检索：表名、行、列、期间、单位和单元格
    neighborhood_retriever — 父子与近邻检索：父节点、兄弟节点、前后节点
    relation_retriever    — 引用关系检索：条款和表格引用链

编排层：
    retrieval_api    — 统一检索入口 + RRF 融合 + Cross-Encoder 精排

评测层：
    retrieval_eval   — 检索评测框架

工具：
    utils            — ModelScope 模型下载等
"""

# ============================================================
# 核心入口
# ============================================================
from .retrieval_api import RetrievalAPI

# ============================================================
# 数据库层（可直接使用，不依赖检索器）
# ============================================================
from .retrieval_db import RetrievalDB

# ============================================================
# 数据结构
# ============================================================
from .chunk import Chunk, load_json_chunks, flatten_metadata, build_index_text, CHUNK_TYPE_ICONS

# ============================================================
# 迁移层
# ============================================================
from .migration import migrate, export_db, export_tables, MigrationRunner

# ============================================================
# 评测层
# ============================================================
from .retrieval_eval import RetrievalEval

# ============================================================
# 7 大独立检索器（高级用户可直接按需组合）
# ============================================================
from .lexical_retriever import LexicalRetriever
from .dense_retriever import DenseRetriever
from .exact_retriever import ExactRetriever
from .metadata_retriever import MetadataRetriever
from .table_retriever import TableRetriever
from .relation_retriever import RelationRetriever
from .neighborhood_retriever import NeighborhoodRetriever
from .retrieval_request import (
    RetrievalRequest,
    RetrievalHit,
    RetrievalStrategy,
    RerankMode,
)
# Agent Tool 包装层
from .agent_tool import (
    ToolHit,
    search_as_tool,
    to_llm_text,
    SEARCH_TOOL_SCHEMA,
    ToolSpec,
    make_tool_spec,
    Retriever,
)

__all__ = [
    # 核心
    "RetrievalAPI",
    "RetrievalDB",
    # 数据结构
    "Chunk",
    "load_json_chunks",
    "flatten_metadata",
    "build_index_text",
    "CHUNK_TYPE_ICONS",
    # 迁移
    "migrate",
    "export_db",
    "export_tables",
    "MigrationRunner",
    # 评测
    "RetrievalEval",
    # 检索器
    "LexicalRetriever",
    "DenseRetriever",
    "ExactRetriever",
    "MetadataRetriever",
    "TableRetriever",
    "RelationRetriever",
    "NeighborhoodRetriever",
    # 结构化请求/命中
    "RetrievalRequest",
    "RetrievalHit",
    "RetrievalStrategy",
    "RerankMode",
    # Agent Tool 包装层
    "ToolHit",
    "search_as_tool",
    "to_llm_text",
    "SEARCH_TOOL_SCHEMA",
    "ToolSpec",
    "make_tool_spec",
    "Retriever",
]
