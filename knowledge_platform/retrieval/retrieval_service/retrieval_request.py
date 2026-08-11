"""
RetrievalRequest / RetrievalHit — 结构化检索请求与命中结果

设计目标：
  - RetrievalRequest:  稳定、可序列化、可复现的检索输入
  - RetrievalHit:      可解释、可追溯、自带出处的检索输出

用法：
    from retrieval_service import RetrievalRequest, RetrievalHit

    req = RetrievalRequest(
        query="核心一级资本合格标准",
        top_k=5,
        filters={"chunk_type": "clause", "applicable_scope": "全部"},
        strategy="hybrid",
    )
    hits = api.search_request(req)   # → List[RetrievalHit]

    for h in hits:
        print(h.explain())           # 人类可读的解释
        print(h.trace)               # 检索链路追溯
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# 检索策略枚举
# ============================================================
class RetrievalStrategy(str, Enum):
    """检索通道策略"""
    BM25     = "bm25"       # 仅 BM25 关键词
    DENSE    = "dense"      # 仅向量语义
    HYBRID   = "hybrid"     # BM25 + Dense → RRF 融合（默认）
    EXACT    = "exact"      # 精确/子串/正则匹配
    METADATA = "metadata"   # 仅元数据过滤
    RELATION = "relation"   # 结构化关系查询
    TABLE    = "table"      # 表格行列检索


class RerankMode(str, Enum):
    """精排模式"""
    NONE      = "none"       # 不精排
    CROSS_ENC = "cross_enc"  # Cross-Encoder 重排序
    LLM       = "llm"        # LLM 重排序（预留）


# ============================================================
# RetrievalRequest
# ============================================================
@dataclass
class RetrievalRequest:
    """
    结构化检索请求 — 一次检索的完整输入规格。

    所有字段均可序列化（JSON-friendly），保证同参数 → 同结果（确定性）。
    """

    # ── 核心查询 ──
    query: str = ""
    """用户问题原文 / 检索关键词"""

    top_k: int = 10
    """最终返回条数"""

    # ── 策略选择 ──
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    """检索通道策略"""

    rerank: RerankMode = RerankMode.NONE
    """精排模式"""

    # ── 策略参数 ──
    bm25_k: int = 20
    """BM25 粗排候选数（strategy=hybrid 时生效）"""

    vector_k: int = 20
    """向量粗排候选数（strategy=hybrid 时生效）"""

    rrf_k: int = 60
    """RRF 融合平滑常数"""

    rerank_k: int = 30
    """送入精排模型的候选数（top_k 的倍数）"""

    # ── 过滤条件 ──
    filters: Dict[str, Any] = field(default_factory=dict)
    """
    元数据过滤条件，AND 组合。支持的 key：
      chunk_type, doc_id, doc_name, chapter_number,
      clause_number, applicable_scope, normative_level,
      table_name, glossary_term, attachment_no, ...
    示例：
      {"chunk_type": "clause", "doc_id": "400", "applicable_scope": "全部"}
    """

    # ── 精确检索参数（strategy=exact 时生效）──
    exact_mode: str = "contains"
    """精确匹配模式: 'contains' | 'exact' | 'regex' | 'prefix'"""

    # ── 上下文扩展 ──
    expand_context: bool = False
    """是否自动扩展邻域上下文（父/子/前/后 chunk）"""

    context_window: int = 2
    """邻域窗口大小（expand_context=True 时生效）"""

    # ── 输出控制 ──
    include_content_raw: bool = False
    """是否附带 content_raw（原始未精简文本）"""

    include_evidence: bool = True
    """是否附带 evidence_snippet（出处证据）"""

    max_chars_per_hit: int = 2000
    """单条命中内容的字符截断上限（0=不截断）"""

    # ── 请求元数据（可追溯）──
    request_id: str = ""
    """请求唯一 ID（空则自动生成）"""

    caller: str = ""
    """调用方标识（用于日志/监控）"""

    timestamp: float = 0.0
    """请求时间戳（空则自动填充）"""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = _make_request_id(self.query)
        if not self.timestamp:
            self.timestamp = time.time()

    # ── 便利方法 ──

    def to_dict(self) -> Dict[str, Any]:
        """序列化为纯 dict（JSON 安全）"""
        d = asdict(self)
        d["strategy"] = self.strategy.value
        d["rerank"] = self.rerank.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalRequest":
        """从 dict 反序列化"""
        d = dict(d)  # shallow copy
        d.setdefault("strategy", RetrievalStrategy.HYBRID)
        d.setdefault("rerank", RerankMode.NONE)
        if isinstance(d["strategy"], str):
            d["strategy"] = RetrievalStrategy(d["strategy"])
        if isinstance(d["rerank"], str):
            d["rerank"] = RerankMode(d["rerank"])
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def fingerprint(self) -> str:
        """
        请求指纹 — 相同指纹 ≈ 相同语义输入（忽略 request_id/timestamp）。
        用于缓存去重、回归测试。
        """
        core = {
            "query": self.query,
            "top_k": self.top_k,
            "strategy": self.strategy.value,
            "rerank": self.rerank.value,
            "filters": self.filters,
            "exact_mode": self.exact_mode,
        }
        raw = __import__("json").dumps(core, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return (f"RetrievalRequest(id={self.request_id}, "
                f"strategy={self.strategy.value}, top_k={self.top_k}, "
                f"query='{self.query[:60]}')")


# ============================================================
# RetrievalHit — 单条命中
# ============================================================
@dataclass
class RetrievalHit:
    """
    单条检索命中结果 — 稳定、可解释、可追溯。

    字段分组：
      identity   — 命中的是什么
      content    — 命中的内容
      scores     — 多维得分（解释为什么排在这里）
      trace      — 检索链路（追溯怎么来的）
      context    — 邻域上下文（可选的扩展信息）
    """

    # ════════════════════════════════════════════════════════════
    # identity — 命中的是什么
    # ════════════════════════════════════════════════════════════
    chunk_id: str = ""
    chunk_type: str = "clause"
    doc_id: str = ""
    doc_name: str = ""
    doc_title: str = ""
    hierarchy_path: str = ""
    source_file: str = ""

    # ════════════════════════════════════════════════════════════
    # content — 命中的内容
    # ════════════════════════════════════════════════════════════
    content: str = ""
    content_raw: str = ""
    evidence_snippet: str = ""
    """出处证据：章/节/条 层级路径，用于 LLM 引用"""

    # ════════════════════════════════════════════════════════════
    # scores — 多维得分（解释为什么排在这里）
    # ════════════════════════════════════════════════════════════
    score: float = 0.0
    """最终综合得分（排序依据）"""

    scores_detail: Dict[str, float] = field(default_factory=dict)
    """
    分项得分，至少包含最终排序所使用的得分项。
    示例：
      {"bm25": 12.34, "dense": 0.87, "rrf": 0.032, "rerank": 3.21}
    """

    rank: int = -1
    """在本次结果中的排名（1-based）"""

    # ════════════════════════════════════════════════════════════
    # trace — 检索链路（追溯怎么来的）
    # ════════════════════════════════════════════════════════════
    matched_by: List[str] = field(default_factory=list)
    """
    哪些通道匹配了此条。示例：["bm25", "dense"]
    若仅一个通道命中，则为单元素列表；融合命中则为多元素。
    """

    trace: Dict[str, Any] = field(default_factory=dict)
    """
    完整检索链路，记录每一步的关键信息。
    示例：
      {
        "request_id": "a1b2c3d4",
        "strategy": "hybrid",
        "bm25_rank": 3,
        "bm25_score": 12.34,
        "dense_rank": 7,
        "dense_score": 0.87,
        "rrf_rank": 2,
        "rrf_score": 0.032,
        "rerank_input_rank": 2,
        "rerank_score": 3.21,
        "filters_applied": {"chunk_type": "clause"},
      }
    """

    # ════════════════════════════════════════════════════════════
    # context — 可选的邻域上下文
    # ════════════════════════════════════════════════════════════
    context: Optional[Dict[str, Any]] = None
    """
    当 expand_context=True 时填充，包含：
      parent   — 父 chunk
      children — 子 chunk 列表
      siblings — 同级 chunk 列表
      prev     — 前一 chunk
      next     — 后一 chunk
    """

    # ════════════════════════════════════════════════════════════
    # metadata — 原始元数据（透传）
    # ════════════════════════════════════════════════════════════
    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    完整元数据，字段同 chunk_json约定.md v1.2：
      文档级:   parser_type, parser_version, parse_timestamp, source_url, sha256
      结构级:   attachment_no, applicable_scope, chapter_number, clause_number,
               subclause_number, capital_tool_level, glossary_term, ...
      语义级:   normative_level, numeric_conditions, keywords,
               cross_attachment_refs, cross_table_refs
      表格级:   table_name, table_full_name, table_section_name, sheet_name,
               row_count, col_count, merge_info, cross_refs
    """

    # ── 便利方法 ──

    def explain(self, verbose: bool = False) -> str:
        """
        生成人类可读的解释文本。
        简洁模式: 一行摘要
        详细模式: 多行完整追溯
        """
        scope = (f" [适用: {self.metadata.get('applicable_scope', '')}]"
                 if self.metadata.get("applicable_scope") not in ("全部", "未指定", "", None)
                 else "")
        attachment = f" {self.metadata.get('attachment_no', '')}" if self.metadata.get("attachment_no") else ""

        lines = [
            f"[{self.rank}] {self.chunk_type}{attachment} — "
            f"{self.doc_name}{scope} — {self.evidence_snippet or self.hierarchy_path}",
            f"    score={self.score:.4f}  matched_by={self.matched_by}  "
            f"chunk_id={self.chunk_id}",
        ]

        if verbose:
            if self.scores_detail:
                detail = " ".join(f"{k}={v:.4f}" for k, v in self.scores_detail.items())
                lines.append(f"    scores_detail: {detail}")
            if self.trace:
                trace_str = ", ".join(f"{k}={v}" for k, v in self.trace.items()
                                      if k not in ("request_id", "strategy", "filters_applied"))
                lines.append(f"    trace: {trace_str}")
            if self.content:
                preview = self.content[:200].replace("\n", " ")
                lines.append(f"    content: {preview}...")

        return "\n".join(lines)

    @property
    def is_hybrid_match(self) -> bool:
        """是否被多路通道同时命中"""
        return len(self.matched_by) > 1

    @property
    def citation(self) -> str:
        """
        LLM 引用格式：文档名 + 附件号 + 条款层级路径。
        示例："《资本管理办法》 附件1 第二章 第十二条"
        """
        parts = [f"《{self.doc_name}》"]
        att = self.metadata.get("attachment_no", "")
        if att:
            parts.append(f"附件{att}")
        if self.evidence_snippet:
            parts.append(self.evidence_snippet)
        elif self.hierarchy_path:
            parts.append(self.hierarchy_path)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为纯 dict（JSON 安全），对齐 _chunk_to_result 格式"""
        return {
            # identity
            "chunk_id":       self.chunk_id,
            "chunk_type":     self.chunk_type,
            "doc_id":         self.doc_id,
            "doc_name":       self.doc_name,
            "doc_title":      self.doc_title,
            "hierarchy_path": self.hierarchy_path,
            "source":         self.source_file,
            # content
            "citation":         self.citation,
            "content":          self.content,
            "content_raw":      self.content_raw,
            "evidence_snippet": self.evidence_snippet,
            # scores
            "score":         self.score,
            "scores_detail": self.scores_detail,
            "rank":          self.rank,
            # trace
            "matched_by": self.matched_by,
            "trace":      self.trace,
            # context
            "context":  self.context,
            # metadata
            "metadata": self.metadata,
        }

    @classmethod
    def from_chunk_result(cls,
                          result: Dict[str, Any],
                          rank: int = -1,
                          matched_by: Optional[List[str]] = None,
                          trace: Optional[Dict[str, Any]] = None,
                          context: Optional[Dict[str, Any]] = None) -> "RetrievalHit":
        """
        从 RetrievalAPI._chunk_to_result 的 dict 输出构造 RetrievalHit。
        兼容现有 API，无需大规模重构即可获得结构化命中。

        用法：
            raw = api.search("核心一级资本", top_k=5)
            hits = [RetrievalHit.from_chunk_result(r, rank=i+1,
                      matched_by=["bm25","dense"]) for i, r in enumerate(raw)]
        """
        meta = result.get("metadata", {})
        return cls(
            chunk_id=result.get("chunk_id", ""),
            chunk_type=result.get("chunk_type", "clause"),
            doc_id=result.get("doc_id", ""),
            doc_name=result.get("doc_name", ""),
            doc_title=result.get("doc_title", ""),
            hierarchy_path=result.get("hierarchy_path", ""),
            source_file=result.get("source", ""),
            content=result.get("content", ""),
            content_raw=result.get("content_raw", ""),
            evidence_snippet=result.get("evidence_snippet", ""),
            score=float(result.get("score", 0)),
            rank=rank,
            matched_by=matched_by or [],
            trace=trace or {},
            context=context,
            metadata={
                k: v for k, v in meta.items()
                if k not in ("parent_chunk_id", "sub_chunks")
            },
        )

    def __repr__(self) -> str:
        return (f"RetrievalHit(rank={self.rank}, score={self.score:.4f}, "
                f"chunk_id={self.chunk_id[:40]}, "
                f"matched_by={self.matched_by})")


# ============================================================
# 内部工具
# ============================================================
def _make_request_id(query: str) -> str:
    """基于 query + 时间戳生成唯一请求 ID"""
    raw = f"{query}|{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
