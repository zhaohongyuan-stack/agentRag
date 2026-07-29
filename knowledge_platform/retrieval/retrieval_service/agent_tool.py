from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .chunk import CHUNK_TYPE_ICONS
from .retrieval_api import RetrievalAPI
from .retrieval_request import RetrievalRequest, RetrievalStrategy, RetrievalHit


# ════════════════════════════════════════════════════════════════
# ToolHit — 给 LLM 消费的精简命中
# ════════════════════════════════════════════════════════════════

@dataclass
class ToolHit:
    """
    LLM 友好的精简命中结果。

    从 RetrievalHit 裁剪而来，去掉了 trace/scores_detail/metadata 等
    LLM 不需要的内部字段，只保留对生成答案有用的信息。
    """

    rank: int
    """在本次结果中的排名（1-based）"""

    chunk_id: str
    """chunk 唯一标识，用于溯源和二次检索（如 expand_context）"""

    chunk_type: str
    """chunk 类型: clause / cell_fact / note / table / glossary / report_summary ..."""

    doc_name: str
    """文档名称，如 '商业银行资本管理办法'"""

    hierarchy_path: str
    """层级路径，如 '2025年银行业金融机构总资产 / 资产负债季度 / 1. 银行业金融机构'"""

    content: str
    """正文内容（已截断，默认 500 字符）"""

    citation: str
    """引用格式: '《xxx》 Sheet=资产负债季度, Cell=C15'，LLM 可直接复制"""

    score: float
    """综合相关性得分（0~1 或 BM25 原始分），LLM 据此判断结果可信度"""

    evidence_snippet: str = ""
    """出处证据短串"""

    # 元数据精选字段——帮助 LLM 理解数据口径
    applicable_scope: str = ""
    """适用范围: 全部 / 大型商业银行 / 股份制商业银行 ..."""

    attachment_no: str = ""
    """附件编号"""

    clause_number: str = ""
    """条款号"""

    table_name: str = ""
    """表格名称"""

    sheet_name: str = ""
    """工作表名称（Excel）"""

    @classmethod
    def from_retrieval_hit(cls, hit: RetrievalHit, max_chars: int = 500) -> "ToolHit":
        """从完整 RetrievalHit 裁剪为 LLM 友好的 ToolHit"""
        content = hit.content
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars]

        meta = hit.metadata or {}

        return cls(
            rank=hit.rank,
            chunk_id=hit.chunk_id,
            chunk_type=hit.chunk_type,
            doc_name=hit.doc_name or hit.source_file or "",
            hierarchy_path=hit.hierarchy_path or "",
            content=content,
            citation=hit.citation,
            score=hit.score,
            evidence_snippet=hit.evidence_snippet or "",
            applicable_scope=meta.get("applicable_scope", ""),
            attachment_no=meta.get("attachment_no", ""),
            clause_number=meta.get("clause_number", ""),
            table_name=meta.get("table_name", ""),
            sheet_name=meta.get("sheet_name", ""),
        )


# ════════════════════════════════════════════════════════════════
# 核心入口: Agent 调用的检索函数
# ════════════════════════════════════════════════════════════════

def search_as_tool(
    api: RetrievalAPI,
    query: str,
    strategy: str = "hybrid",
    top_k: int = 10,
    filters: Optional[Dict[str, str]] = None,
    max_chars_per_hit: int = 500,
) -> List[ToolHit]:
    """
    Agent Tool 入口 — 接受 LLM 传来的参数，返回 ToolHit 列表。

    参数直接对应 SEARCH_TOOL_SCHEMA 的 properties，
    保证 Agent 的 function calling 参数可以无转换传入。

    Args:
        api: 已加载的 RetrievalAPI 实例
        query: 检索查询文本
        strategy: hybrid | bm25 | dense | exact | table | metadata
        top_k: 返回条数
        filters: 元数据过滤，如 {"chunk_type": "cell_fact"}
        max_chars_per_hit: 每条结果截断上限（LLM 用 300~800 合适）

    Returns:
        List[ToolHit]: 精简命中列表
    """
    req = RetrievalRequest(
        query=query,
        strategy=RetrievalStrategy(strategy),
        top_k=top_k,
        filters=filters or {},
        max_chars_per_hit=max_chars_per_hit,
        include_evidence=True,
    )

    hits: List[RetrievalHit] = api.search_request(req)
    return [ToolHit.from_retrieval_hit(h, max_chars=max_chars_per_hit) for h in hits]


# ════════════════════════════════════════════════════════════════
# 文本格式化: ToolHit → LLM context 文本
# ════════════════════════════════════════════════════════════════


def to_llm_text(hits: List[ToolHit], max_total_chars: int = 4000) -> str:
    """
    将 ToolHit 列表格式化为 LLM 可直接消费的参考文本。

    每条包含: 编号、类型、文档名、层级路径、正文、出处、得分。
    Agent 把这段文本拼到 tool result 的 content 里传给 LLM。

    Args:
        hits: ToolHit 列表
        max_total_chars: 总字符上限（超出则截断）

    Returns:
        格式化的参考文本
    """
    if not hits:
        return "【检索结果为空 — 未找到相关内容】"

    lines = [
        "【检索参考资料 — 共 {} 条，严格基于此回答并注明出处】\n".format(len(hits))
    ]

    total = len(lines[0])
    for h in hits:
        icon = CHUNK_TYPE_ICONS.get(h.chunk_type, "📄")

        # 第一行: 编号 + 类型 + 文档 + 层级
        header_parts = [f"[{h.rank}]", icon, h.chunk_type]
        if h.doc_name:
            header_parts.append(f"《{h.doc_name}》")
        if h.attachment_no:
            header_parts.append(f"附件{h.attachment_no}")
        header = " ".join(header_parts)

        entry_lines = [header]

        # 层级路径（如果有）
        if h.hierarchy_path:
            entry_lines.append(f"    路径: {h.hierarchy_path}")

        # 适用范围 / 口径（关键！LLM 可能选错口径的数据）
        if h.applicable_scope and h.applicable_scope not in ("全部", "未指定", ""):
            entry_lines.append(f"    口径: {h.applicable_scope}")

        # 正文
        content = h.content.replace("\n", " ").strip()
        entry_lines.append(f"    内容: {content}")

        # 出处 + 得分
        meta_parts = []
        if h.citation:
            meta_parts.append(f"出处: {h.citation}")
        meta_parts.append(f"得分: {h.score:.4f}")
        entry_lines.append(f"    {' | '.join(meta_parts)}")

        block = "\n".join(entry_lines) + "\n"
        total += len(block)

        if total > max_total_chars:
            lines.append(f"\n【已达到 {max_total_chars} 字符限制，后续 {len(hits) - h.rank} 条未展示】")
            break

        lines.append(block)

    lines.append("【回答时请引用编号 [1][2]... 并注明出处】")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Tool Schema — 给 LLM 的 function-calling JSON Schema
# ════════════════════════════════════════════════════════════════

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_regulatory_docs",
        "description": (
            "检索中国银行业监管法规和统计数据。覆盖银保监会/金融监管总局发布的"
            "监管办法、统计报表、指标数据。\n\n"

            "**何时使用**: 用户询问监管指标、法规条文、银行经营数据（总资产、总负债、"
            "不良贷款率等）、特定条款内容时，必须先调此工具检索，再基于结果回答。\n\n"

            "**策略选择指南**:\n"
            "- hybrid: 默认推荐，BM25+语义双路融合，适合大多数场景\n"
            "- bm25: 纯关键词匹配，适合精确查条款号/专有名词\n"
            "- dense: 纯语义匹配，适合模糊问题/同义词/概念性查询\n"
            "- exact: 精确子串/正则匹配，适合查具体条文原文\n"
            "- table: 直接查表格行列，适合查具体数据单元格\n"
            "- metadata: 按类型/附件/章节等元数据过滤，适合浏览结构\n\n"

            "**filters 常用组合**:\n"
            '- 只查数据: {"chunk_type": "cell_fact"}\n'
            '- 只查法规条款: {"chunk_type": "clause"}\n'
            '- 查特定文档: {"doc_id": "400"}\n'
            '- 查特定附件: {"attachment_no": "附件1"}\n'
            '- 组合使用: {"chunk_type": "cell_fact", "doc_id": "011"}\n\n'

            "**技巧**: 如果首次检索结果不理想（得分低、内容不相关），"
            "换 strategy 或调整 query 关键词再试一次。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索查询文本。从用户问题中提取核心关键词，去掉疑问词"
                        "（如'是多少''有哪些''请说明'）。"
                        "示例: '银行业金融机构 总资产 2025年 三季度'"
                    ),
                },
                "strategy": {
                    "type": "string",
                    "enum": ["hybrid", "bm25", "dense", "exact", "table", "metadata"],
                    "description": "检索策略，默认 hybrid",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "返回结果数量，默认 10。简单问题用 5，复杂分析用 20",
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "元数据过滤条件，AND 组合。可选 key:\n"
                        "- chunk_type: clause(法规条款) | cell_fact(数据单元格) | "
                        "note(报表说明) | report_summary(报告摘要) | glossary(术语定义) | "
                        "table(表格结构)\n"
                        "- doc_id: 文档编号\n"
                        "- applicable_scope: 全部 | 大型商业银行 | 股份制商业银行 | ...\n"
                        "- attachment_no: 附件1 | 附件2 | ...\n"
                        "- clause_number: 条款号，如 '第十二条'\n"
                        "- table_name: 表格名称"
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["query"],
        },
    },
}


# ════════════════════════════════════════════════════════════════
# ToolSpec — 把 schema + executor 打包，方便 Agent 直接使用
# ════════════════════════════════════════════════════════════════

@dataclass
class ToolSpec:
    """一个 LLM Tool 的完整规格: schema（给 LLM）+ execute（执行函数）"""

    schema: Dict[str, Any]
    """OpenAI/Claude function-calling JSON Schema"""

    execute: Callable[..., str]
    """
    执行函数。签名为 (**kwargs) -> str。
    kwargs 直接来自 LLM tool_call.arguments 的 JSON 解析结果。
    返回值是纯文本，直接塞进 tool result message 的 content。
    """


def make_tool_spec(api: RetrievalAPI) -> ToolSpec:
    """
    基于已加载的 RetrievalAPI 实例创建 ToolSpec。

    闭包捕获 api，Agent 只需调 tool_spec.execute(**args) 即可。

    Args:
        api: 已加载的 RetrievalAPI 实例

    Returns:
        ToolSpec: schema + execute 的打包

    Example:
        api = RetrievalAPI()
        api.load("regulatory_docs/")
        tool = make_tool_spec(api)

        # Agent 循环中:
        args = json.loads(tool_call.arguments)
        result_text = tool.execute(**args)
    """
    def _execute(
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        filters: Optional[Dict[str, str]] = None,
        **kwargs,  # 忽略 LLM 可能多传的参数
    ) -> str:
        hits = search_as_tool(
            api,
            query=query,
            strategy=strategy,
            top_k=top_k,
            filters=filters,
        )
        return to_llm_text(hits)

    return ToolSpec(
        schema=SEARCH_TOOL_SCHEMA,
        execute=_execute,
    )


# ════════════════════════════════════════════════════════════════
# 便捷模块级工具（导入即用）
# ════════════════════════════════════════════════════════════════

# 方便外部直接 import 的别名
tool_schema  = SEARCH_TOOL_SCHEMA
format_hits  = to_llm_text


# ════════════════════════════════════════════════════════════════
# Retriever — 一行加载，一行检索的最简入口
# ════════════════════════════════════════════════════════════════

# 支持的检索通道（字符串即可，无需 import enum）
STRATEGIES = ("hybrid", "bm25", "dense", "exact", "table", "metadata")
EXACT_MODES = ("contains", "exact", "regex", "prefix")


class Retriever:
    """
    最简检索入口 — 加载数据后，只需一个 search() 方法。

    用法:
        r = Retriever("regulatory_docs/")
        r.load()

        # 切换通道只需改 strategy 字符串
        r.search("核心一级资本合格标准")                         # 默认 hybrid
        r.search("不良贷款率", strategy="bm25", top_k=10)
        r.search("第十二条", strategy="exact", exact_mode="exact")
        r.search("银行业金融机构总资产", strategy="dense", top_k=5)
        r.search("KM1", strategy="table", table_name="KM1")
        r.search("", strategy="metadata", filters={"chunk_type": "clause"})
    """

    def __init__(self,
                 data_dir: str = "regulatory_docs",
                 embed_model: str = "BAAI/bge-small-zh-v1.5"):
        self._data_dir = data_dir
        self._embed_model = embed_model
        self._api: Optional[RetrievalAPI] = None

    def load(self) -> "Retriever":
        """加载数据 + 构建全部索引（启动时调用一次）"""
        self._api = RetrievalAPI(embed_model=self._embed_model)
        self._api.load(self._data_dir)
        return self

    def search(self,
               query: str,
               *,
               strategy: str = "hybrid",
               top_k: int = 10,
               filters: Optional[Dict[str, str]] = None,
               exact_mode: str = "contains",
               bm25_k: int = 20,
               vector_k: int = 20,
               expand_context: bool = False,
               ) -> List[Dict[str, Any]]:
        """
        统一检索入口 — strategy 切换通道，字符串即可。

        参数:
            query:    查询文本
            strategy: 检索通道 — hybrid | bm25 | dense | exact | table | metadata
            top_k:    返回条数
            filters:  元数据过滤，如 {"chunk_type": "clause", "doc_id": "400"}
            exact_mode: strategy=exact 时的匹配模式 — contains | exact | regex | prefix
            bm25_k:   hybrid 时 BM25 粗排候选数
            vector_k: hybrid 时向量粗排候选数
            expand_context: 是否扩展邻域上下文

        返回:
            List[dict]: 每条含 chunk_id, content, score, doc_name, evidence_snippet ...
        """
        if self._api is None:
            raise RuntimeError("请先调用 .load() 加载数据")

        if strategy not in STRATEGIES:
            raise ValueError(f"不支持的通道: {strategy}，可选: {STRATEGIES}")

        req = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy(strategy),
            top_k=top_k,
            filters=filters or {},
            bm25_k=bm25_k,
            vector_k=vector_k,
            exact_mode=exact_mode,
            expand_context=expand_context,
        )
        hits = self._api.search_request(req)
        return [h.to_dict() for h in hits]

    # ── 便捷属性：直接访问底层检索器和 API ──
    @property
    def api(self) -> Optional[RetrievalAPI]:
        return self._api

    @property
    def chunk_count(self) -> int:
        return self._api.chunk_count if self._api else 0

    @property
    def tables(self) -> List[str]:
        return self._api.list_tables() if self._api else []
