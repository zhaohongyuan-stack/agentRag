"""
查询改写器

组合指代消解和同义词扩展，生成多通道检索查询。

改写流程:
  1. 指代消解 — 如果有会话上下文，解析代词和模糊指代（"这个比例" → "核心一级资本充足率"）
  2. 同义词扩展 — 将监管术语扩展为包含同义词/缩写的查询（"资本充足率" → 追加 "CAR"）
  3. 通道查询生成 — 为每个检索通道生成专用查询文本
     - lexical: 指代消解后的查询（适合 BM25 关键词匹配）
     - dense: 同义词扩展后的查询（适合向量语义检索）
     - exact: 关键术语提取（适合精确/子串匹配）
  4. 歧义标记 — 如果指代消解失败（多个候选或无候选），标记歧义

支持两种模式:
  - LLM 模式: 使用 LLM 进行高级指代消解
  - 规则模式（Mock/降级）: 使用规则进行指代消解

用法:
    rewriter = QueryRewriter()
    result = rewriter.rewrite("这个比例适用吗", session_context=ctx)
    # result.contextualized_query = "核心一级资本充足率适用吗"
    # result.channel_queries = {"lexical": "...", "dense": "...", "exact": "..."}
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...runtime.llm_client import LLMClient, LLMMessage, get_llm_client
from .reference_resolver import ReferenceResolver, ResolutionResult, SessionContext
from .synonym_dict import SynonymDict

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """
    改写后的查询结果

    包含原始查询、指代消解后的查询、各检索通道专用查询，
    以及歧义标记信息。
    """

    original_query: str
    """用户原始查询文本"""

    contextualized_query: str
    """指代消解后的查询文本（无会话上下文时与原始查询相同）。
    如果含选项且为数值型，此处为去除选项后的纯净查询；
    如果含选项且为文本型，此处为增强后（含选项文本）的查询。"""

    channel_queries: Dict[str, str] = field(default_factory=dict)
    """各检索通道专用查询文本: {lexical: ..., dense: ..., exact: ...}"""

    rewrites: List[str] = field(default_factory=list)
    """所有改写版本列表（去重），供后续模块选择使用"""

    ambiguity_flagged: bool = False
    """是否因指代消解失败而标记歧义"""

    ambiguity_reason: str = ""
    """歧义原因描述（无歧义时为空字符串）"""

    # ── 选项处理字段 ──
    options: List[Dict[str, Any]] = field(default_factory=list)
    """提取的选项列表，格式 [{"label": "A", "text": "...", "type": "numeric/textual"}]"""

    option_set_type: str = "none"
    """选项集类型: "numeric"(全部数值) | "textual"(含文本表述) | "none"(无选项)"""

    prompt_mode: str = "normal"
    """生成模式: "normal" | "calculate_and_match"(数值计算型) | "verify_each_option"(事实判断型)"""

    clean_query: str = ""
    """去除选项后的纯净问题文本（不含选项），用于 LLM 生成的 question 字段"""

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "contextualized_query": self.contextualized_query,
            "channel_queries": dict(self.channel_queries),
            "rewrites": list(self.rewrites),
            "ambiguity_flagged": self.ambiguity_flagged,
            "ambiguity_reason": self.ambiguity_reason,
            "options": list(self.options),
            "option_set_type": self.option_set_type,
            "prompt_mode": self.prompt_mode,
            "clean_query": self.clean_query,
        }


class QueryRewriter:
    """
    查询改写器

    组合 LLM 指代消解 + 同义词扩展，生成多通道检索查询。
    当 LLM 不可用（Mock 模式）时，自动降级为规则版指代消解。
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        synonym_dict: Optional[SynonymDict] = None,
        reference_resolver: Optional[ReferenceResolver] = None,
    ):
        """
        初始化查询改写器

        Args:
            llm_client: LLM 客户端，为 None 时使用全局单例
            synonym_dict: 同义词词典，为 None 时使用默认词典
            reference_resolver: 指代消解器，为 None 时使用默认实例
        """
        self._llm_client = llm_client or get_llm_client()
        self._synonym_dict = synonym_dict or SynonymDict()
        self._reference_resolver = reference_resolver or ReferenceResolver()

    def rewrite(
        self,
        original_query: str,
        query_spec: Optional[Any] = None,
        session_context: Optional[SessionContext] = None,
    ) -> RewrittenQuery:
        """
        改写用户查询

        执行指代消解 → 同义词扩展 → 通道查询生成的完整流程。

        Args:
            original_query: 用户原始查询
            query_spec: 查询规格（可选，用于提取额外关键术语）
            session_context: 会话上下文（可选，用于多轮指代消解）

        Returns:
            RewrittenQuery 对象
        """
        # 空查询处理
        if not original_query or not original_query.strip():
            return RewrittenQuery(
                original_query=original_query or "",
                contextualized_query=original_query or "",
                channel_queries={},
                rewrites=[],
                ambiguity_flagged=False,
                ambiguity_reason="",
            )

        query = original_query.strip()
        ambiguity_flagged = False
        ambiguity_reason = ""

        # ── 步骤 0: 选项提取与分类 ──
        options, clean_query = self._extract_options(query)
        option_set_type = self._classify_option_set(options)

        if option_set_type == "numeric":
            # 数值计算型：剥离选项，用清洁查询检索
            search_query = clean_query
            prompt_mode = "calculate_and_match"
            logger.info(
                "[选项处理] 数值计算型 → 剥离 %d 个选项，clean_query='%s'",
                len(options), clean_query[:80],
            )
        elif option_set_type == "textual":
            # 事实判断型：保留选项文本，增强检索查询
            search_query = self._build_enhanced_query(clean_query, options)
            prompt_mode = "verify_each_option"
            logger.info(
                "[选项处理] 事实判断型 → 增强 %d 个选项，search_query='%s'",
                len(options), search_query[:80],
            )
        else:
            # 无选项：正常流程
            search_query = query
            prompt_mode = "normal"

        # ── 步骤 1: 指代消解 ──
        contextualized = search_query
        # 仅当存在会话历史时才执行指代消解（首次查询无需 LLM 调用）
        _has_context = session_context is not None and (
            session_context.previous_queries
            or session_context.mentioned_metrics
            or session_context.mentioned_docs
            or session_context.previous_entities
        )
        if _has_context:
            if self._llm_client.is_mock:
                # Mock 模式: 使用规则消解
                result = self._rule_based_resolve(query, session_context)
            else:
                # LLM 模式: 使用 LLM 消解（失败时降级到规则）
                result = self._llm_resolve(query, session_context)

            contextualized = result.resolved_query
            ambiguity_flagged = result.ambiguity_flagged
            ambiguity_reason = result.ambiguity_reason

        # ── 步骤 2: 同义词扩展 ──
        expanded = self._synonym_dict.expand_query(contextualized)

        # ── 步骤 2b: 注入消歧后的术语扩展 ──
        if query_spec is not None:
            resolved_terms = getattr(query_spec, "resolved_terms", None)
            if resolved_terms:
                for term, info in resolved_terms.items():
                    meaning = info.get("resolved_meaning", "")
                    # 如果原查询包含多义术语但不含消歧后的含义词，补充注入
                    if term in contextualized and meaning and meaning not in contextualized:
                        contextualized = contextualized.replace(term, f"{term}（{meaning}）")
                        # 同步更新 expanded
                        if term in expanded and meaning not in expanded:
                            expanded = expanded.replace(term, f"{term}（{meaning}）")

        # ── 步骤 3: 生成通道查询 ──
        channel_queries = self._generate_channel_queries(
            contextualized, expanded, query_spec
        )

        # ── 步骤 4: 收集所有改写版本（去重） ──
        rewrites = self._collect_rewrites(query, contextualized, expanded, channel_queries)

        return RewrittenQuery(
            original_query=original_query,
            contextualized_query=contextualized,
            channel_queries=channel_queries,
            rewrites=rewrites,
            ambiguity_flagged=ambiguity_flagged,
            ambiguity_reason=ambiguity_reason,
            options=options,
            option_set_type=option_set_type,
            prompt_mode=prompt_mode,
            clean_query=clean_query if clean_query else original_query,
        )

    # ============================================================
    # 指代消解
    # ============================================================

    def _llm_resolve(
        self, query: str, session_context: SessionContext
    ) -> ResolutionResult:
        """
        使用 LLM 进行指代消解

        构建系统提示和用户消息，调用 LLM 解析代词和模糊指代。
        LLM 调用失败时自动降级到规则版消解。

        Args:
            query: 用户查询
            session_context: 会话上下文

        Returns:
            ResolutionResult 对象
        """
        system_prompt = """你是一个监管法规领域的指代消解专家。
给定用户问题和会话上下文，解析其中的代词和模糊指代（如"这个比例"、"那个文件"）。

请返回 JSON 格式:
{
  "resolved_query": "消解后的查询文本",
  "was_resolved": true/false,
  "ambiguity_flagged": true/false,
  "ambiguity_reason": "歧义原因（如有）",
  "resolved_entity": "解析到的具体实体名称（如有）"
}

消解规则:
- 如果指代明确（上下文中只有一个候选），将指代词替换为具体实体名称
- 如果有多个候选，不替换并标记 ambiguity_flagged=true，在 ambiguity_reason 中列出候选
- 如果无指代词，原样返回，was_resolved=false
- 替换时保持查询其余部分不变"""

        user_content = json.dumps(
            {
                "query": query,
                "session_context": session_context.to_dict(),
            },
            ensure_ascii=False,
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            result = self._llm_client.chat_json(messages, temperature=0.1)
            return ResolutionResult(
                resolved_query=result.get("resolved_query", query),
                was_resolved=result.get("was_resolved", False),
                ambiguity_flagged=result.get("ambiguity_flagged", False),
                ambiguity_reason=result.get("ambiguity_reason", ""),
                resolved_entity=result.get("resolved_entity"),
            )
        except Exception as e:
            logger.warning(f"LLM 指代消解失败，回退到规则模式: {e}")
            return self._rule_based_resolve(query, session_context)

    def _rule_based_resolve(
        self, query: str, session_context: SessionContext
    ) -> ResolutionResult:
        """
        规则版指代消解（LLM 不可用时的降级方案）

        使用 ReferenceResolver 的规则匹配进行消解。

        Args:
            query: 用户查询
            session_context: 会话上下文

        Returns:
            ResolutionResult 对象
        """
        return self._reference_resolver.resolve_detailed(query, session_context)

    # ============================================================
    # 通道查询生成
    # ============================================================

    def _generate_channel_queries(
        self,
        contextualized: str,
        expanded: str,
        query_spec: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        生成各检索通道的专用查询文本

        - lexical: 指代消解后的查询，适合 BM25 关键词匹配
        - dense: 同义词扩展后的查询，适合向量语义检索
        - exact: 关键术语提取，适合精确/子串匹配

        Args:
            contextualized: 指代消解后的查询
            expanded: 同义词扩展后的查询
            query_spec: 查询规格（可选，用于提取额外关键术语）

        Returns:
            通道名 → 查询文本 的字典
        """
        return {
            "lexical": contextualized,
            "dense": expanded,
            "exact": self._extract_key_terms(contextualized, query_spec),
        }

    def _extract_key_terms(
        self, query: str, query_spec: Optional[Any] = None
    ) -> str:
        """
        提取查询中的关键术语（用于精确检索通道）

        提取条款号、文档名、已知监管指标/术语等关键信息，
        拼接为空格分隔的术语字符串。

        Args:
            query: 查询文本
            query_spec: 查询规格（可选，从中提取已识别的实体）

        Returns:
            关键术语字符串，无关键术语时返回原始查询
        """
        key_terms: List[str] = []

        # 从 query_spec 中提取已识别的实体
        if query_spec is not None:
            entities = self._get_entities(query_spec)
            for entity in entities:
                entity_type = entity.get("entity_type", "")
                value = entity.get("value", "")
                if entity_type in ("metric_name", "doc_name", "clause_number") and value:
                    key_terms.append(value)

        # 从查询文本中提取条款号
        clause_match = re.search(r"第[一二三四五六七八九十百千零\d]+条", query)
        if clause_match:
            key_terms.append(clause_match.group(0))

        # 从查询文本中提取文档名
        doc_match = re.search(r"《[^》]+》", query)
        if doc_match:
            key_terms.append(doc_match.group(0))

        # 从查询文本中提取已知监管术语
        known_terms = self._synonym_dict.find_terms(query)
        key_terms.extend(known_terms)

        # 去重并过滤空值
        seen: set = set()
        result: List[str] = []
        for term in key_terms:
            if term and term not in seen:
                seen.add(term)
                result.append(term)

        if result:
            return " ".join(result)
        return query

    def _get_entities(self, query_spec: Any) -> List[dict]:
        """从 query_spec 中提取实体列表（兼容对象和字典）"""
        if hasattr(query_spec, "entities"):
            entities = query_spec.entities
            # 如果实体对象有 to_dict 方法，转换为字典
            result = []
            for e in entities:
                if hasattr(e, "to_dict"):
                    result.append(e.to_dict())
                elif isinstance(e, dict):
                    result.append(e)
                else:
                    result.append({"entity_type": getattr(e, "entity_type", ""),
                                   "value": getattr(e, "value", "")})
            return result
        elif isinstance(query_spec, dict):
            return query_spec.get("entities", [])
        return []

    # ============================================================
    # 选项提取与分类
    # ============================================================

    # 选项边界：匹配 "选项A：" / "A." / "A、" / "(A) " 等
    _OPTION_BOUNDARY = re.compile(
        r'(?:选项)?[（(]?\s*([A-D])\s*[）)、．.：:]'
    )

    def _extract_options(self, query: str) -> tuple:
        """
        从查询中提取选项并返回清洁查询

        定位所有选项边界（A/B/C/D 标签），通过相邻边界截取每个选项的完整文本。

        Args:
            query: 用户原始查询

        Returns:
            (options, clean_query) 元组
            options: [{"label": "A", "text": "...", "type": "numeric/textual"}, ...]
            clean_query: 去除选项后的问题文本
        """
        matches = list(self._OPTION_BOUNDARY.finditer(query))
        if len(matches) < 2:
            return [], query

        options = []
        for i, match in enumerate(matches):
            label = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(query)
            text = query[start:end].strip().rstrip("。，、；")

            options.append({
                "label": label,
                "text": text,
                "type": self._classify_option(text),
            })

        # 清洁查询：截取第一个选项边界之前的部分
        clean_end = matches[0].start()
        clean_query = query[:clean_end].strip().rstrip("。，、；")

        return options, clean_query

    def _classify_option(self, text: str) -> str:
        """
        判断单个选项是数值型还是文本型

        去掉常见单位后，判断剩余部分是否为纯数字。

        Args:
            text: 选项文本

        Returns:
            "numeric" 或 "textual"
        """
        cleaned = text.strip()
        # 去除常见单位
        cleaned = re.sub(
            r"(亿元|万亿|亿|万|元|百分比|百分点|个点|%)", "", cleaned
        ).strip()
        # 去除首尾标点
        cleaned = cleaned.strip("，。、；：")
        # 判断是否为数字（含负号、小数点、千分位逗号）
        if re.match(r"^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$", cleaned) or \
           re.match(r"^[-+]?\d+(?:\.\d+)?$", cleaned):
            return "numeric"
        return "textual"

    def _classify_option_set(self, options: List[Dict[str, Any]]) -> str:
        """
        判断选项集类型

        保守策略：只要有一个非数值选项就归为 textual，避免误剥离。

        Args:
            options: 选项列表

        Returns:
            "numeric": 全部选项为数值（剥离选项，独立计算）
            "textual": 含文本表述（保留选项，逐条验证）
            "none":    无选项
        """
        if not options:
            return "none"

        numeric_count = sum(1 for o in options if o.get("type") == "numeric")
        if numeric_count == len(options):
            return "numeric"
        return "textual"

    def _build_enhanced_query(
        self, clean_query: str, options: List[Dict[str, Any]]
    ) -> str:
        """
        将问题主干与选项文本拼接为增强检索查询

        用于事实判断型问题：选项文本（"750日移动平均""流动性溢价"等）
        是检索锚点，拼接后 BM25/Dense 可命中包含这些具体表述的 chunk。

        Args:
            clean_query: 去除选项后的问题主干
            options: 选项列表

        Returns:
            增强后的检索查询
        """
        option_texts = [opt.get("text", "") for opt in options if opt.get("text")]
        stem = clean_query.rstrip("？?。")
        enhanced = f"{stem} {' '.join(option_texts)}"
        return enhanced

    # ============================================================
    # 辅助方法
    # ============================================================

    def _collect_rewrites(
        self,
        original: str,
        contextualized: str,
        expanded: str,
        channel_queries: Dict[str, str],
    ) -> List[str]:
        """
        收集所有改写版本（去重）

        按优先级收集: 指代消解版 → 同义词扩展版 → 各通道查询版。
        确保原始查询始终包含在结果中。

        Args:
            original: 原始查询
            contextualized: 指代消解后的查询
            expanded: 同义词扩展后的查询
            channel_queries: 各通道查询

        Returns:
            去重后的改写版本列表
        """
        rewrites: List[str] = []

        # 指代消解版（与原始不同时才添加）
        if contextualized != original:
            rewrites.append(contextualized)

        # 同义词扩展版
        if expanded != contextualized and expanded not in rewrites:
            rewrites.append(expanded)

        # 各通道查询版
        for cq in channel_queries.values():
            if cq not in rewrites:
                rewrites.append(cq)

        # 确保原始查询始终在列表中
        if original not in rewrites:
            rewrites.insert(0, original)

        return rewrites
