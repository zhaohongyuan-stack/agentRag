"""
多轮指代消解器

在多轮对话中解析用户问题里的代词和模糊指代，将其替换为上下文中的具体实体。

消解类型:
  - 指标指代: "这个比例" / "该指标" → 上下文中提到的具体监管指标
  - 文档指代: "那个文件" / "该规定" → 上下文中提到的具体法规文档
  - 条款指代: "那条" / "该条款" → 上下文中提到的具体条款号
  - 主题指代: "之前提到的" / "刚才说的" → 上下文中提到的指标或文档

消解策略:
  1. 检测查询中的指代词，判定指代类型（指标/文档/条款/主题）
  2. 从会话上下文中提取对应类型的候选实体
  3. 候选唯一时直接替换；候选多个时标记歧义；无候选时标记无法消解

用法:
    resolver = ReferenceResolver()
    ctx = SessionContext(mentioned_metrics=["核心一级资本充足率"])
    result = resolver.resolve_detailed("这个比例适用吗", ctx)
    # result.resolved_query = "核心一级资本充足率适用吗"
    # result.was_resolved = True
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 指代词模式表
# (模式名, 正则, 指代类型)
# 顺序敏感: 更具体的模式排在前面，优先匹配
# ============================================================
REFERENCE_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # ── 指标指代 ── 指向 mentioned_metrics
    (
        "metric_ref",
        re.compile(
            r"这个比例|这个指标|这个比率|那个比例|那个指标|那个比率"
            r"|该比例|该指标|该比率|此比例|此指标|此比率"
            r"|这个数|这个值|那个数|那个值"
        ),
        "metric",
    ),
    # ── 文档指代 ── 指向 mentioned_docs
    (
        "doc_ref",
        re.compile(
            r"那个文件|那个规定|那个制度|那部法规|那部法律"
            r"|该文件|该规定|该制度|该法规"
            r"|这个文件|这个规定|这个制度"
            r"|此文件|此规定|此制度"
        ),
        "doc",
    ),
    # ── 条款指代 ── 指向 previous_queries 中的条款号
    (
        "clause_ref",
        re.compile(r"那条|该条|这个条款|那个条款|该条款|此条|此条款"),
        "clause",
    ),
    # ── 主题指代 ── 指向 mentioned_metrics + mentioned_docs
    (
        "topic_ref",
        re.compile(
            r"之前提到的|之前说的|刚才提到的|刚才说的"
            r"|上面提到的|上面说的|前文提到的|前面提到的"
        ),
        "topic",
    ),
]

# 条款号正则 — 用于从历史查询中提取条款号
CLAUSE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零\d]+条")


@dataclass
class SessionContext:
    """
    会话上下文 — 多轮对话的历史信息

    由会话管理器维护，记录前序对话中出现的查询、实体、指标和文档。
    指代消解器据此解析当前查询中的代词和模糊指代。
    """

    previous_queries: List[str] = field(default_factory=list)
    """前序用户查询文本列表"""

    previous_entities: List[dict] = field(default_factory=list)
    """前序查询抽取的实体列表，每项为 {entity_type, value, ...}"""

    mentioned_metrics: List[str] = field(default_factory=list)
    """对话中提到过的监管指标名称，如 ['核心一级资本充足率', '杠杆率']"""

    mentioned_docs: List[str] = field(default_factory=list)
    """对话中提到过的法规文档名称，如 ['商业银行资本管理办法']"""

    def to_dict(self) -> dict:
        return {
            "previous_queries": self.previous_queries,
            "previous_entities": self.previous_entities,
            "mentioned_metrics": self.mentioned_metrics,
            "mentioned_docs": self.mentioned_docs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionContext":
        """从字典构建 SessionContext"""
        return cls(
            previous_queries=d.get("previous_queries", []),
            previous_entities=d.get("previous_entities", []),
            mentioned_metrics=d.get("mentioned_metrics", []),
            mentioned_docs=d.get("mentioned_docs", []),
        )


@dataclass
class ResolutionResult:
    """
    指代消解结果

    包含消解后的查询文本，以及消解过程中是否检测到歧义等信息。
    """

    resolved_query: str
    """消解后的查询文本（未消解时与原始查询相同）"""

    was_resolved: bool
    """是否成功消解了指代"""

    ambiguity_flagged: bool
    """是否因候选不唯一而标记歧义"""

    ambiguity_reason: str
    """歧义原因描述（无歧义时为空字符串）"""

    resolved_entity: Optional[str] = None
    """解析到的具体实体名称（成功消解时非空）"""

    reference_type: str = ""
    """指代类型: metric / doc / clause / topic"""

    def to_dict(self) -> dict:
        return {
            "resolved_query": self.resolved_query,
            "was_resolved": self.was_resolved,
            "ambiguity_flagged": self.ambiguity_flagged,
            "ambiguity_reason": self.ambiguity_reason,
            "resolved_entity": self.resolved_entity,
            "reference_type": self.reference_type,
        }


class ReferenceResolver:
    """
    多轮指代消解器

    从会话上下文中解析代词和模糊指代，替换为具体实体名称。
    当上下文中存在多个候选时，标记歧义而不强行替换。
    """

    def resolve(self, query: str, session_context: SessionContext) -> str:
        """
        指代消解，返回消解后的查询文本

        Args:
            query: 用户当前查询
            session_context: 会话上下文

        Returns:
            消解后的查询文本。无法消解时返回原始查询。
        """
        result = self.resolve_detailed(query, session_context)
        return result.resolved_query

    def resolve_detailed(
        self, query: str, session_context: SessionContext
    ) -> ResolutionResult:
        """
        详细指代消解，返回包含歧义信息的完整结果

        Args:
            query: 用户当前查询
            session_context: 会话上下文

        Returns:
            ResolutionResult 对象
        """
        # 空查询或无上下文
        if not query or not query.strip():
            return ResolutionResult(
                resolved_query=query or "",
                was_resolved=False,
                ambiguity_flagged=False,
                ambiguity_reason="",
            )

        if session_context is None:
            return ResolutionResult(
                resolved_query=query,
                was_resolved=False,
                ambiguity_flagged=False,
                ambiguity_reason="",
            )

        query_stripped = query.strip()

        # 1. 检测指代类型
        ref_type, ref_match = self._detect_reference(query_stripped)

        if ref_type is None:
            # 无指代词，无需消解
            return ResolutionResult(
                resolved_query=query,
                was_resolved=False,
                ambiguity_flagged=False,
                ambiguity_reason="",
            )

        # 2. 获取候选实体
        candidates = self._get_candidates(ref_type, session_context)

        if not candidates:
            # 无候选实体，无法消解
            return ResolutionResult(
                resolved_query=query,
                was_resolved=False,
                ambiguity_flagged=True,
                ambiguity_reason=(
                    f"指代词 '{ref_match}' 无法从上下文中找到对应的"
                    f"{'指标' if ref_type == 'metric' else '文档' if ref_type == 'doc' else '条款' if ref_type == 'clause' else '实体'}"
                ),
                reference_type=ref_type,
            )

        if len(candidates) == 1:
            # 唯一候选，直接替换
            entity = candidates[0]
            resolved = self._replace_reference(query_stripped, ref_match, entity)
            return ResolutionResult(
                resolved_query=resolved,
                was_resolved=True,
                ambiguity_flagged=False,
                ambiguity_reason="",
                resolved_entity=entity,
                reference_type=ref_type,
            )

        # 多个候选，标记歧义
        return ResolutionResult(
            resolved_query=query,
            was_resolved=False,
            ambiguity_flagged=True,
            ambiguity_reason=(
                f"指代词 '{ref_match}' 在上下文中有多个候选: {', '.join(candidates)}"
            ),
            reference_type=ref_type,
        )

    def _detect_reference(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        检测查询中的指代词

        按模式表顺序匹配，返回第一个命中的指代类型和匹配文本。

        Args:
            query: 查询文本

        Returns:
            (指代类型, 匹配到的指代词文本)，无匹配时为 (None, None)
        """
        for _pattern_name, pattern, ref_type in REFERENCE_PATTERNS:
            match = pattern.search(query)
            if match:
                return ref_type, match.group(0)
        return None, None

    def _get_candidates(
        self, ref_type: str, session_context: SessionContext
    ) -> List[str]:
        """
        根据指代类型从会话上下文中获取候选实体

        Args:
            ref_type: 指代类型 (metric / doc / clause / topic)
            session_context: 会话上下文

        Returns:
            候选实体列表（去重）
        """
        candidates: List[str] = []

        if ref_type == "metric":
            candidates.extend(session_context.mentioned_metrics)
            # 从 previous_entities 中补充
            candidates.extend(self._extract_from_entities(
                session_context.previous_entities, "metric_name"
            ))

        elif ref_type == "doc":
            candidates.extend(session_context.mentioned_docs)
            candidates.extend(self._extract_from_entities(
                session_context.previous_entities, "doc_name"
            ))

        elif ref_type == "clause":
            candidates.extend(self._extract_clauses_from_history(session_context))

        elif ref_type == "topic":
            # 主题指代: 合并指标和文档
            candidates.extend(session_context.mentioned_metrics)
            candidates.extend(session_context.mentioned_docs)
            candidates.extend(self._extract_from_entities(
                session_context.previous_entities, "metric_name"
            ))
            candidates.extend(self._extract_from_entities(
                session_context.previous_entities, "doc_name"
            ))

        # 去重并过滤空值
        seen: set = set()
        unique: List[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _extract_from_entities(
        self, entities: List[dict], entity_type: str
    ) -> List[str]:
        """从实体列表中提取指定类型的实体值"""
        result = []
        for entity in entities:
            if entity.get("entity_type") == entity_type:
                value = entity.get("value", "")
                if value:
                    result.append(value)
        return result

    def _extract_clauses_from_history(
        self, session_context: SessionContext
    ) -> List[str]:
        """从历史查询和实体中提取条款号"""
        clauses: List[str] = []

        # 从 previous_queries 中提取
        for prev_query in session_context.previous_queries:
            for match in CLAUSE_PATTERN.finditer(prev_query):
                clause = match.group(0)
                if clause not in clauses:
                    clauses.append(clause)

        # 从 previous_entities 中提取
        for entity in session_context.previous_entities:
            if entity.get("entity_type") == "clause_number":
                value = entity.get("value", "")
                if value:
                    clause_ref = f"第{value}条"
                    if clause_ref not in clauses:
                        clauses.append(clause_ref)

        return clauses

    def _replace_reference(
        self, query: str, ref_text: str, entity: str
    ) -> str:
        """
        将查询中的指代词替换为具体实体

        Args:
            query: 原始查询
            ref_text: 匹配到的指代词
            entity: 替换目标实体

        Returns:
            替换后的查询文本
        """
        return query.replace(ref_text, entity, 1)
