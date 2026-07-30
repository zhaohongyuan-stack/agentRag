"""
多轮指代消解器（增强版）— M5.2 记忆模块

基于 MemoryContext 进行指代消解，利用三类记忆提供的丰富上下文。

与 query_understanding.query_rewriter.reference_resolver 的区别:
  - 后者基于简单的 SessionContext（mentioned_metrics/docs + previous_queries）
  - 本模块基于完整的 MemoryContext（工作记忆 + 摘要 + 事实 + 最近轮次）
  - 支持基于摘要的远程指代消解（"之前提到的" 可解析到摘要中的实体）
  - 支持基于确认事实的指代消解

消解流程:
  1. 从 MemoryContext 获取上下文（指标、文档、实体、历史查询）
  2. 检测查询中的指代词
  3. 根据指代类型获取候选实体
  4. 候选唯一 → 替换；候选多个 → 标记歧义；无候选 → 标记无法消解
  5. 保留 derived_from_turn_ids 用于追溯
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ...query_understanding.query_rewriter.reference_resolver import (
    REFERENCE_PATTERNS,
    CLAUSE_PATTERN,
)
from ..memory_models import MemoryContext

logger = logging.getLogger(__name__)


@dataclass
class EnhancedResolutionResult:
    """
    增强指代消解结果

    在基础消解结果上增加来源信息，便于追溯。

    Attributes:
        resolved_query: 消解后的查询文本
        was_resolved: 是否成功消解
        ambiguity_flagged: 是否标记歧义
        ambiguity_reason: 歧义原因
        resolved_entity: 解析到的具体实体
        reference_type: 指代类型
        derived_from_turn_ids: 消解来源的轮次 ID 列表
        source: 消解来源类型 (working_memory / summary / facts / recent_turns)
    """

    resolved_query: str
    was_resolved: bool
    ambiguity_flagged: bool
    ambiguity_reason: str = ""
    resolved_entity: Optional[str] = None
    reference_type: str = ""
    derived_from_turn_ids: List[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "resolved_query": self.resolved_query,
            "was_resolved": self.was_resolved,
            "ambiguity_flagged": self.ambiguity_flagged,
            "ambiguity_reason": self.ambiguity_reason,
            "resolved_entity": self.resolved_entity,
            "reference_type": self.reference_type,
            "derived_from_turn_ids": list(self.derived_from_turn_ids),
            "source": self.source,
        }


class MemoryReferenceResolver:
    """
    基于 MemoryContext 的增强指代消解器

    利用三类记忆提供的丰富上下文进行指代消解。

    用法:
        resolver = MemoryReferenceResolver()
        ctx = await manager.get_context_for_new_query(session_id)
        result = resolver.resolve("这个比例适用于非系统重要性银行吗？", ctx)
        # result.resolved_query = "核心一级资本充足率适用于非系统重要性银行吗？"
    """

    def resolve(
        self, query: str, context: MemoryContext
    ) -> EnhancedResolutionResult:
        """
        指代消解

        Args:
            query: 用户当前查询
            context: 完整多轮记忆上下文

        Returns:
            EnhancedResolutionResult
        """
        # 空查询或无上下文
        if not query or not query.strip():
            return EnhancedResolutionResult(
                resolved_query=query or "",
                was_resolved=False,
                ambiguity_flagged=False,
            )

        if context is None:
            return EnhancedResolutionResult(
                resolved_query=query,
                was_resolved=False,
                ambiguity_flagged=False,
            )

        query_stripped = query.strip()

        # 1. 检测指代类型
        ref_type, ref_match = self._detect_reference(query_stripped)

        if ref_type is None:
            return EnhancedResolutionResult(
                resolved_query=query,
                was_resolved=False,
                ambiguity_flagged=False,
            )

        # 2. 获取候选实体及来源
        candidates, source, source_turn_ids = self._get_candidates_with_source(
            ref_type, context
        )

        if not candidates:
            return EnhancedResolutionResult(
                resolved_query=query,
                was_resolved=False,
                ambiguity_flagged=True,
                ambiguity_reason=self._build_no_candidate_reason(
                    ref_match, ref_type
                ),
                reference_type=ref_type,
            )

        if len(candidates) == 1:
            # 唯一候选，直接替换
            entity = candidates[0]
            resolved = query_stripped.replace(ref_match, entity, 1)
            return EnhancedResolutionResult(
                resolved_query=resolved,
                was_resolved=True,
                ambiguity_flagged=False,
                resolved_entity=entity,
                reference_type=ref_type,
                derived_from_turn_ids=source_turn_ids,
                source=source,
            )

        # 多个候选，标记歧义
        return EnhancedResolutionResult(
            resolved_query=query,
            was_resolved=False,
            ambiguity_flagged=True,
            ambiguity_reason=(
                f"指代词 '{ref_match}' 在上下文中有多个候选: "
                f"{', '.join(candidates)}"
            ),
            reference_type=ref_type,
            derived_from_turn_ids=source_turn_ids,
            source=source,
        )

    def resolve_to_session_context(self, context: MemoryContext):
        """
        将 MemoryContext 转换为 query_rewriter 的 SessionContext

        便于复用已有的 ReferenceResolver 逻辑。

        Args:
            context: MemoryContext

        Returns:
            SessionContext 对象
        """
        from ...query_understanding.query_rewriter.reference_resolver import (
            SessionContext,
        )

        return SessionContext(
            previous_queries=context.previous_queries,
            previous_entities=context.previous_entities,
            mentioned_metrics=context.mentioned_metrics,
            mentioned_docs=context.mentioned_docs,
        )

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _detect_reference(query: str) -> Tuple[Optional[str], Optional[str]]:
        """检测查询中的指代词"""
        for _name, pattern, ref_type in REFERENCE_PATTERNS:
            match = pattern.search(query)
            if match:
                return ref_type, match.group(0)
        return None, None

    def _get_candidates_with_source(
        self, ref_type: str, context: MemoryContext
    ) -> Tuple[List[str], str, List[str]]:
        """
        根据指代类型从 MemoryContext 获取候选实体

        优先级: 工作记忆 > 摘要 > 确认事实 > 最近轮次

        Returns:
            (候选实体列表, 来源类型, 来源轮次ID列表)
        """
        candidates: List[str] = []
        source = ""
        source_turn_ids: List[str] = []

        if ref_type == "metric":
            # 指标指代
            candidates, source, source_turn_ids = self._get_metric_candidates(
                context
            )

        elif ref_type == "doc":
            # 文档指代
            candidates, source, source_turn_ids = self._get_doc_candidates(
                context
            )

        elif ref_type == "clause":
            # 条款指代
            candidates, source, source_turn_ids = self._get_clause_candidates(
                context
            )

        elif ref_type == "topic":
            # 主题指代: 合并指标和文档
            metrics, m_source, m_ids = self._get_metric_candidates(context)
            docs, d_source, d_ids = self._get_doc_candidates(context)
            candidates = metrics + docs
            source = f"{m_source}+{d_source}" if m_source and d_source else (m_source or d_source)
            source_turn_ids = list(set(m_ids + d_ids))

        # 去重保序
        seen: set = set()
        unique: List[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)

        return unique, source, source_turn_ids

    @staticmethod
    def _get_metric_candidates(
        context: MemoryContext,
    ) -> Tuple[List[str], str, List[str]]:
        """获取指标候选"""
        # 优先从工作记忆获取
        if context.working_memory and context.working_memory.mentioned_metrics:
            return (
                list(context.working_memory.mentioned_metrics),
                "working_memory",
                [],
            )

        # 其次从摘要获取
        if context.summary and context.summary.key_metrics:
            return (
                list(context.summary.key_metrics),
                "summary",
                [],
            )

        # 从最近轮次的实体中获取
        metrics: List[str] = []
        turn_ids: List[str] = []
        for turn in context.recent_turns:
            for entity in turn.entities:
                if entity.get("entity_type") == "metric_name":
                    val = entity.get("value", "")
                    if val and val not in metrics:
                        metrics.append(val)
                        if turn.turn_id not in turn_ids:
                            turn_ids.append(turn.turn_id)

        if metrics:
            return metrics, "recent_turns", turn_ids

        return [], "", []

    @staticmethod
    def _get_doc_candidates(
        context: MemoryContext,
    ) -> Tuple[List[str], str, List[str]]:
        """获取文档候选"""
        # 优先从工作记忆获取
        if context.working_memory and context.working_memory.mentioned_docs:
            return (
                list(context.working_memory.mentioned_docs),
                "working_memory",
                [],
            )

        # 其次从摘要获取
        if context.summary and context.summary.key_docs:
            return (
                list(context.summary.key_docs),
                "summary",
                [],
            )

        # 从最近轮次的实体中获取
        docs: List[str] = []
        turn_ids: List[str] = []
        for turn in context.recent_turns:
            for entity in turn.entities:
                if entity.get("entity_type") == "doc_name":
                    val = entity.get("value", "")
                    if val and val not in docs:
                        docs.append(val)
                        if turn.turn_id not in turn_ids:
                            turn_ids.append(turn.turn_id)

        if docs:
            return docs, "recent_turns", turn_ids

        return [], "", []

    @staticmethod
    def _get_clause_candidates(
        context: MemoryContext,
    ) -> Tuple[List[str], str, List[str]]:
        """获取条款候选"""
        clauses: List[str] = []
        turn_ids: List[str] = []

        for turn in context.recent_turns:
            # 从查询文本中提取条款号
            for match in CLAUSE_PATTERN.finditer(turn.query):
                clause = match.group(0)
                if clause not in clauses:
                    clauses.append(clause)
                    if turn.turn_id not in turn_ids:
                        turn_ids.append(turn.turn_id)

            # 从实体中提取条款号
            for entity in turn.entities:
                if entity.get("entity_type") == "clause_number":
                    val = entity.get("value", "")
                    if val:
                        clause_ref = f"第{val}条"
                        if clause_ref not in clauses:
                            clauses.append(clause_ref)
                            if turn.turn_id not in turn_ids:
                                turn_ids.append(turn.turn_id)

        source = "recent_turns" if clauses else ""
        return clauses, source, turn_ids

    @staticmethod
    def _build_no_candidate_reason(ref_match: str, ref_type: str) -> str:
        """构建无候选实体的原因描述"""
        type_label = {
            "metric": "指标",
            "doc": "文档",
            "clause": "条款",
            "topic": "实体",
        }.get(ref_type, "实体")
        return f"指代词 '{ref_match}' 无法从上下文中找到对应的{type_label}"
