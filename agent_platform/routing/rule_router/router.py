"""
规则路由器 — 基于规则的请求级和问题类型路由

三层路由:
  第一层 — 请求级路由: 问候→直接回复；知识库查询→RAG；歧义→澄清
  第二层 — 问题类型路由: 根据意图选择检索通道
  第三层 — 复杂度路由: L0-L4 级别，决定执行路径

输入: QuerySpec
输出: RouteDecision（包含通道、预算、复杂度、是否需澄清/拆解）
"""

from typing import Any, Dict, Optional

from .complexity_router import ComplexityRouter
from .route_table import RouteDecision, RouteTable


class RuleRouter:
    """
    规则路由器

    接收 QuerySpec，输出 RouteDecision。
    纯规则驱动，不依赖 LLM（Q1.4 确认）。
    """

    def __init__(self, route_table: Optional[RouteTable] = None):
        """
        Args:
            route_table: 路由决策表，为 None 时使用内置默认值
        """
        self._route_table = route_table or RouteTable()
        self._complexity_router = ComplexityRouter()

    def route(self, query_spec: Any) -> RouteDecision:
        """
        对 QuerySpec 进行路由决策

        Args:
            query_spec: QuerySpec 对象（或兼容的 dict）

        Returns:
            RouteDecision 对象
        """
        # 兼容 QuerySpec 对象和 dict
        if hasattr(query_spec, "intent"):
            intent = query_spec.intent
            ambiguities = query_spec.ambiguities
            entities = query_spec.entities
            complexity = query_spec.complexity
        else:
            intent = query_spec.get("intent", "unknown")
            ambiguities = query_spec.get("ambiguities", [])
            entities = query_spec.get("entities", [])
            complexity = query_spec.get("complexity", "L2")

        has_ambiguities = len(ambiguities) > 0
        entity_count = len(entities)

        # 从路由表获取基础决策
        decision = self._route_table.get(intent)

        # 复杂度路由（可能升级）
        final_complexity = self._complexity_router.route(
            intent=intent,
            has_ambiguities=has_ambiguities,
            entity_count=entity_count,
        )

        # 是否需要澄清
        need_clarification = self._complexity_router.needs_clarification(
            intent, has_ambiguities
        )

        # 是否需要拆解
        need_decomposition = self._complexity_router.needs_decomposition(
            intent, final_complexity
        )

        # 构建最终决策
        return RouteDecision(
            intent=intent,
            level=final_complexity,
            channels=decision.channels,
            top_k=decision.top_k,
            rerank=decision.rerank,
            rerank_top_n=decision.rerank_top_n,
            budget_ms=decision.budget_ms,
            need_clarification=need_clarification,
            need_decomposition=need_decomposition,
            description=decision.description,
        )
