"""
复杂度路由器

基于纯规则的复杂度评级（L0-L4），不依赖 LLM 判定。

评级规则:
  L0: 问候/打招呼 → 直接回复，无需检索
  L1: 条款查询 → 精确文号/条款号，单路检索
  L2: 定义/阈值/表格 → 普通事实查询，混合检索
  L3: 比较查询 → 需拆解为多个子问题，多路并行
  L4: 合规/复杂查询 → 多跳推理，完整 DAG 执行

来源: 问题确认.md Q1.4 用户确认"纯规则匹配，LLM不参与判定"
"""

from typing import List, Optional

from ..rule_router.route_table import DEFAULT_ROUTE_TABLE


class ComplexityRouter:
    """
    复杂度路由器 — 纯规则版

    根据 QuerySpec 的意图、实体、歧义情况评定复杂度级别。
    """

    def __init__(self):
        pass

    def route(
        self,
        intent: str,
        has_ambiguities: bool = False,
        entity_count: int = 0,
    ) -> str:
        """
        评定复杂度级别

        Args:
            intent: 查询意图
            has_ambiguities: 是否存在歧义
            entity_count: 抽取到的实体数量

        Returns:
            复杂度级别字符串（L0-L4）
        """
        # 从默认路由表获取基础复杂度
        base_decision = DEFAULT_ROUTE_TABLE.get(intent, DEFAULT_ROUTE_TABLE["unknown"])
        level = base_decision.level

        # 歧义存在时提升一级（L0 除外）
        if has_ambiguities and level != "L0":
            level_num = int(level[1:])
            level_num = min(level_num + 1, 4)
            level = f"L{level_num}"

        return level

    def needs_clarification(self, intent: str, has_ambiguities: bool) -> bool:
        """
        判断是否需要澄清

        Args:
            intent: 查询意图
            has_ambiguities: 是否存在歧义

        Returns:
            True 如果需要先向用户澄清
        """
        # L0 问候不需要澄清
        if intent == "greeting":
            return False

        # 有歧义时需要澄清
        return has_ambiguities

    def needs_decomposition(self, intent: str, complexity: str) -> bool:
        """
        判断是否需要问题拆解

        Args:
            intent: 查询意图
            complexity: 复杂度级别

        Returns:
            True 如果需要拆解为子问题
        """
        base_decision = DEFAULT_ROUTE_TABLE.get(intent, DEFAULT_ROUTE_TABLE["unknown"])
        if base_decision.need_decomposition:
            return True

        # L3+ 都需要拆解
        if complexity in ("L3", "L4"):
            return True

        return False
