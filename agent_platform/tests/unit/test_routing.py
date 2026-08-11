"""
路由模块单元测试

测试用例:
  - 条款查询 L1 → channels=[exact,metadata], rerank=false
  - 阈值查询 L2 → channels=[lexical,dense,metadata], rerank=true
  - 比较查询 L3 → level=L3, need_decomposition=true
  - 问候 → level=L0, 无检索通道
  - 歧义时复杂度升级
"""

import pytest

from agent_platform.query_understanding import QuerySpecBuilder
from agent_platform.routing.rule_router import (
    ComplexityRouter,
    RouteDecision,
    RouteTable,
    RuleRouter,
)


class TestRuleRouter:
    """规则路由器测试"""

    def setup_method(self):
        self.router = RuleRouter()
        self.spec_builder = QuerySpecBuilder()

    def test_clause_query_l1(self):
        """条款查询路由到 L1"""
        spec = self.spec_builder.build("《商业银行资本管理办法》第43条")
        decision = self.router.route(spec)
        assert decision.level == "L1"
        assert "exact" in decision.channels
        assert "metadata" in decision.channels
        assert decision.rerank is False
        assert decision.budget_ms == 2000

    def test_threshold_query_l2(self):
        """阈值查询路由到 L2"""
        spec = self.spec_builder.build("核心一级资本充足率最低要求是多少")
        decision = self.router.route(spec)
        assert decision.level == "L2"
        assert "lexical" in decision.channels
        assert "dense" in decision.channels
        assert decision.rerank is True

    def test_comparison_l3(self):
        """比较查询路由到 L3"""
        spec = self.spec_builder.build("比较核心一级资本充足率和一级资本充足率的区别")
        decision = self.router.route(spec)
        assert decision.level == "L3"
        assert decision.need_decomposition is True

    def test_greeting_l0(self):
        """问候路由到 L0"""
        spec = self.spec_builder.build("你好")
        decision = self.router.route(spec)
        assert decision.level == "L0"
        assert decision.channels == []

    def test_compliance_l4(self):
        """合规查询路由到 L4"""
        spec = self.spec_builder.build("银行是否符合资本充足率要求")
        decision = self.router.route(spec)
        assert decision.level == "L4"
        assert decision.need_decomposition is True

    def test_ambiguity_upgrades_complexity(self):
        """歧义导致复杂度升级"""
        spec = self.spec_builder.build("那个比例是多少")
        decision = self.router.route(spec)
        # 有歧义时应该需要澄清
        if spec.ambiguities:
            assert decision.need_clarification is True

    def test_route_decision_to_dict(self):
        """路由决策序列化"""
        spec = self.spec_builder.build("第43条")
        decision = self.router.route(spec)
        d = decision.to_dict()
        assert "intent" in d
        assert "level" in d
        assert "channels" in d

    def test_table_lookup_routing(self):
        """表格取数路由"""
        spec = self.spec_builder.build("附件1表中的数据")
        decision = self.router.route(spec)
        assert "table" in decision.channels
        assert "metadata" in decision.channels


class TestComplexityRouter:
    """复杂度路由器测试"""

    def setup_method(self):
        self.complexity_router = ComplexityRouter()

    def test_greeting_is_l0(self):
        """问候是 L0"""
        level = self.complexity_router.route("greeting")
        assert level == "L0"

    def test_clause_query_is_l1(self):
        """条款查询是 L1"""
        level = self.complexity_router.route("clause_query")
        assert level == "L1"

    def test_threshold_is_l2(self):
        """阈值查询是 L2"""
        level = self.complexity_router.route("threshold")
        assert level == "L2"

    def test_comparison_is_l3(self):
        """比较查询是 L3"""
        level = self.complexity_router.route("comparison")
        assert level == "L3"

    def test_compliance_is_l4(self):
        """合规查询是 L4"""
        level = self.complexity_router.route("compliance")
        assert level == "L4"

    def test_ambiguity_upgrades_level(self):
        """歧义导致复杂度升级"""
        level = self.complexity_router.route("threshold", has_ambiguities=True)
        assert level == "L3"  # L2 → L3

    def test_greeting_no_upgrade_with_ambiguity(self):
        """L0 不因歧义升级"""
        level = self.complexity_router.route("greeting", has_ambiguities=True)
        assert level == "L0"

    def test_needs_clarification_with_ambiguity(self):
        """有歧义时需要澄清"""
        assert self.complexity_router.needs_clarification("threshold", True) is True

    def test_no_clarification_without_ambiguity(self):
        """无歧义时不需要澄清"""
        assert self.complexity_router.needs_clarification("threshold", False) is False

    def test_greeting_no_clarification(self):
        """问候不需要澄清"""
        assert self.complexity_router.needs_clarification("greeting", True) is False


class TestRouteTable:
    """路由决策表测试"""

    def test_get_known_intent(self):
        """获取已知意图的路由"""
        table = RouteTable()
        decision = table.get("clause_query")
        assert decision.intent == "clause_query"
        assert decision.level == "L1"

    def test_get_unknown_intent(self):
        """未知意图返回默认"""
        table = RouteTable()
        decision = table.get("nonexistent_intent")
        assert decision.intent == "unknown"

    def test_list_intents(self):
        """列出所有意图"""
        table = RouteTable()
        intents = table.list_intents()
        assert "clause_query" in intents
        assert "threshold" in intents
        assert "greeting" in intents

    def test_update_route(self):
        """更新路由"""
        table = RouteTable()
        new_decision = RouteDecision(
            intent="custom",
            level="L2",
            channels=["hybrid"],
        )
        table.update("custom", new_decision)
        assert table.get("custom").level == "L2"
