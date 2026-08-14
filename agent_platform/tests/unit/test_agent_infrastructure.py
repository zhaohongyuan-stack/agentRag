"""
Agent 基础设施单元测试

测试 AgentContext 和 BaseAgent 的核心功能：
  1. AgentContext 初始化、序列化、Loop控制
  2. AgentResult 结构化输出
  3. BaseAgent Mock 模式下执行流程
"""

import json
import pytest

from agent_platform.agents import AgentContext, AgentResult, BaseAgent
from agent_platform.orchestration.budget_controller.controller import BudgetController
from agent_platform.runtime.llm_client import LLMClient, LLMMessage


# ============================================================
# AgentResult 测试
# ============================================================

class TestAgentResult:
    """AgentResult 结构化输出测试"""

    def test_basic_creation(self):
        """基本创建"""
        result = AgentResult(
            agent_name="Planner",
            decision="proceed",
            data={"intent": "clause_query"},
            latency_ms=150,
        )
        assert result.agent_name == "Planner"
        assert result.decision == "proceed"
        assert result.data["intent"] == "clause_query"
        assert result.latency_ms == 150
        assert result.success is True
        assert result.error == ""

    def test_to_dict(self):
        """序列化为字典"""
        result = AgentResult(
            agent_name="Evaluator",
            decision="retry",
            data={"score": 0.72},
            latency_ms=200,
        )
        d = result.to_dict()
        assert d["agent_name"] == "Evaluator"
        assert d["decision"] == "retry"
        assert d["data"]["score"] == 0.72
        assert d["latency_ms"] == 200
        assert d["success"] is True

    def test_error_result(self):
        """错误结果"""
        result = AgentResult(
            agent_name="Verifier",
            decision="error",
            data={},
            latency_ms=50,
            success=False,
            error="LLM调用超时",
        )
        assert result.success is False
        assert result.error == "LLM调用超时"


# ============================================================
# AgentContext 测试
# ============================================================

class TestAgentContext:
    """AgentContext 共享上下文测试"""

    def test_initialization(self):
        """初始化"""
        ctx = AgentContext(session_id="test-001", query="核心一级资本充足率")
        assert ctx.session_id == "test-001"
        assert ctx.query == "核心一级资本充足率"
        assert ctx.query_spec is None
        assert ctx.route_decision is None
        assert ctx.retrieval_plan == {}
        assert ctx.evidence_bundle is None
        assert ctx.evaluation_result == {}
        assert ctx.retrieval_suggestion == {}
        assert ctx.generated_answer is None
        assert ctx.verification_result == {}
        assert ctx.loop_count == 0
        assert ctx.budget_controller is None
        assert ctx.retrieval_history == []

    def test_increment_loop_without_budget(self):
        """无预算控制器时Loop递增"""
        ctx = AgentContext(session_id="test-002")
        action = ctx.increment_loop()
        assert ctx.loop_count == 1
        assert action == "continue"

    def test_increment_loop_with_budget(self):
        """有预算控制器时Loop递增"""
        ctx = AgentContext(
            session_id="test-003",
            budget_controller=BudgetController("P2"),
        )
        # P2 默认2轮检索
        action1 = ctx.increment_loop()
        assert ctx.loop_count == 1
        assert action1 in ("continue", "downgrade")

        action2 = ctx.increment_loop()
        assert ctx.loop_count == 2

        action3 = ctx.increment_loop()
        assert ctx.loop_count == 3
        assert action3 == "stop"  # 超过P2上限

    def test_can_continue_loop_without_budget(self):
        """无预算控制器时默认上限3轮"""
        ctx = AgentContext(session_id="test-004")
        ctx.loop_count = 2
        assert ctx.can_continue_loop() is True

        ctx.loop_count = 3
        assert ctx.can_continue_loop() is False

    def test_can_continue_loop_with_budget(self):
        """有预算控制器时检查剩余轮次"""
        ctx = AgentContext(
            session_id="test-005",
            budget_controller=BudgetController("P2"),
        )
        # P2 默认2轮
        assert ctx.can_continue_loop() is True
        ctx.budget_controller.consume_retrieval_round()
        ctx.budget_controller.consume_retrieval_round()
        assert ctx.can_continue_loop() is False

    def test_add_retrieval_record(self):
        """记录检索历史"""
        ctx = AgentContext(session_id="test-006")
        ctx.add_retrieval_record("核心一级资本充足率", 28, "hybrid", 800)
        ctx.add_retrieval_record("资本充足率 阈值", 15, "bm25", 600)

        assert len(ctx.retrieval_history) == 2
        assert ctx.retrieval_history[0]["round"] == 1
        assert ctx.retrieval_history[0]["query"] == "核心一级资本充足率"
        assert ctx.retrieval_history[0]["hits"] == 28
        assert ctx.retrieval_history[1]["round"] == 2
        assert ctx.retrieval_history[1]["strategy"] == "bm25"

    def test_to_trace_dict(self):
        """序列化为trace字典"""
        ctx = AgentContext(session_id="test-007", query="测试问题")
        ctx.loop_count = 1
        ctx.retrieval_plan = {"strategies": ["hybrid"]}
        ctx.evaluation_result = {"score": 0.85}
        ctx.add_retrieval_record("测试", 10, "hybrid", 500)

        trace = ctx.to_trace_dict()
        assert trace["session_id"] == "test-007"
        assert trace["query"] == "测试问题"
        assert trace["loop_count"] == 1
        assert trace["retrieval_plan"]["strategies"] == ["hybrid"]
        assert trace["evaluation_result"]["score"] == 0.85
        assert len(trace["retrieval_history"]) == 1
        assert trace["evidence_bundle"] is None
        assert trace["generated_answer"] is None


# ============================================================
# BaseAgent 测试
# ============================================================

class TestBaseAgent:
    """BaseAgent 基类测试"""

    def test_mock_agent_run(self):
        """Mock模式下BaseAgent执行流程"""

        class DummyAgent(BaseAgent):
            """测试用DummyAgent"""

            def _build_prompt(self, context):
                # system prompt含"回答"触发Mock的JSON回答分支
                return [
                    LLMMessage(role="system", content="你是测试Agent，请回答以下问题"),
                    LLMMessage(role="user", content=context.query),
                ]

            def _parse_response(self, response):
                decision = response.get("is_refusal", False) and "refuse" or "proceed"
                return decision, response

        # 使用Mock LLM客户端
        llm = LLMClient(mock=True)
        agent = DummyAgent(llm, "DummyAgent")
        ctx = AgentContext(session_id="test-agent-001", query="测试问题")

        result = agent.run(ctx)

        assert result.agent_name == "DummyAgent"
        assert result.success is True
        assert result.latency_ms >= 0
        assert isinstance(result.data, dict)

    def test_not_implemented(self):
        """未实现子类方法时报错"""
        llm = LLMClient(mock=True)
        agent = BaseAgent(llm, "Base")

        ctx = AgentContext(session_id="test-agent-002", query="测试")
        with pytest.raises(NotImplementedError):
            agent.run(ctx)
