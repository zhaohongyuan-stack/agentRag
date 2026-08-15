"""
阶段 3: LangGraph 图编排流程集成测试

验证 handler.py 在 USE_LANGGRAPH=True 时走 langgraph_graph 图编排，
行为与手写 _run_agent_flow 等价:
  1. 正常图流程（Planner proceed → Retriever → Evaluator sufficient → Verifier verified）
  2. 越界检测（Planner out_of_domain → 直接拒答）
  3. Loop 多轮检索（Evaluator insufficient → retrieval_suggestion → 重试）
  4. Verifier needs_retry 触发补充检索
  5. 图流程异常自动回退手写 _run_agent_flow
  6. USE_LANGGRAPH=False 时不构建图（走原手写流程）

复用 test_agent_loop_handler 的 AgentMockLLM 与预设响应。
"""

import pytest

from agent_platform.agents import EvaluatorAgent, PlannerAgent, VerifierAgent
from agent_platform.gateway.request_handler import QueryRequest, RequestHandler
from agent_platform.gateway.request_handler import handler as handler_mod
from agent_platform.gateway.request_handler.handler import RequestHandler as HandlerCls

from .test_agent_loop_handler import (
    AgentMockLLM,
    _default_evaluator_insufficient,
    _default_evaluator_sufficient,
    _default_planner_out_of_domain,
    _default_planner_proceed,
    _default_verifier_needs_retry,
    _default_verifier_verified,
)


@pytest.fixture
def enable_langgraph(monkeypatch):
    """开启 USE_LANGGRAPH feature flag（测试结束后自动还原）"""
    monkeypatch.setattr(handler_mod, "USE_LANGGRAPH", True)


def _make_handler(mock_llm: AgentMockLLM) -> RequestHandler:
    """构建使用自定义 MockLLM 的 handler"""
    return HandlerCls(
        planner_agent=PlannerAgent(llm_client=mock_llm),
        evaluator_agent=EvaluatorAgent(llm_client=mock_llm),
        verifier_agent=VerifierAgent(llm_client=mock_llm),
        enable_agent_loop=True,
    )


class TestLangGraphFlow:
    """LangGraph 图编排流程集成测试"""

    # ──────────────────────────────────────────────────────
    # 场景1: 正常图流程
    # ──────────────────────────────────────────────────────

    def test_graph_normal_flow(self, enable_langgraph):
        """Planner proceed → Retriever → Evaluator sufficient → Verifier verified"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = _make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Agent 都被调用（与手写流程一致）
        assert mock_llm.planner_calls == 1, "Planner应被调用1次"
        assert mock_llm.evaluator_calls == 1, "Evaluator应被调用1次"
        assert mock_llm.verifier_calls == 1, "Verifier应被调用1次"

        # 状态轨迹与手写流程一致（状态机保留为 trace 记录器）
        trace = response.state_trace
        assert "PLANNING" in trace, f"状态轨迹应包含PLANNING: {trace}"
        assert "RETRIEVING" in trace
        assert "EVIDENCE_VALIDATING" in trace
        assert "GENERATING" in trace
        assert "ANSWER_VALIDATING" in trace
        assert trace[-1] == "RESPONDING"

        # 图确实被执行（编译后的图缓存在 handler 上）
        assert getattr(handler, "_langgraph_graph", None) is not None

        assert response.answer
        assert len(response.answer) > 0

    # ──────────────────────────────────────────────────────
    # 场景2: 越界检测 — Planner 判定越界直接拒答
    # ──────────────────────────────────────────────────────

    def test_graph_out_of_domain_refusal(self, enable_langgraph):
        """Planner 判定越界 → 直接拒答，不进入检索"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_out_of_domain(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = _make_handler(mock_llm)

        request = QueryRequest(query="今天北京天气怎么样")
        response = handler.handle_query(request)

        # Planner 被调用，但 Evaluator/Verifier 不应被调用
        assert mock_llm.planner_calls == 1
        assert mock_llm.evaluator_calls == 0, "越界时不应调用Evaluator"
        assert mock_llm.verifier_calls == 0, "越界时不应调用Verifier"

        # 越界拒答需经 4 个 skip 状态到达 REFUSING（状态机硬约束）
        trace = response.state_trace
        assert "PLANNING" in trace
        assert "REFUSING" in trace
        assert "RESPONDING" in trace
        assert response.is_refusal is True

    # ──────────────────────────────────────────────────────
    # 场景3: Loop 多轮检索 — Evaluator 不足 → 重试
    # ──────────────────────────────────────────────────────

    def test_graph_evaluator_insufficient_triggers_loop(self, enable_langgraph):
        """Evaluator 判定不足 → 图的条件边驱动多轮检索 Loop"""

        class LoopMockLLM(AgentMockLLM):
            """第1轮评估不足触发重试，第2轮评估充分退出 Loop"""

            def __init__(self):
                super().__init__(
                    planner_response=_default_planner_proceed(),
                    evaluator_response=_default_evaluator_insufficient(),
                    verifier_response=_default_verifier_verified(),
                )
                self._evaluator_call_count = 0

            def chat_json(self, messages, model=None, temperature=0.1, max_tokens=2048):
                system_content = ""
                for msg in messages:
                    if msg.role == "system":
                        system_content = msg.content
                        break

                if "评估Agent" in system_content:
                    self._evaluator_call_count += 1
                    self.evaluator_calls += 1
                    if self._evaluator_call_count == 1:
                        return _default_evaluator_insufficient()
                    return _default_evaluator_sufficient()
                return super().chat_json(messages, model, temperature, max_tokens)

        mock_llm = LoopMockLLM()
        handler = _make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Evaluator 应被调用至少2次（第1轮不足 → 重试 → 第2轮充分）
        assert mock_llm.evaluator_calls >= 2, (
            f"Evaluator应被调用>=2次，实际{mock_llm.evaluator_calls}"
        )

        # 状态轨迹应包含多次 RETRIEVING（Loop 重试）
        trace = response.state_trace
        retry_count = trace.count("RETRIEVING")
        assert retry_count >= 2, f"应有多轮检索（>=2次RETRIEVING），实际{retry_count}"

        assert trace[-1] == "RESPONDING"
        assert response.answer

    # ──────────────────────────────────────────────────────
    # 场景4: Verifier needs_retry 触发补充检索
    # ──────────────────────────────────────────────────────

    def test_graph_verifier_needs_retry(self, enable_langgraph):
        """Verifier needs_retry=True → verifier_retry 节点补充检索 + 重新生成"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_needs_retry(),
        )
        handler = _make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Verifier 应被调用（至少1次；有新证据时重新验证为2次）
        assert mock_llm.verifier_calls >= 1

        # 状态轨迹应包含 RETRYING（Verifier 触发的重试）
        trace = response.state_trace
        assert "RETRYING" in trace, f"应包含RETRYING状态: {trace}"

        # 补充检索最多 1 次（verifier_retry_done 硬约束）
        assert trace.count("RETRYING") <= 1, "Verifier补充检索最多1次"

        assert trace[-1] == "RESPONDING"
        assert response.answer

    # ──────────────────────────────────────────────────────
    # 场景5: 图流程异常 → 自动回退手写 Agent 流程
    # ──────────────────────────────────────────────────────

    def test_graph_exception_fallback_to_agent_flow(self, enable_langgraph, monkeypatch):
        """build_graph 失败 → 回退手写 _run_agent_flow 仍能正常响应"""
        import agent_platform.orchestration.langgraph_graph as lg_mod

        def _broken_build_graph():
            raise RuntimeError("图构建故障测试")

        monkeypatch.setattr(lg_mod, "build_graph", _broken_build_graph)

        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = _make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # 回退手写流程后 Agent 正常执行
        assert mock_llm.planner_calls == 1, "应回退到手写Agent流程并调用Planner"
        assert response.answer
        assert response.state_trace[-1] == "RESPONDING"

    # ──────────────────────────────────────────────────────
    # 场景6: 开关关闭时不构建图（默认行为不变）
    # ──────────────────────────────────────────────────────

    def test_flag_off_keeps_handwritten_flow(self):
        """USE_LANGGRAPH=False（默认）→ 走手写流程，不编译图"""
        assert handler_mod.USE_LANGGRAPH is False, "测试环境默认不应开启LangGraph"

        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = _make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        assert response.state_trace[-1] == "RESPONDING"
        # 手写流程不应触发图编译
        assert getattr(handler, "_langgraph_graph", None) is None
