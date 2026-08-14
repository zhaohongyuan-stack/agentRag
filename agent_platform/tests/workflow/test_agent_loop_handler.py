"""
Phase 5 Agent 协作 Loop 集成测试

验证 handler.py 的 Agent 协作主流程:
  1. 正常 Agent 协作流程（Planner proceed → Retriever → Evaluator sufficient → Verifier verified）
  2. 越界检测（Planner out_of_domain → 直接拒答）
  3. Agent 异常自动回退 V1 线性流程
  4. enable_agent_loop=False 走 V1 流程
  5. Loop 多轮检索（Evaluator insufficient → retrieval_suggestion → 重试）
  6. Verifier needs_retry 触发补充检索

测试通过自定义 MockLLM 控制 Agent 的 JSON 输出，
避免依赖默认 mock_chat 的非 JSON 响应。
"""

import json
from typing import Any, Dict, List

import pytest

from agent_platform.agents import (
    AgentContext,
    EvaluatorAgent,
    PlannerAgent,
    VerifierAgent,
)
from agent_platform.gateway.request_handler import QueryRequest, RequestHandler
from agent_platform.gateway.request_handler.handler import RequestHandler as HandlerCls
from agent_platform.runtime.llm_client import LLMClient, LLMMessage


# ============================================================
# 自定义 MockLLM — 按 Agent 类型返回预设 JSON
# ============================================================


class AgentMockLLM(LLMClient):
    """
    根据调用来源（Planner/Evaluator/Verifier）返回预设 JSON 的 Mock LLM

    通过 system_prompt 中的关键词识别 Agent 类型：
      - "规划Agent" → Planner
      - "评估Agent" → Evaluator
      - "验证Agent" → Verifier
      - 其他（改写/回答）→ 走父类默认 mock
    """

    def __init__(
        self,
        planner_response: Dict[str, Any] = None,
        evaluator_response: Dict[str, Any] = None,
        verifier_response: Dict[str, Any] = None,
    ):
        super().__init__(mock=True)
        self._planner_resp = planner_response or _default_planner_proceed()
        self._evaluator_resp = evaluator_response or _default_evaluator_sufficient()
        self._verifier_resp = verifier_response or _default_verifier_verified()
        # 记录调用次数，便于断言
        self.planner_calls = 0
        self.evaluator_calls = 0
        self.verifier_calls = 0

    def chat_json(
        self,
        messages: List[LLMMessage],
        model: Any = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        system_content = ""
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break

        # 按 system_prompt 关键词识别 Agent 类型
        if "规划Agent" in system_content:
            self.planner_calls += 1
            return self._planner_resp
        elif "评估Agent" in system_content:
            self.evaluator_calls += 1
            return self._evaluator_resp
        elif "验证Agent" in system_content:
            self.verifier_calls += 1
            return self._verifier_resp
        else:
            # 其他调用（如查询改写）走父类默认 mock
            return super().chat_json(messages, model, temperature, max_tokens)


# ============================================================
# 默认 Agent 响应数据
# ============================================================


def _default_planner_proceed() -> Dict[str, Any]:
    """Planner 正常检索响应"""
    return {
        "is_out_of_domain": False,
        "domain_confidence": 0.95,
        "intent": "clause_query",
        "complexity": "L1",
        "sub_queries": [
            {"id": "q1", "text": "商业银行资本管理办法 第43条", "strategy": "hybrid", "filters": {}}
        ],
        "retrieval_plan": {
            "strategies": ["hybrid"],
            "top_k": 10,
            "rerank": True,
            "expand_context": False,
        },
        "reasoning": "条款查询，属于监管领域",
    }


def _default_planner_out_of_domain() -> Dict[str, Any]:
    """Planner 越界响应"""
    return {
        "is_out_of_domain": True,
        "domain_confidence": 0.05,
        "intent": "unknown",
        "complexity": "L0",
        "sub_queries": [],
        "retrieval_plan": {},
        "reasoning": "天气问题不属于监管领域",
    }


def _default_evaluator_sufficient() -> Dict[str, Any]:
    """Evaluator 证据充分响应"""
    return {
        "is_sufficient": True,
        "sufficiency_score": 0.92,
        "dimensions": {
            "coverage": 1.0,
            "authority": 0.95,
            "version_validity": 0.9,
            "conflict": 1.0,
        },
        "missing_claims": [],
        "conflicts": [],
        "retrieval_suggestion": {},
    }


def _default_evaluator_insufficient() -> Dict[str, Any]:
    """Evaluator 证据不足响应（带检索建议）"""
    return {
        "is_sufficient": False,
        "sufficiency_score": 0.42,
        "dimensions": {
            "coverage": 0.5,
            "authority": 0.8,
            "version_validity": 0.7,
            "conflict": 0.9,
        },
        "missing_claims": ["缺少第43条具体内容"],
        "conflicts": [],
        "retrieval_suggestion": {
            "direction": "补充条款内容",
            "suggested_strategy": "bm25",
            "suggested_query": "商业银行资本管理办法 第四十三条 条款内容",
            "reason": "当前证据缺少条款具体内容",
        },
    }


def _default_verifier_verified() -> Dict[str, Any]:
    """Verifier 验证通过响应"""
    return {
        "verified": True,
        "claims": [
            {"text": "核心一级资本充足率不得低于5%", "status": "verified", "evidence_id": "ev-001"}
        ],
        "needs_retry": False,
        "retry_query": "",
        "unverified_count": 0,
    }


def _default_verifier_needs_retry() -> Dict[str, Any]:
    """Verifier 需要重试响应"""
    return {
        "verified": False,
        "claims": [
            {"text": "核心一级资本充足率不得低于5%", "status": "verified", "evidence_id": "ev-001"},
            {"text": "杠杆率不得低于3%", "status": "unverified", "reason": "未找到证据"},
        ],
        "needs_retry": True,
        "retry_query": "杠杆率 最低要求 商业银行资本管理办法",
        "unverified_count": 1,
    }


# ============================================================
# 测试用例
# ============================================================


class TestAgentLoopHandler:
    """Phase 5 Agent 协作 Loop 集成测试"""

    def _make_handler(self, mock_llm: AgentMockLLM) -> RequestHandler:
        """构建使用自定义 MockLLM 的 handler"""
        return HandlerCls(
            planner_agent=PlannerAgent(llm_client=mock_llm),
            evaluator_agent=EvaluatorAgent(llm_client=mock_llm),
            verifier_agent=VerifierAgent(llm_client=mock_llm),
            enable_agent_loop=True,
        )

    # ──────────────────────────────────────────────────────
    # 场景1: 正常 Agent 协作流程
    # ──────────────────────────────────────────────────────

    def test_normal_agent_flow(self):
        """正常 Agent 协作流程：Planner proceed → Retriever → Evaluator sufficient → Verifier verified"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = self._make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Agent 都被调用
        assert mock_llm.planner_calls == 1, "Planner应被调用1次"
        assert mock_llm.evaluator_calls == 1, "Evaluator应被调用1次"
        assert mock_llm.verifier_calls == 1, "Verifier应被调用1次"

        # 状态轨迹包含 PLANNING（Phase 5 新增状态）
        trace = response.state_trace
        assert "PLANNING" in trace, f"状态轨迹应包含PLANNING: {trace}"
        assert "RETRIEVING" in trace
        assert "EVIDENCE_VALIDATING" in trace
        assert "GENERATING" in trace
        assert "ANSWER_VALIDATING" in trace
        assert trace[-1] == "RESPONDING"

        # 应有回答
        assert response.answer
        assert len(response.answer) > 0

    # ──────────────────────────────────────────────────────
    # 场景2: 越界检测 — Planner 判定越界直接拒答
    # ──────────────────────────────────────────────────────

    def test_out_of_domain_refusal(self):
        """Planner 判定越界 → 直接拒答，不进入检索"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_out_of_domain(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = self._make_handler(mock_llm)

        # 用非监管领域的问题触发越界（Planner响应已是越界）
        request = QueryRequest(query="今天北京天气怎么样")
        response = handler.handle_query(request)

        # Planner 被调用，但 Evaluator/Verifier 不应被调用
        assert mock_llm.planner_calls == 1
        assert mock_llm.evaluator_calls == 0, "越界时不应调用Evaluator"
        assert mock_llm.verifier_calls == 0, "越界时不应调用Verifier"

        # 状态轨迹: PLANNING → (状态机路径) → REFUSING → RESPONDING
        # 注意：状态机要求 PLANNING → RETRIEVING → EVIDENCE_ASSEMBLING →
        #       EVIDENCE_VALIDATING → REFUSING 的合法路径，所以 trace 会包含
        #       这些状态，但实际未执行检索（Evaluator/Verifier 未被调用）
        trace = response.state_trace
        assert "PLANNING" in trace
        assert "REFUSING" in trace
        assert "RESPONDING" in trace

    # ──────────────────────────────────────────────────────
    # 场景3: Agent 异常自动回退 V1
    # ──────────────────────────────────────────────────────

    def test_agent_exception_fallback_to_v1(self):
        """Planner 抛异常 → 自动回退 V1 线性流程"""

        class FailingPlanner(PlannerAgent):
            def run(self, context: AgentContext):
                raise RuntimeError("Planner 故障测试")

        mock_llm = AgentMockLLM()
        handler = HandlerCls(
            planner_agent=FailingPlanner(llm_client=mock_llm),
            evaluator_agent=EvaluatorAgent(llm_client=mock_llm),
            verifier_agent=VerifierAgent(llm_client=mock_llm),
            enable_agent_loop=True,
        )

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # 回退 V1 后仍能正常响应
        assert response.answer
        assert response.state_trace[-1] == "RESPONDING"

        # V1 流程不经过 PLANNING 状态
        # 注意：fallback 会重置状态机，trace 可能包含 fallback 标记
        trace = response.state_trace
        assert "RESPONDING" in trace

    # ──────────────────────────────────────────────────────
    # 场景4: enable_agent_loop=False 走 V1 流程
    # ──────────────────────────────────────────────────────

    def test_disable_agent_loop_uses_v1(self):
        """禁用 Agent Loop 时走 V1 线性流程"""
        mock_llm = AgentMockLLM()
        handler = HandlerCls(
            planner_agent=PlannerAgent(llm_client=mock_llm),
            evaluator_agent=EvaluatorAgent(llm_client=mock_llm),
            verifier_agent=VerifierAgent(llm_client=mock_llm),
            enable_agent_loop=False,
        )

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Agent 不应被调用
        assert mock_llm.planner_calls == 0, "禁用Agent时不应调用Planner"
        assert mock_llm.evaluator_calls == 0, "禁用Agent时不应调用Evaluator"
        assert mock_llm.verifier_calls == 0, "禁用Agent时不应调用Verifier"

        # V1 流程不包含 PLANNING 状态
        trace = response.state_trace
        assert "PLANNING" not in trace
        assert "RETRIEVING" in trace
        assert "RESPONDING" in trace

    # ──────────────────────────────────────────────────────
    # 场景5: Loop 多轮检索 — Evaluator 不足 → 重试
    # ──────────────────────────────────────────────────────

    def test_evaluator_insufficient_triggers_retry(self):
        """Evaluator 判定不足 → 触发多轮检索 Loop"""

        # 用计数器控制：第1次insufficient，第2次sufficient
        class LoopMockLLM(AgentMockLLM):
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
                    if self._evaluator_call_count == 1:
                        # 第1轮：不足，触发重试
                        self.evaluator_calls += 1
                        return _default_evaluator_insufficient()
                    else:
                        # 第2轮：充分，退出 Loop
                        self.evaluator_calls += 1
                        return _default_evaluator_sufficient()
                return super().chat_json(messages, model, temperature, max_tokens)

        mock_llm = LoopMockLLM()
        handler = self._make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Evaluator 应被调用至少2次（第1轮不足 → 重试 → 第2轮充分）
        assert mock_llm.evaluator_calls >= 2, (
            f"Evaluator应被调用>=2次，实际{mock_llm.evaluator_calls}"
        )

        # 状态轨迹应包含多次 RETRIEVING（重试）
        trace = response.state_trace
        retry_count = trace.count("RETRIEVING")
        assert retry_count >= 2, f"应有多轮检索（>=2次RETRIEVING），实际{retry_count}"

        # 最终应正常响应
        assert trace[-1] == "RESPONDING"
        assert response.answer

    # ──────────────────────────────────────────────────────
    # 场景6: Verifier needs_retry 触发补充检索
    # ──────────────────────────────────────────────────────

    def test_verifier_needs_retry_triggers_supplementary_retrieval(self):
        """Verifier needs_retry=True → 触发补充检索 + 重新生成"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_needs_retry(),
        )
        handler = self._make_handler(mock_llm)

        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = handler.handle_query(request)

        # Verifier 应被调用（至少1次）
        assert mock_llm.verifier_calls >= 1

        # 状态轨迹应包含 RETRYING（Verifier 触发的重试）
        trace = response.state_trace
        assert "RETRYING" in trace, f"应包含RETRYING状态: {trace}"

        # 最终正常响应
        assert trace[-1] == "RESPONDING"
        assert response.answer

    # ──────────────────────────────────────────────────────
    # 场景7: L0 问候不走 Agent 流程
    # ──────────────────────────────────────────────────────

    def test_l0_greeting_skips_agent_flow(self):
        """L0 问候直接回复，不走 Agent 流程"""
        mock_llm = AgentMockLLM()
        handler = self._make_handler(mock_llm)

        request = QueryRequest(query="你好")
        response = handler.handle_query(request)

        # L0 不经过任何 Agent
        assert mock_llm.planner_calls == 0
        assert mock_llm.evaluator_calls == 0
        assert mock_llm.verifier_calls == 0

        assert response.intent == "greeting"
        assert response.complexity == "L0"
        assert "RETRIEVING" not in response.state_trace
        assert response.is_refusal is False

    # ──────────────────────────────────────────────────────
    # 场景8: 空问题处理
    # ──────────────────────────────────────────────────────

    def test_empty_query_handled(self):
        """空问题不触发 Agent"""
        mock_llm = AgentMockLLM()
        handler = self._make_handler(mock_llm)

        request = QueryRequest(query="")
        response = handler.handle_query(request)

        assert response.is_refusal is True
        assert mock_llm.planner_calls == 0
