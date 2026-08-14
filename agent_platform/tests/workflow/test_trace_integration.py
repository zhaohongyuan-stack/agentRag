"""
Phase 6 执行追踪 Handler 集成测试

验证 handler 在 Agent 协作流程中正确收集 trace 并持久化到 SQLite。

测试场景:
  1. 正常 Agent 流程生成完整 trace（含 Planner/Retriever/Evaluator/Generator/Verifier）
  2. 越界检测 trace（含 Planner 决策，无检索轮次）
  3. trace 可按 session_id 从数据库查询
  4. enable_trace=False 时不存储 trace
"""

import os
import tempfile

import pytest

from agent_platform.agents import EvaluatorAgent, PlannerAgent, VerifierAgent
from agent_platform.gateway.request_handler import QueryRequest
from agent_platform.gateway.request_handler.handler import RequestHandler
from agent_platform.observability import TraceStore
from agent_platform.tests.workflow.test_agent_loop_handler import AgentMockLLM
from agent_platform.tests.workflow.test_agent_loop_handler import (
    _default_evaluator_sufficient,
    _default_planner_out_of_domain,
    _default_planner_proceed,
    _default_verifier_verified,
)


@pytest.fixture
def temp_trace_db(monkeypatch):
    """让 handler 使用临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = TraceStore(db_path=path)
    yield store
    try:
        os.remove(path)
    except OSError:
        pass


class TestTraceIntegration:
    """Phase 6 trace handler 集成测试"""

    def _make_handler(self, mock_llm, trace_store):
        """构建使用临时 trace_store 的 handler"""
        handler = RequestHandler(
            planner_agent=PlannerAgent(llm_client=mock_llm),
            evaluator_agent=EvaluatorAgent(llm_client=mock_llm),
            verifier_agent=VerifierAgent(llm_client=mock_llm),
            enable_agent_loop=True,
            enable_trace=True,
        )
        # 替换为临时 store
        handler._trace_store = trace_store
        return handler

    def test_normal_flow_generates_trace(self, temp_trace_db):
        """正常 Agent 流程生成完整 trace"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = self._make_handler(mock_llm, temp_trace_db)

        session_id = "trace-test-normal"
        request = QueryRequest(query="《商业银行资本管理办法》第43条", session_id=session_id)
        response = handler.handle_query(request)

        # 验证 trace 已存储
        traces = temp_trace_db.get_by_session(session_id)
        assert len(traces) == 1
        trace = traces[0]["trace"]

        # trace 包含完整信息
        assert trace["session_id"] == session_id
        assert trace["query"] == "《商业银行资本管理办法》第43条"
        assert trace["planner"] is not None
        assert trace["planner"]["intent"] == "clause_query"
        assert trace["planner"]["is_out_of_domain"] is False
        assert len(trace["retrieval_rounds"]) >= 1
        assert trace["retrieval_rounds"][0]["hits"] >= 0
        assert len(trace["evaluations"]) >= 1
        assert trace["evaluations"][0]["is_sufficient"] is True
        assert trace["generation"] is not None
        assert trace["verification"] is not None
        assert trace["verification"]["needs_retry"] is False
        assert trace["total_latency_ms"] >= 0
        assert "PLANNING" in trace["state_trace"]
        assert "RESPONDING" in trace["state_trace"]

    def test_out_of_domain_trace(self, temp_trace_db):
        """越界检测的 trace（无检索轮次）"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_out_of_domain(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = self._make_handler(mock_llm, temp_trace_db)

        session_id = "trace-test-out-of-domain"
        request = QueryRequest(query="今天天气", session_id=session_id)
        handler.handle_query(request)

        traces = temp_trace_db.get_by_session(session_id)
        assert len(traces) == 1
        trace = traces[0]["trace"]

        # Planner 记录越界
        assert trace["planner"]["is_out_of_domain"] is True
        # 越界不执行检索
        assert len(trace["retrieval_rounds"]) == 0
        assert len(trace["evaluations"]) == 0
        assert trace["verification"] is None

    def test_trace_queryable_by_session(self, temp_trace_db):
        """trace 可按 session_id 查询，多次问答生成多条"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = self._make_handler(mock_llm, temp_trace_db)

        session_id = "trace-multi-query"
        # 同一 session 两次问答
        handler.handle_query(QueryRequest(query="问题1", session_id=session_id))
        handler.handle_query(QueryRequest(query="问题2", session_id=session_id))

        traces = temp_trace_db.get_by_session(session_id)
        assert len(traces) == 2
        queries = [t["trace"]["query"] for t in traces]
        assert "问题1" in queries
        assert "问题2" in queries

    def test_disable_trace_no_storage(self, temp_trace_db):
        """enable_trace=False 时不存储 trace"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = RequestHandler(
            planner_agent=PlannerAgent(llm_client=mock_llm),
            evaluator_agent=EvaluatorAgent(llm_client=mock_llm),
            verifier_agent=VerifierAgent(llm_client=mock_llm),
            enable_agent_loop=True,
            enable_trace=False,
        )
        handler._trace_store = temp_trace_db

        session_id = "trace-disabled"
        handler.handle_query(QueryRequest(query="测试", session_id=session_id))

        # 不应存储任何 trace
        assert temp_trace_db.count() == 0
        assert len(temp_trace_db.get_by_session(session_id)) == 0

    def test_trace_storage_failure_does_not_break_flow(self, temp_trace_db):
        """trace 存储失败不影响主流程"""
        mock_llm = AgentMockLLM(
            planner_response=_default_planner_proceed(),
            evaluator_response=_default_evaluator_sufficient(),
            verifier_response=_default_verifier_verified(),
        )
        handler = self._make_handler(mock_llm, temp_trace_db)

        # 让 trace_store.save 抛异常
        class FailingStore:
            def save(self, trace):
                raise RuntimeError("存储故障")

        handler._trace_store = FailingStore()

        session_id = "trace-failure"
        request = QueryRequest(query="测试", session_id=session_id)
        response = handler.handle_query(request)

        # 主流程不受影响
        assert response.answer
        assert response.state_trace[-1] == "RESPONDING"
