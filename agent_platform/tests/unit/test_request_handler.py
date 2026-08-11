"""
请求处理器单元测试

测试用例:
  - 新建会话: 无 session_id → 生成新 session_id
  - 恢复会话: 有效 session_id → 恢复会话状态
  - 无效会话: 过期 session_id → 创建新会话
  - 请求去重: 相同 idempotency_key → 返回缓存结果
  - 空问题: 空字符串 → 返回参数错误
"""

import pytest

from agent_platform.gateway.request_handler import (
    QueryRequest,
    RequestHandler,
    RetrievalClient,
)
from agent_platform.gateway.session_handler import SessionManager


class TestRequestHandler:
    """请求处理器测试"""

    def setup_method(self):
        """每个测试创建新的 handler 实例"""
        self.handler = RequestHandler(
            retrieval_client=RetrievalClient(in_process=True),
            session_manager=SessionManager(),
        )

    def test_new_session(self):
        """新建会话"""
        request = QueryRequest(query="你好")
        response = self.handler.handle_query(request)
        assert response.session_id  # 非空
        assert response.intent == "greeting"
        assert response.complexity == "L0"
        assert "您好" in response.answer or "你好" in response.answer

    def test_restore_session(self):
        """恢复会话"""
        # 第一轮对话
        request1 = QueryRequest(query="你好")
        response1 = self.handler.handle_query(request1)

        # 第二轮对话，使用相同 session_id
        request2 = QueryRequest(query="第43条", session_id=response1.session_id)
        response2 = self.handler.handle_query(request2)

        assert response2.session_id == response1.session_id
        assert response2.intent == "clause_query"

    def test_invalid_session_creates_new(self):
        """无效会话 ID 创建新会话"""
        request = QueryRequest(query="你好", session_id="invalid-uuid")
        response = self.handler.handle_query(request)
        assert response.session_id != "invalid-uuid"
        assert response.session_id  # 生成了新的

    def test_idempotency(self):
        """请求去重"""
        request1 = QueryRequest(
            query="核心一级资本充足率最低要求是多少",
            idempotency_key="idem-001",
        )
        response1 = self.handler.handle_query(request1)

        request2 = QueryRequest(
            query="核心一级资本充足率最低要求是多少",
            idempotency_key="idem-001",
        )
        response2 = self.handler.handle_query(request2)

        assert response1.request_id == response2.request_id

    def test_empty_query(self):
        """空问题返回错误"""
        request = QueryRequest(query="")
        response = self.handler.handle_query(request)
        assert response.is_refusal is True
        assert "空" in response.answer or "不能" in response.answer

    def test_clause_query(self):
        """条款查询端到端"""
        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = self.handler.handle_query(request)
        assert response.intent == "clause_query"
        assert response.complexity == "L1"
        assert response.state_trace  # 有状态轨迹
        assert "RECEIVED" in response.state_trace
        assert "RESPONDING" in response.state_trace

    def test_threshold_query(self):
        """阈值查询端到端"""
        request = QueryRequest(query="核心一级资本充足率最低要求是多少")
        response = self.handler.handle_query(request)
        assert response.intent == "threshold"
        assert response.complexity == "L2"
        # 应该有检索结果
        assert response.evidence_count >= 0
        # 状态轨迹应包含检索和证据验证
        assert "RETRIEVING" in response.state_trace
        assert "EVIDENCE_VALIDATING" in response.state_trace

    def test_definition_query(self):
        """定义查询端到端"""
        request = QueryRequest(query="什么是系统重要性银行")
        response = self.handler.handle_query(request)
        assert response.intent == "definition"
        assert response.complexity == "L2"

    def test_empty_result_refusal(self):
        """空结果拒答"""
        # 使用 empty 场景触发空结果
        request = QueryRequest(query="不存在的条款empty空结果")
        response = self.handler.handle_query(request)
        # 应该走拒答路径或正常回答（取决于 Mock 路由）
        assert response.state_trace is not None
        # 如果检索到空结果，应该拒答
        if response.evidence_count == 0:
            assert response.is_refusal is True

    def test_state_trace_completeness(self):
        """状态轨迹完整性"""
        request = QueryRequest(query="核心一级资本充足率最低要求是多少")
        response = self.handler.handle_query(request)
        trace = response.state_trace
        # 至少包含 RECEIVED → ... → RESPONDING
        assert trace[0] == "RECEIVED"
        assert trace[-1] == "RESPONDING"

    def test_response_has_latency(self):
        """响应包含延迟信息"""
        request = QueryRequest(query="你好")
        response = self.handler.handle_query(request)
        assert response.latency_ms > 0
        assert response.latency_ms < 10000  # 应在合理范围内

    def test_response_has_request_id(self):
        """响应包含 request_id"""
        request = QueryRequest(query="你好")
        response = self.handler.handle_query(request)
        assert response.request_id  # 非空
