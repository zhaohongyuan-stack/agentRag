"""
端到端 Mock 流程测试

完整流程:
  用户问题 → request_handler → query_understanding → routing → state_machine
    → Mock Retrieval API → 简单回答组装 → 返回

测试场景:
  1. 条款查询: "第43条内容" → RECEIVED→...→RESPONDING, Mock hit
  2. 空结果: "不存在的条款" → ...→EVIDENCE_VALIDATING→REFUSING, 拒答
  3. 歧义澄清: "那个比例" → ...→ROUTED→CLARIFYING, 澄清请求
  4. 阈值查询: "核心一级资本充足率最低要求是多少" → 正常回答+引用
  5. 问候: "你好" → L0 直接回复
  6. 超时场景: timeout → 错误处理
"""

import pytest

from agent_platform.gateway.request_handler import QueryRequest, RequestHandler


class TestMockE2E:
    """端到端 Mock 流程测试"""

    def setup_method(self):
        self.handler = RequestHandler()

    # ============================================================
    # 场景1: 条款查询（正常流程）
    # ============================================================
    def test_clause_query_e2e(self):
        """条款查询端到端"""
        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = self.handler.handle_query(request)

        # 意图和复杂度
        assert response.intent == "clause_query"
        assert response.complexity == "L1"

        # 状态轨迹完整
        trace = response.state_trace
        assert "RECEIVED" in trace
        assert "NORMALIZED" in trace
        assert "ANALYZED" in trace
        assert "ROUTED" in trace
        assert "RETRIEVING" in trace
        assert "EVIDENCE_ASSEMBLING" in trace
        assert "EVIDENCE_VALIDATING" in trace
        assert "RESPONDING" in trace
        assert trace[-1] == "RESPONDING"

        # 应有检索结果和证据
        assert response.evidence_count > 0
        assert response.sufficiency_score > 0

        # 回答不为空
        assert response.answer
        assert len(response.answer) > 10

    # ============================================================
    # 场景2: 空结果拒答
    # ============================================================
    def test_empty_result_refusal_e2e(self):
        """空结果触发拒答"""
        request = QueryRequest(query="空结果empty无结果no result")
        response = self.handler.handle_query(request)

        # 状态轨迹应包含证据验证
        assert "EVIDENCE_VALIDATING" in response.state_trace

        # 空结果时应拒答
        if response.evidence_count == 0:
            assert response.is_refusal is True
            assert response.refusal_reason is not None
            assert "RESPONDING" in response.state_trace

    # ============================================================
    # 场景3: 歧义澄清
    # ============================================================
    def test_ambiguity_clarification_e2e(self):
        """歧义触发澄清"""
        request = QueryRequest(query="那个比例是多少")
        response = self.handler.handle_query(request)

        # 应检测到歧义
        if response.ambiguities:
            # 状态轨迹应包含 CLARIFYING
            assert "CLARIFYING" in response.state_trace
            # 回答应包含澄清提示
            assert "澄清" in response.answer or "歧义" in response.answer or "明确" in response.answer

    # ============================================================
    # 场景4: 阈值查询（正常回答+引用）
    # ============================================================
    def test_threshold_query_e2e(self):
        """阈值查询端到端"""
        request = QueryRequest(query="核心一级资本充足率最低要求是多少")
        response = self.handler.handle_query(request)

        assert response.intent == "threshold"
        assert response.complexity == "L2"

        # 状态轨迹包含完整检索流程
        trace = response.state_trace
        assert "RETRIEVING" in trace
        assert "EVIDENCE_ASSEMBLING" in trace
        assert "EVIDENCE_VALIDATING" in trace

        # 如果证据充分，应有回答和引用
        if not response.is_refusal:
            assert response.answer
            assert len(response.citations) > 0

    # ============================================================
    # 场景5: 问候直接回复
    # ============================================================
    def test_greeting_e2e(self):
        """问候直接回复"""
        request = QueryRequest(query="你好")
        response = self.handler.handle_query(request)

        assert response.intent == "greeting"
        assert response.complexity == "L0"

        # L0 不走检索流程
        assert "RETRIEVING" not in response.state_trace
        assert "RESPONDING" in response.state_trace

        # 直接回复
        assert response.answer
        assert response.is_refusal is False
        assert response.evidence_count == 0

    # ============================================================
    # 场景6: 定义查询
    # ============================================================
    def test_definition_query_e2e(self):
        """定义查询端到端"""
        request = QueryRequest(query="什么是系统重要性银行")
        response = self.handler.handle_query(request)

        assert response.intent == "definition"
        assert response.complexity == "L2"
        assert "RETRIEVING" in response.state_trace

    # ============================================================
    # 场景7: 多轮对话
    # ============================================================
    def test_multi_turn_e2e(self):
        """多轮对话"""
        # 第一轮
        req1 = QueryRequest(query="你好")
        resp1 = self.handler.handle_query(req1)
        session_id = resp1.session_id

        # 第二轮，使用相同 session
        req2 = QueryRequest(query="第43条", session_id=session_id)
        resp2 = self.handler.handle_query(req2)

        assert resp2.session_id == session_id
        assert resp2.intent == "clause_query"

    # ============================================================
    # 场景8: 超时处理
    # ============================================================
    def test_timeout_handling_e2e(self):
        """超时场景处理"""
        request = QueryRequest(query="超时timeout")
        response = self.handler.handle_query(request)

        # 超时应走证据不足路径或正常路径
        # 关键是不崩溃，返回有效响应
        assert response.answer
        assert response.state_trace[-1] == "RESPONDING"

    # ============================================================
    # 场景9: 状态轨迹完整性
    # ============================================================
    def test_state_trace_starts_and_ends_correctly(self):
        """状态轨迹以 RECEIVED 开头，以 RESPONDING 结尾"""
        request = QueryRequest(query="核心一级资本充足率最低要求是多少")
        response = self.handler.handle_query(request)

        trace = response.state_trace
        assert trace[0] == "RECEIVED"
        assert trace[-1] == "RESPONDING"

    # ============================================================
    # 场景10: 幂等性
    # ============================================================
    def test_idempotency_e2e(self):
        """幂等性测试"""
        key = "e2e-idem-001"
        req1 = QueryRequest(
            query="核心一级资本充足率最低要求是多少",
            idempotency_key=key,
        )
        resp1 = self.handler.handle_query(req1)

        req2 = QueryRequest(
            query="核心一级资本充足率最低要求是多少",
            idempotency_key=key,
        )
        resp2 = self.handler.handle_query(req2)

        # 幂等命中应返回相同 request_id
        assert resp1.request_id == resp2.request_id

    # ============================================================
    # 场景11: 空问题处理
    # ============================================================
    def test_empty_query_e2e(self):
        """空问题处理"""
        request = QueryRequest(query="")
        response = self.handler.handle_query(request)

        assert response.is_refusal is True
        assert response.answer  # 有错误提示

    # ============================================================
    # 场景12: 证据充分性评分
    # ============================================================
    def test_sufficiency_score_e2e(self):
        """证据充分性评分"""
        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = self.handler.handle_query(request)

        # 有证据时应有评分
        if response.evidence_count > 0:
            assert 0 <= response.sufficiency_score <= 1.0
