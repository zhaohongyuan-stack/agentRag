"""
Phase 2 端到端集成测试

验证完整的 Phase 2 工作流:
  用户问题 → 查询理解 → 查询改写 → 综合路由(P0-P4) → Mock检索
  → 证据组装(去重+父聚合) → LLM回答生成(Mock模式) → 带引用回答

测试场景:
  1. 条款查询端到端（L1 → P1 → 精确检索 → 模板/LLM回答）
  2. 阈值查询端到端（L2 → P2/P3 → 多路检索 → 带引用回答）
  3. 定义查询端到端（L2 → P2 → 检索 → 定义回答）
  4. 多轮对话指代消解（第1轮提到指标 → 第2轮"这个比例"消解）
  5. 同义词扩展（"CAR最低多少" → 扩展为"资本充足率"）
  6. 风险评级影响路由（合规判断 → 高风险 → P4）
  7. 证据去重（相同内容 → 去重后单条证据）
  8. 空结果拒答
  9. 状态轨迹完整性（Phase 2 新增 path_id、risk_level）
"""

import pytest

from agent_platform.gateway.request_handler import (
    QueryRequest,
    RequestHandler,
    RetrievalClient,
)
from agent_platform.gateway.session_handler import SessionManager


class TestPhase2E2E:
    """Phase 2 端到端集成测试"""

    def setup_method(self):
        """每个测试创建新的 handler 实例（Phase 2 默认配置）"""
        self.handler = RequestHandler(
            retrieval_client=RetrievalClient(in_process=True),
            session_manager=SessionManager(),
        )

    # ============================================================
    # 基础查询场景
    # ============================================================

    def test_clause_query_e2e(self):
        """条款查询端到端 — L1 路由 → 精确检索 → 带引用回答"""
        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = self.handler.handle_query(request)

        # 意图和复杂度
        assert response.intent == "clause_query"
        assert response.complexity == "L1"

        # 状态轨迹完整
        trace = response.state_trace
        assert "RECEIVED" in trace
        assert "ANALYZED" in trace
        assert "ROUTED" in trace
        assert "RETRIEVING" in trace
        assert "EVIDENCE_ASSEMBLING" in trace
        assert "RESPONDING" in trace
        assert trace[-1] == "RESPONDING"

        # 有检索结果和证据
        assert response.evidence_count > 0
        assert response.sufficiency_score > 0

        # 回答不为空
        assert response.answer
        assert len(response.answer) > 10

    def test_threshold_query_e2e(self):
        """阈值查询端到端 — L2 路由 → 多路检索 → 带比例的回答"""
        request = QueryRequest(query="核心一级资本充足率最低要求是多少")
        response = self.handler.handle_query(request)

        assert response.intent == "threshold"
        assert response.complexity == "L2"

        # 应有证据
        assert response.evidence_count > 0

        # 回答不为空
        assert response.answer

    def test_definition_query_e2e(self):
        """定义查询端到端 — L2 路由 → 检索 → 定义回答"""
        request = QueryRequest(query="什么是系统重要性银行")
        response = self.handler.handle_query(request)

        assert response.intent == "definition"
        assert response.complexity == "L2"

        # 回答不为空
        assert response.answer

    # ============================================================
    # Phase 2 新特性: 查询改写
    # ============================================================

    def test_synonym_expansion_e2e(self):
        """同义词扩展 — "CAR最低多少" 应能路由到 threshold 并检索"""
        request = QueryRequest(query="CAR最低要求是多少")
        response = self.handler.handle_query(request)

        # 应识别为 threshold 意图
        assert response.intent == "threshold"

    def test_multi_turn_coreference_e2e(self):
        """多轮对话指代消解"""
        # 第一轮: 提到核心一级资本充足率
        req1 = QueryRequest(query="核心一级资本充足率最低要求是多少")
        resp1 = self.handler.handle_query(req1)
        assert resp1.session_id

        # 第二轮: 使用指代 "这个比例"
        req2 = QueryRequest(
            query="这个比例适用所有银行吗",
            session_id=resp1.session_id,
        )
        resp2 = self.handler.handle_query(req2)

        # 应成功处理（不报错）
        assert resp2.answer
        assert resp2.session_id == resp1.session_id

    # ============================================================
    # Phase 2 新特性: 综合路由
    # ============================================================

    def test_compliance_query_high_risk_e2e(self):
        """合规判断 → 高风险路由"""
        request = QueryRequest(query="银行是否符合资本充足率要求")
        response = self.handler.handle_query(request)

        # 合规判断意图
        assert response.intent == "compliance"
        # 复杂度应为 L4
        assert response.complexity == "L4"

    def test_greeting_p0_e2e(self):
        """问候 → P0 无检索直接回复"""
        request = QueryRequest(query="你好")
        response = self.handler.handle_query(request)

        assert response.intent == "greeting"
        assert response.complexity == "L0"
        # 问候不应有检索证据
        assert response.evidence_count == 0

    # ============================================================
    # Phase 2 新特性: 证据组装增强
    # ============================================================

    def test_evidence_sufficiency_e2e(self):
        """证据充分性评分"""
        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = self.handler.handle_query(request)

        # 充分性评分应在合理范围
        assert 0.0 <= response.sufficiency_score <= 1.0

    # ============================================================
    # Phase 2 新特性: 拒答
    # ============================================================

    def test_empty_result_refusal_e2e(self):
        """空结果正确拒答"""
        request = QueryRequest(query="不存在的规定条款")
        response = self.handler.handle_query(request)

        # 应触发拒答或返回
        assert response.answer
        # 如果证据不足，应标记拒答
        if response.sufficiency_score < 0.85:
            assert response.is_refusal

    # ============================================================
    # Phase 2 新特性: 状态轨迹增强
    # ============================================================

    def test_state_trace_completeness_e2e(self):
        """状态轨迹包含完整迁移链"""
        request = QueryRequest(query="什么是杠杆率")
        response = self.handler.handle_query(request)

        trace = response.state_trace
        # 起始状态
        assert trace[0] == "RECEIVED"
        # 终止状态
        assert trace[-1] == "RESPONDING"
        # 关键中间状态
        assert "NORMALIZED" in trace
        assert "ANALYZED" in trace
        assert "ROUTED" in trace

    def test_latency_e2e(self):
        """延迟计算"""
        request = QueryRequest(query="什么是流动性覆盖率")
        response = self.handler.handle_query(request)

        assert response.latency_ms > 0
        assert response.latency_ms < 10000  # 10秒内完成

    def test_request_id_e2e(self):
        """请求ID生成"""
        request = QueryRequest(query="第43条")
        response = self.handler.handle_query(request)

        assert response.request_id
        assert len(response.request_id) > 0

    # ============================================================
    # Phase 2 新特性: 幂等性
    # ============================================================

    def test_idempotency_e2e(self):
        """幂等性 — 相同 key 返回缓存结果"""
        key = "test-idempotency-key-001"
        req1 = QueryRequest(query="什么是拨备覆盖率", idempotency_key=key)
        resp1 = self.handler.handle_query(req1)

        req2 = QueryRequest(query="什么是拨备覆盖率", idempotency_key=key)
        resp2 = self.handler.handle_query(req2)

        # 幂等命中，应返回相同结果
        assert resp1.answer == resp2.answer

    # ============================================================
    # Phase 2 新特性: 引用格式
    # ============================================================

    def test_citations_format_e2e(self):
        """引用格式 — 回答应包含引用信息"""
        request = QueryRequest(query="《商业银行资本管理办法》第43条")
        response = self.handler.handle_query(request)

        # 如果有证据，应有引用
        if response.evidence_count > 0 and not response.is_refusal:
            assert response.citations is not None

    # ============================================================
    # Phase 2 新特性: 多轮对话
    # ============================================================

    def test_multi_turn_session_e2e(self):
        """多轮对话会话保持"""
        # 第一轮
        req1 = QueryRequest(query="你好")
        resp1 = self.handler.handle_query(req1)

        # 第二轮
        req2 = QueryRequest(query="什么是核心一级资本", session_id=resp1.session_id)
        resp2 = self.handler.handle_query(req2)

        # 第三轮
        req3 = QueryRequest(query="它的最低要求是多少", session_id=resp1.session_id)
        resp3 = self.handler.handle_query(req3)

        # 会话保持
        assert resp1.session_id == resp2.session_id
        assert resp2.session_id == resp3.session_id

        # 第三轮应成功处理（指代消解）
        assert resp3.answer
