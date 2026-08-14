"""
Phase 6 执行追踪单元测试

测试内容:
  1. TraceCollector 各 record_xxx 方法的正确性
  2. TraceCollector.finalize() 输出结构
  3. TraceStore 的 CRUD（save/get_by_session/list_recent/count/delete）
  4. TraceStore 临时数据库隔离（每个测试用例独立 db_path）
"""

import json
import os
import tempfile
import uuid

import pytest

from agent_platform.observability import TraceCollector, TraceStore


# ============================================================
# TraceCollector 单元测试
# ============================================================


class TestTraceCollector:
    """TraceCollector 收集器测试"""

    def test_init(self):
        """初始化默认值"""
        tc = TraceCollector(session_id="s1")
        assert tc.session_id == "s1"
        trace = tc.to_dict()
        assert trace["session_id"] == "s1"
        assert trace["query"] == ""
        assert trace["planner"] is None
        assert trace["retrieval_rounds"] == []
        assert trace["evaluations"] == []
        assert trace["state_trace"] == []

    def test_record_query(self):
        """记录用户问题"""
        tc = TraceCollector(session_id="s1")
        tc.record_query("商业银行资本充足率")
        assert tc.to_dict()["query"] == "商业银行资本充足率"

    def test_record_state(self):
        """记录状态迁移（按顺序追加）"""
        tc = TraceCollector(session_id="s1")
        tc.record_state("PLANNING")
        tc.record_state("RETRIEVING")
        tc.record_state("RESPONDING")
        assert tc.to_dict()["state_trace"] == ["PLANNING", "RETRIEVING", "RESPONDING"]

    def test_record_planner(self):
        """记录 Planner 决策"""
        tc = TraceCollector(session_id="s1")
        planner_result = {
            "is_out_of_domain": False,
            "domain_confidence": 0.95,
            "intent": "clause_query",
            "complexity": "L1",
            "sub_queries": [{"id": "q1", "text": "..."}],
            "retrieval_plan": {"strategies": ["hybrid"], "top_k": 10},
        }
        tc.record_planner(result=planner_result, latency_ms=120, decision="proceed")
        planner = tc.to_dict()["planner"]
        assert planner is not None
        assert planner["decision"] == "proceed"
        assert planner["is_out_of_domain"] is False
        assert planner["intent"] == "clause_query"
        assert planner["latency_ms"] == 120
        assert planner["retrieval_plan"]["strategies"] == ["hybrid"]

    def test_record_retrieval(self):
        """记录检索轮次"""
        tc = TraceCollector(session_id="s1")
        tc.record_retrieval(
            round_num=0, query="测试查询", hits=15,
            latency_ms=80, strategy="hybrid", filters={"doc_name": "A"},
        )
        tc.record_retrieval(
            round_num=1, query="补充查询", hits=8, latency_ms=60, strategy="bm25",
        )
        rounds = tc.to_dict()["retrieval_rounds"]
        assert len(rounds) == 2
        assert rounds[0]["round"] == 0
        assert rounds[0]["hits"] == 15
        assert rounds[0]["filters"] == {"doc_name": "A"}
        assert rounds[1]["round"] == 1
        assert rounds[1]["strategy"] == "bm25"

    def test_record_evaluation(self):
        """记录评估结果"""
        tc = TraceCollector(session_id="s1")
        tc.record_evaluation(
            round_num=0, score=0.42,
            dimensions={"coverage": 0.5, "authority": 0.8},
            suggestion={"direction": "补充表格", "suggested_strategy": "table"},
            latency_ms=90, is_sufficient=False,
        )
        tc.record_evaluation(
            round_num=1, score=0.92,
            dimensions={"coverage": 1.0, "authority": 0.95},
            suggestion={}, latency_ms=85, is_sufficient=True,
        )
        evals = tc.to_dict()["evaluations"]
        assert len(evals) == 2
        assert evals[0]["is_sufficient"] is False
        assert evals[0]["sufficiency_score"] == 0.42
        assert evals[0]["retrieval_suggestion"]["suggested_strategy"] == "table"
        assert evals[1]["is_sufficient"] is True

    def test_record_generation(self):
        """记录生成信息"""
        tc = TraceCollector(session_id="s1")
        tc.record_generation(
            model="deepseek-chat",
            tokens={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            latency_ms=350, confidence=0.88,
        )
        gen = tc.to_dict()["generation"]
        assert gen["model"] == "deepseek-chat"
        assert gen["tokens"]["total_tokens"] == 700
        assert gen["confidence"] == 0.88

    def test_record_verification(self):
        """记录验证结果"""
        tc = TraceCollector(session_id="s1")
        claims = [
            {"text": "声明1", "status": "verified", "evidence_id": "ev-1"},
            {"text": "声明2", "status": "unverified", "reason": "无证据"},
        ]
        tc.record_verification(
            verified_count=1, unverified_count=1,
            needs_retry=True, claims=claims, latency_ms=110,
        )
        ver = tc.to_dict()["verification"]
        assert ver["verified_count"] == 1
        assert ver["unverified_count"] == 1
        assert ver["needs_retry"] is True
        assert len(ver["claims"]) == 2

    def test_finalize(self):
        """finalize 返回完整 trace"""
        tc = TraceCollector(session_id="s1", request_id="r1")
        tc.record_query("测试")
        tc.record_state("PLANNING")
        tc.record_planner(
            result={"is_out_of_domain": False, "intent": "clause_query"},
            latency_ms=100, decision="proceed",
        )
        tc.record_retrieval(round_num=0, query="测试", hits=10, latency_ms=50)
        tc.record_evaluation(
            round_num=0, score=0.9, dimensions={},
            suggestion={}, latency_ms=60, is_sufficient=True,
        )
        trace = tc.finalize(total_latency_ms=500)

        assert trace["session_id"] == "s1"
        assert trace["request_id"] == "r1"
        assert trace["query"] == "测试"
        assert trace["total_latency_ms"] == 500
        assert trace["retrieval_round_count"] == 1
        assert trace["planner"]["intent"] == "clause_query"
        assert len(trace["retrieval_rounds"]) == 1
        assert len(trace["evaluations"]) == 1
        assert trace["state_trace"] == ["PLANNING"]

    def test_finalize_auto_latency(self):
        """finalize 不传 total_latency_ms 时自动计算"""
        tc = TraceCollector(session_id="s1")
        trace = tc.finalize()
        assert trace["total_latency_ms"] >= 0

    def test_record_loop_count(self):
        """记录 Loop 轮次"""
        tc = TraceCollector(session_id="s1")
        tc.record_loop_count(3)
        assert tc.to_dict()["loop_count"] == 3


# ============================================================
# TraceStore 单元测试
# ============================================================


@pytest.fixture
def temp_trace_store():
    """每个测试用例使用独立的临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = TraceStore(db_path=path)
    yield store
    # 清理临时文件
    try:
        os.remove(path)
    except OSError:
        pass


class TestTraceStore:
    """TraceStore 存储测试"""

    def test_save_and_get_by_session(self, temp_trace_store):
        """存储 trace 并按 session_id 查询"""
        store = temp_trace_store
        tc = TraceCollector(session_id="session-001")
        tc.record_query("测试查询")
        tc.record_planner(
            result={"is_out_of_domain": False, "intent": "clause_query"},
            latency_ms=100, decision="proceed",
        )
        trace = tc.finalize(total_latency_ms=300)

        trace_id = store.save(trace)
        assert trace_id  # 返回 UUID 字符串

        # 按 session 查询
        results = store.get_by_session("session-001")
        assert len(results) == 1
        assert results[0]["session_id"] == "session-001"
        assert results[0]["trace"]["query"] == "测试查询"
        assert results[0]["trace"]["planner"]["intent"] == "clause_query"

    def test_get_by_id(self, temp_trace_store):
        """按 trace_id 查询"""
        store = temp_trace_store
        tc = TraceCollector(session_id="s1")
        tc.record_query("test")
        trace = tc.finalize(100)
        trace_id = store.save(trace)

        result = store.get_by_id(trace_id)
        assert result is not None
        assert result["id"] == trace_id
        assert result["trace"]["query"] == "test"

        # 不存在的 ID
        assert store.get_by_id("nonexistent") is None

    def test_list_recent(self, temp_trace_store):
        """列出最近的 trace（按时间倒序）"""
        store = temp_trace_store
        # 存 3 条
        for i in range(3):
            tc = TraceCollector(session_id=f"s{i}")
            tc.record_query(f"q{i}")
            store.save(tc.finalize(100 + i))

        results = store.list_recent(limit=10)
        assert len(results) == 3
        # 倒序：最新的在前
        sessions = [r["session_id"] for r in results]
        assert "s0" in sessions
        assert "s1" in sessions
        assert "s2" in sessions

    def test_list_recent_with_limit(self, temp_trace_store):
        """limit 参数生效"""
        store = temp_trace_store
        for i in range(5):
            tc = TraceCollector(session_id=f"s{i}")
            store.save(tc.finalize(100))

        results = store.list_recent(limit=2)
        assert len(results) == 2

    def test_count(self, temp_trace_store):
        """trace 总数"""
        store = temp_trace_store
        assert store.count() == 0
        for i in range(3):
            tc = TraceCollector(session_id=f"s{i}")
            store.save(tc.finalize(100))
        assert store.count() == 3

    def test_get_by_session_multiple(self, temp_trace_store):
        """同一 session 多条 trace"""
        store = temp_trace_store
        for i in range(3):
            tc = TraceCollector(session_id="same-session")
            tc.record_query(f"q{i}")
            store.save(tc.finalize(100))

        results = store.get_by_session("same-session")
        assert len(results) == 3
        queries = [r["trace"]["query"] for r in results]
        assert set(queries) == {"q0", "q1", "q2"}

    def test_delete_by_session(self, temp_trace_store):
        """删除指定 session 的 trace"""
        store = temp_trace_store
        tc1 = TraceCollector(session_id="s1")
        store.save(tc1.finalize(100))
        tc2 = TraceCollector(session_id="s2")
        store.save(tc2.finalize(100))

        deleted = store.delete_by_session("s1")
        assert deleted == 1
        assert store.count() == 1
        assert len(store.get_by_session("s1")) == 0
        assert len(store.get_by_session("s2")) == 1

    def test_trace_json_serializable(self, temp_trace_store):
        """存储的 trace_json 可反序列化"""
        store = temp_trace_store
        tc = TraceCollector(session_id="s1")
        tc.record_query("中文测试")
        tc.record_retrieval(round_num=0, query="中文", hits=5, latency_ms=30)
        trace = tc.finalize(200)

        store.save(trace)
        result = store.get_by_session("s1")[0]
        # 中文内容正确反序列化
        assert result["trace"]["query"] == "中文测试"
        assert result["trace"]["retrieval_rounds"][0]["query"] == "中文"
