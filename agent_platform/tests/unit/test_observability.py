"""
可观测性模块单元测试 — M5.5 指标/追踪/日志

测试用例覆盖:
  - 指标采集（Counter / Histogram / Gauge）
  - 追踪生成（完整 Span 树）
  - 日志格式（JSON 格式，含 request_id）
  - 日志可重放（从日志重建决策过程）
"""

import json
import logging
import time

import pytest

from agent_platform.runtime.observability import (
    Counter,
    Gauge,
    Histogram,
    LogReplayer,
    MetricsCollector,
    Span,
    SpanContext,
    StructuredLogFormatter,
    Tracer,
    generate_request_id,
    get_collector,
    set_request_context,
)


# ============================================================
# 指标采集测试
# ============================================================


class TestMetrics:
    """指标采集测试"""

    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_counter_increment(self, collector):
        """Counter 递增"""
        c = collector.counter("test_counter")
        c.inc()
        c.inc()
        c.inc(5)

        assert c.get() == 7.0

    def test_counter_with_labels(self, collector):
        """Counter 带标签"""
        c = collector.counter("queries_total")
        c.inc(path="P0")
        c.inc(path="P0")
        c.inc(path="P1")

        assert c.get(path="P0") == 2.0
        assert c.get(path="P1") == 1.0

    def test_histogram_observe(self, collector):
        """Histogram 观测"""
        h = collector.histogram("latency")
        h.observe(0.1)
        h.observe(0.5)
        h.observe(1.0)

        assert h.count == 3
        assert h.sum == 1.6
        assert abs(h.avg - 0.5333) < 0.01

    def test_histogram_buckets(self, collector):
        """Histogram 桶分布"""
        h = collector.histogram("latency", buckets=[0.1, 0.5, 1.0])
        h.observe(0.05)
        h.observe(0.3)
        h.observe(0.8)

        buckets = h.get_buckets()
        assert buckets[0.1] == 1  # 0.05 <= 0.1
        assert buckets[0.5] == 2  # 0.05, 0.3 <= 0.5
        assert buckets[1.0] == 3  # 全部 <= 1.0

    def test_histogram_percentile(self, collector):
        """Histogram 百分位"""
        h = collector.histogram("latency", buckets=[0.1, 0.5, 1.0, 5.0])
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h.observe(v)

        p50 = h.percentile(0.5)
        assert p50 > 0
        assert p50 <= 5.0

    def test_gauge_set(self, collector):
        """Gauge 设置"""
        g = collector.gauge("cache_hit_rate")
        g.set(0.85)

        assert g.value == 0.85

    def test_gauge_inc_dec(self, collector):
        """Gauge 增减"""
        g = collector.gauge("active_connections")
        g.set(10)
        g.inc()
        g.inc(5)
        g.dec(3)

        assert g.value == 13.0

    def test_default_metrics_exist(self, collector):
        """默认指标已初始化"""
        export = collector.export()

        assert "agent_queries_total" in export
        assert "agent_query_latency_seconds" in export
        assert "budget_consumed_ratio" in export
        assert "evidence_sufficiency_score" in export

    def test_export_format(self, collector):
        """导出格式正确"""
        collector.counter("test_c").inc()
        collector.histogram("test_h").observe(0.5)
        collector.gauge("test_g").set(1.0)

        export = collector.export()

        assert export["test_c"]["type"] == "counter"
        assert export["test_h"]["type"] == "histogram"
        assert export["test_g"]["type"] == "gauge"

    def test_reset(self, collector):
        """重置后指标归零"""
        collector.counter("test_c").inc(10)
        collector.reset()

        c = collector.counter("test_c")
        assert c.get() == 0.0

    def test_query_flow_metrics(self, collector):
        """模拟一次查询流程的指标采集"""
        # 查询计数
        collector.counter("agent_queries_total").inc()
        collector.counter("agent_queries_by_path_total").inc(path="P1")

        # 延迟记录
        collector.histogram("agent_query_latency_seconds").observe(1.5)
        collector.histogram("retrieval_latency_seconds").observe(0.3)
        collector.histogram("generation_latency_seconds").observe(0.8)

        # 质量指标
        collector.histogram("evidence_sufficiency_score").observe(0.92)
        collector.gauge("budget_consumed_ratio").set(0.65)

        export = collector.export()

        assert export["agent_queries_total"]["values"][""] == 1.0
        assert export["agent_query_latency_seconds"]["count"] == 1
        assert export["evidence_sufficiency_score"]["avg"] == 0.92
        assert export["budget_consumed_ratio"]["value"] == 0.65


# ============================================================
# 链路追踪测试
# ============================================================


class TestTracing:
    """链路追踪测试"""

    @pytest.fixture
    def tracer(self):
        return Tracer()

    def test_single_span(self, tracer):
        """单个 Span"""
        span = tracer.start_span("TestSpan")
        tracer.end_span(span)

        root = tracer.finish()
        assert root is not None
        assert root.name == "TestSpan"
        assert root.end_time is not None
        assert root.duration_ms >= 0

    def test_nested_spans(self, tracer):
        """嵌套 Span 树"""
        root = tracer.start_span("QueryRequest")
        with SpanContext(tracer, tracer.start_span("Retrieval", parent=root)) as retrieval:
            retrieval.set_attribute("hits", 5)
        with SpanContext(tracer, tracer.start_span("Verification", parent=root)) as verify:
            verify.set_attribute("passed", True)

        tracer.end_span(root)
        trace = tracer.finish()

        assert trace is not None
        assert trace.name == "QueryRequest"
        assert len(trace.children) == 2
        assert trace.children[0].name == "Retrieval"
        assert trace.children[1].name == "Verification"

    def test_span_attributes(self, tracer):
        """Span 属性"""
        span = tracer.start_span("TestSpan", query="测试查询", user="user-1")
        span.set_attribute("result", "success")
        tracer.end_span(span)

        root = tracer.finish()
        assert root.attributes["query"] == "测试查询"
        assert root.attributes["user"] == "user-1"
        assert root.attributes["result"] == "success"

    def test_span_events(self, tracer):
        """Span 事件"""
        span = tracer.start_span("TestSpan")
        span.add_event("started", component="retrieval")
        span.add_event("completed", duration=0.5)
        tracer.end_span(span)

        root = tracer.finish()
        assert len(root.events) == 2
        assert root.events[0]["name"] == "started"
        assert root.events[1]["attributes"]["duration"] == 0.5

    def test_span_error(self, tracer):
        """Span 错误状态"""
        span = tracer.start_span("FailingSpan")
        span.set_error("执行超时")
        tracer.end_span(span)

        root = tracer.finish()
        assert root.status == "error"
        assert any(e["name"] == "error" for e in root.events)

    def test_span_context_manager(self, tracer):
        """SpanContext 上下文管理器"""
        with SpanContext(tracer, tracer.start_span("ManagedSpan")) as span:
            span.set_attribute("inside", True)

        root = tracer.finish()
        assert root.end_time is not None
        assert root.attributes["inside"] is True

    def test_complete_query_trace(self, tracer):
        """完整查询追踪树"""
        with SpanContext(tracer, tracer.start_span("QueryRequest", query="核心一级资本充足率")) as root:
            with SpanContext(tracer, tracer.start_span("QueryUnderstanding", parent=root)) as qu:
                qu.set_attribute("intent", "threshold_query")
                with SpanContext(tracer, tracer.start_span("IntentClassification", parent=qu)):
                    pass

            with SpanContext(tracer, tracer.start_span("Retrieval", parent=root)) as ret:
                ret.set_attribute("hits", 5)

            with SpanContext(tracer, tracer.start_span("Verification", parent=root)) as verify:
                verify.set_attribute("numeric_valid", True)

        trace = tracer.finish()

        assert trace.name == "QueryRequest"
        assert len(trace.children) == 3
        # QueryUnderstanding 有子 Span
        qu_child = trace.children[0]
        assert qu_child.name == "QueryUnderstanding"
        assert len(qu_child.children) == 1
        assert qu_child.children[0].name == "IntentClassification"

    def test_to_dict(self, tracer):
        """序列化为字典"""
        span = tracer.start_span("TestSpan")
        span.set_attribute("key", "value")
        tracer.end_span(span)

        root = tracer.finish()
        d = root.to_dict()

        assert d["name"] == "TestSpan"
        assert d["attributes"]["key"] == "value"
        assert "duration_ms" in d


# ============================================================
# 结构化日志测试
# ============================================================


class TestStructuredLogging:
    """结构化 JSON 日志测试"""

    def test_json_format(self):
        """日志输出为 JSON 格式"""
        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="测试消息",
            args=None,
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "测试消息"
        assert parsed["module"] == "test.module"
        assert "timestamp" in parsed

    def test_log_with_request_id(self):
        """日志包含 request_id"""
        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="处理请求",
            args=None,
            exc_info=None,
        )
        record.request_id = "req-123"
        record.span_id = "span-456"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["request_id"] == "req-123"
        assert parsed["span_id"] == "span-456"

    def test_log_with_extra(self):
        """日志包含额外字段"""
        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="查询完成",
            args=None,
            exc_info=None,
        )
        record.latency_ms = 150.5
        record.path = "P1"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["extra"]["latency_ms"] == 150.5
        assert parsed["extra"]["path"] == "P1"

    def test_generate_request_id(self):
        """生成 request_id"""
        rid = generate_request_id()
        assert rid.startswith("req-")
        assert len(rid) > 4

    def test_request_context_adapter(self):
        """请求上下文适配器"""
        logger = logging.getLogger("test.context")
        adapter = set_request_context(
            logger, request_id="req-123", session_id="sess-456"
        )

        formatter = StructuredLogFormatter()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # 使用 capture
        import io
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        adapter.info("处理请求")

        output = stream.getvalue().strip()
        parsed = json.loads(output)

        assert parsed["request_id"] == "req-123"
        assert parsed["session_id"] == "sess-456"

        logger.removeHandler(handler)


# ============================================================
# 日志重放测试
# ============================================================


class TestLogReplay:
    """日志重放测试"""

    def test_load_logs(self):
        """加载日志"""
        replayer = LogReplayer()
        logs = [
            json.dumps({"timestamp": "2024-01-01T12:00:00", "level": "INFO", "message": "开始"}),
            json.dumps({"timestamp": "2024-01-01T12:00:01", "level": "INFO", "message": "结束"}),
        ]

        count = replayer.load(logs)
        assert count == 2
        assert replayer.entry_count == 2

    def test_filter_by_request(self):
        """按 request_id 过滤"""
        replayer = LogReplayer()
        replayer.load([
            json.dumps({"request_id": "req-1", "message": "A"}),
            json.dumps({"request_id": "req-2", "message": "B"}),
            json.dumps({"request_id": "req-1", "message": "C"}),
        ])

        filtered = replayer.filter_by_request("req-1")
        assert len(filtered) == 2
        assert filtered[0]["message"] == "A"
        assert filtered[1]["message"] == "C"

    def test_rebuild_trace(self):
        """重建追踪树"""
        replayer = LogReplayer()
        replayer.load([
            json.dumps({
                "timestamp": "2024-01-01T12:00:00",
                "level": "INFO",
                "module": "agent_platform.gateway",
                "message": "开始查询",
                "span_id": "span-1",
                "extra": {"span_name": "QueryRequest", "parent_id": None},
            }),
            json.dumps({
                "timestamp": "2024-01-01T12:00:01",
                "level": "INFO",
                "module": "agent_platform.retrieval",
                "message": "检索完成",
                "span_id": "span-2",
                "extra": {"span_name": "Retrieval", "parent_id": "span-1"},
            }),
            json.dumps({
                "timestamp": "2024-01-01T12:00:02",
                "level": "INFO",
                "module": "agent_platform.verification",
                "message": "验证通过",
                "span_id": "span-3",
                "extra": {"span_name": "Verification", "parent_id": "span-1"},
            }),
        ])

        trace = replayer.rebuild_trace()

        assert trace["span_id"] == "span-1"
        assert trace["name"] == "QueryRequest"
        assert len(trace["children"]) == 2
        assert trace["children"][0]["name"] == "Retrieval"
        assert trace["children"][1]["name"] == "Verification"

    def test_rebuild_trace_with_error(self):
        """重建含错误的追踪"""
        replayer = LogReplayer()
        replayer.load([
            json.dumps({
                "timestamp": "2024-01-01T12:00:00",
                "level": "INFO",
                "message": "开始",
                "span_id": "span-1",
                "extra": {"span_name": "QueryRequest", "parent_id": None},
            }),
            json.dumps({
                "timestamp": "2024-01-01T12:00:01",
                "level": "ERROR",
                "message": "检索失败",
                "span_id": "span-1",
                "extra": {"span_name": "QueryRequest", "parent_id": None},
            }),
        ])

        trace = replayer.rebuild_trace()
        assert trace["status"] == "error"
        assert len(trace["events"]) == 2

    def test_empty_logs(self):
        """空日志"""
        replayer = LogReplayer()
        assert replayer.entry_count == 0
        assert replayer.rebuild_trace() == {}

    def test_invalid_json_skipped(self):
        """无效 JSON 被跳过"""
        replayer = LogReplayer()
        replayer.load([
            "invalid json",
            json.dumps({"message": "valid"}),
            "",
        ])

        assert replayer.entry_count == 1
