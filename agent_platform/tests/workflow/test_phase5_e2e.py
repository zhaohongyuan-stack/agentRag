"""
Phase 5 端到端可靠性集成测试

验证 Phase 5 五大模块（声明验证、多轮记忆、工具平台、网关、可观测性）
的协同工作，以及与 Phase 1-4 模块的集成。

测试场景:
  1. 声明验证全流程 — 回答拆分 → 证据匹配 → 数值/版本/引用/范围校验 → accept/retry/refuse
  2. 多轮记忆全流程 — 多轮对话 → 工作记忆更新 → 摘要生成 → 指代消解
  3. 工具调用全流程 — 注册 → 权限 → 执行 → 重试 → 降级 → 事件日志
  4. 网关全流程 — 鉴权 → 限流 → 幂等 → 请求通过
  5. 可观测性全流程 — 指标采集 → 链路追踪 → 日志重放
  6. 跨模块集成 — 网关入口 → 记忆上下文 → 工具调用 → 验证 → 可观测性
"""

import asyncio
import logging
import uuid

import pytest

from agent_platform.evidence.evidence_assembler.builder import (
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.gateway import (
    AuthHandler,
    IdempotencyHandler,
    RateLimiter,
)
from agent_platform.memory import (
    MemoryManager,
    MemoryReferenceResolver,
    Turn,
)
from agent_platform.runtime.observability import (
    LogReplayer,
    MetricsCollector,
    SpanContext,
    StructuredLogFormatter,
    Tracer,
    generate_request_id,
)
from agent_platform.tools import create_default_platform
from agent_platform.verification import AnswerValidator

logger = logging.getLogger(__name__)


# ============================================================
# 测试数据工厂
# ============================================================


def make_evidence(
    content: str = "",
    citation: str = "《商业银行资本管理办法》第23条",
    source_doc: str = "《商业银行资本管理办法》",
    score: float = 0.92,
    version_status: str = "active",
    normative_level: str = "部门规章",
) -> EvidenceItem:
    """构造证据项"""
    return EvidenceItem(
        evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
        chunk_id=f"chunk-{uuid.uuid4().hex[:8]}",
        content=content,
        evidence_snippet=content[:200],
        citation=citation,
        score=score,
        source_doc=source_doc,
        hierarchy_path="第三章/第23条",
        chunk_type="clause",
        version_status=version_status,
        normative_level=normative_level,
    )


def make_bundle(evidence_items=None) -> EvidenceBundle:
    """构造证据包"""
    return EvidenceBundle(
        bundle_id=f"bundle-{uuid.uuid4().hex[:8]}",
        evidence_items=evidence_items or [],
        sufficiency_score=0.90,
        is_sufficient=True,
    )


def make_turn(
    turn_number: int,
    query: str,
    answer: str = "",
    intent: str = "",
    entities=None,
    constraints=None,
    user_confirmed_facts=None,
) -> Turn:
    """构造对话轮次"""
    return Turn(
        turn_id=f"turn-{turn_number}",
        turn_number=turn_number,
        query=query,
        answer=answer,
        intent=intent,
        entities=entities or [],
        constraints=constraints or [],
        user_confirmed_facts=user_confirmed_facts or [],
    )


def run_async(coro):
    """同步运行异步协程（兼容 event loop 已关闭的场景）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ============================================================
# 场景 1: 声明验证全流程
# ============================================================


class TestVerificationE2E:
    """声明验证端到端测试"""

    def test_numeric_claim_validated(self):
        """数值声明验证通过"""
        print("\n[验证层] 进入声明验证流程 — 数值声明")

        evidence = make_evidence(
            content="核心一级资本充足率不得低于5%。",
            citation="《商业银行资本管理办法》第23条",
        )
        bundle = make_bundle([evidence])
        answer = "根据《商业银行资本管理办法》第23条，核心一级资本充足率不得低于5%。"

        print(f"[验证层] 回答文本: {answer}")
        print(f"[验证层] 证据数: {bundle.evidence_count}")

        validator = AnswerValidator()
        result = validator.validate_answer(answer, bundle)

        print(f"[验证层] 验证结果: valid={result.valid}, action={result.action}")
        print(f"[验证层] 声明数: {len(result.claim_results)}")
        for cr in result.claim_results:
            print(f"  - 声明: {cr.claim_text[:40]}... → status={cr.status}")

        assert result.action in ("accept", "retry")
        assert len(result.claim_results) >= 1

    def test_unsupported_claim_detected(self):
        """无证据声明被检测"""
        print("\n[验证层] 进入声明验证流程 — 无证据声明")

        evidence = make_evidence(content="核心一级资本充足率不得低于5%。")
        bundle = make_bundle([evidence])
        answer = (
            "根据规定，核心一级资本充足率不得低于5%。"
            "另外，杠杆率不得低于3%。"
            "此外，流动性覆盖率应不低于100%。"
        )

        print(f"[验证层] 回答含 3 个声明，仅 1 个有证据")

        validator = AnswerValidator()
        result = validator.validate_answer(answer, bundle)

        print(f"[验证层] 无证据声明数: {len(result.unsupported_claims)}")
        for uc in result.unsupported_claims:
            print(f"  - 无证据: {uc.claim_text[:40]}...")

        assert len(result.unsupported_claims) >= 1

    def test_citation_validation(self):
        """引用验证"""
        print("\n[验证层] 进入引用验证流程")

        evidence = make_evidence(
            content="核心一级资本充足率不得低于5%。",
            citation="《商业银行资本管理办法》第23条",
        )
        bundle = make_bundle([evidence])
        answer = "根据《商业银行资本管理办法》第23条，核心一级资本充足率不得低于5%。"

        validator = AnswerValidator()
        result = validator.validate_answer(answer, bundle)

        print(f"[验证层] 引用验证: citation_result={'存在' if result.citation_result else '空'}")
        print(f"[验证层] 验证通过: {result.valid}")

        assert result.citation_result is not None


# ============================================================
# 场景 2: 多轮记忆全流程
# ============================================================


class TestMemoryE2E:
    """多轮记忆端到端测试"""

    def test_multi_turn_memory_flow(self):
        """多轮对话记忆更新与上下文获取"""
        print("\n[记忆层] 进入多轮记忆流程")

        manager = MemoryManager(mock=True)
        session_id = manager.create_session()
        print(f"[记忆层] 创建会话: {session_id}")

        # 第 1 轮
        turn1 = make_turn(
            1,
            "核心一级资本充足率是多少？",
            "核心一级资本充足率不得低于5%。",
            intent="threshold_query",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"},
                {"entity_type": "doc_name", "value": "商业银行资本管理办法"},
            ],
        )
        run_async(manager.on_turn_complete(session_id, turn1))
        print(f"[记忆层] 第 1 轮完成: query='{turn1.query}'")

        # 第 2 轮
        turn2 = make_turn(
            2,
            "这个比例适用于哪些银行？",
            "适用于所有商业银行。",
            intent="scope_query",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"},
            ],
        )
        run_async(manager.on_turn_complete(session_id, turn2))
        print(f"[记忆层] 第 2 轮完成: query='{turn2.query}'")

        # 获取上下文
        ctx = run_async(manager.get_context_for_new_query(session_id))
        print(f"[记忆层] 上下文: metrics={ctx.mentioned_metrics}, docs={ctx.mentioned_docs}")
        print(f"[记忆层] 最近轮次: {len(ctx.recent_turns)}")

        assert len(ctx.recent_turns) >= 1
        assert "核心一级资本充足率" in ctx.mentioned_metrics

    def test_reference_resolution_with_memory(self):
        """基于记忆上下文的指代消解"""
        print("\n[记忆层] 进入指代消解流程")

        manager = MemoryManager(mock=True)
        session_id = manager.create_session()

        turn1 = make_turn(
            1,
            "核心一级资本充足率是多少？",
            "不得低于5%。",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"},
            ],
        )
        run_async(manager.on_turn_complete(session_id, turn1))
        print(f"[记忆层] 第 1 轮: 提到 '核心一级资本充足率'")

        ctx = run_async(manager.get_context_for_new_query(session_id))

        resolver = MemoryReferenceResolver()
        result = resolver.resolve("这个比例适用吗", ctx)

        print(f"[记忆层] 原始查询: '这个比例适用吗'")
        print(f"[记忆层] 消解结果: '{result.resolved_query}'")
        print(f"[记忆层] 是否消解: {result.was_resolved}")

        assert result.was_resolved
        assert "核心一级资本充足率" in result.resolved_query

    def test_user_confirmed_fact_persisted(self):
        """用户确认事实持久化到长期记忆"""
        print("\n[记忆层] 进入确认事实持久化流程")

        manager = MemoryManager(mock=True)
        session_id = manager.create_session()

        turn = make_turn(
            1,
            "确认一下，核心一级资本充足率是5%对吗？",
            "是的，核心一级资本充足率不得低于5%。",
            user_confirmed_facts=[
                {
                    "fact_type": "threshold",
                    "metric": "核心一级资本充足率",
                    "value": "5%",
                    "source": "《商业银行资本管理办法》第23条",
                }
            ],
        )
        run_async(manager.on_turn_complete(session_id, turn))
        print(f"[记忆层] 确认事实已保存")

        ctx = run_async(manager.get_context_for_new_query(session_id))
        print(f"[记忆层] 确认事实数: {len(ctx.confirmed_facts)}")

        assert len(ctx.confirmed_facts) >= 1


# ============================================================
# 场景 3: 工具调用全流程
# ============================================================


class TestToolsE2E:
    """工具平台端到端测试"""

    def test_calculator_tool_flow(self):
        """计算器工具完整调用流程"""
        print("\n[工具层] 进入计算器调用流程")

        platform = create_default_platform()
        print("[工具层] 工具平台已创建")

        result = platform.invoke(
            "calculator",
            {"expression": "8 * 1.25"},
            caller_role="authenticated",
        )

        print(f"[工具层] 调用结果: success={result.success}")
        print(f"[工具层] 返回数据: {result.data}")
        print(f"[工具层] 耗时: {result.execution_time_ms:.1f}ms")
        print(f"[工具层] 事件日志: {len(platform.get_event_log())} 条")

        assert result.success
        assert len(platform.get_event_log()) == 1

    def test_tool_idempotency(self):
        """工具幂等缓存"""
        print("\n[工具层] 进入幂等缓存测试")

        platform = create_default_platform()

        result1 = platform.invoke(
            "calculator",
            {"expression": "100 / 4"},
            caller_role="authenticated",
        )
        print(f"[工具层] 第一次调用: success={result1.success}")

        result2 = platform.invoke(
            "calculator",
            {"expression": "100 / 4"},
            caller_role="authenticated",
        )
        print(f"[工具层] 第二次调用（幂等）: success={result2.success}")

        events = platform.get_event_log()
        print(f"[工具层] 事件日志总数: {len(events)}")

        assert result1.success
        assert result2.success
        # 幂等命中时不写新事件日志（在步骤4直接返回缓存）
        assert len(events) == 1

    def test_tool_not_found(self):
        """调用不存在的工具"""
        print("\n[工具层] 进入工具不存在测试")

        platform = create_default_platform()
        result = platform.invoke("nonexistent_tool", {})

        print(f"[工具层] 不存在工具: success={result.success}, error={result.error}")

        assert not result.success
        assert "不存在" in result.error

    def test_tool_error_recovery_with_fallback(self):
        """工具失败 → 重试 → 降级"""
        print("\n[工具层] 进入错误恢复 + 降级测试")

        from agent_platform.tools.adapters.calculator import (
            CALCULATOR_MANIFEST,
            calculator_handler,
        )
        from agent_platform.tools.executor.executor import ToolExecutor
        from agent_platform.tools.registry.registry import ToolRegistry
        from agent_platform.tools.tool_models import RetryPolicy, ToolManifest

        registry = ToolRegistry()

        fail_manifest = ToolManifest(
            name="fail_tool",
            description="总是失败的工具",
            input_schema={},
            retry_policy=RetryPolicy(
                max_retries=2,
                retryable_errors=["RuntimeError"],
                backoff_base=0.01,
                backoff_max=0.1,
            ),
            fallback_tool="calculator",
        )

        def fail_handler(input_data):
            raise RuntimeError("模拟失败")

        registry.register(fail_manifest, fail_handler)
        registry.register(CALCULATOR_MANIFEST, calculator_handler)

        executor = ToolExecutor(registry)
        result = executor.invoke(
            "fail_tool",
            {"expression": "2+2"},
            caller_role="authenticated",
        )

        print(f"[工具层] 失败后降级: success={result.success}")
        print(f"[工具层] 降级使用: {result.fallback_used}")
        print(f"[工具层] 降级工具: {result.tool_name}")
        print(f"[工具层] 重试次数: {result.retries}")

        assert result.success
        assert result.fallback_used
        assert result.tool_name == "calculator"


# ============================================================
# 场景 4: 网关全流程
# ============================================================


class TestGatewayE2E:
    """网关端到端测试"""

    def test_auth_and_rate_limit_flow(self):
        """鉴权 + 限流完整流程"""
        print("\n[网关层] 进入鉴权 + 限流流程")

        auth = AuthHandler()
        limiter = RateLimiter()

        auth_result = auth.authenticate(None)
        print(f"[网关层] 匿名鉴权: authenticated={auth_result.authenticated}, role={auth_result.role}")

        rate_result = limiter.check("user-1", auth_result.role)
        print(f"[网关层] 限流检查: allowed={rate_result.allowed}, remaining={rate_result.remaining}")

        assert auth_result.role == "anonymous"
        assert rate_result.allowed

    def test_rate_limit_exceeded(self):
        """限流触发"""
        print("\n[网关层] 进入限流触发测试")

        limiter = RateLimiter()

        allowed_count = 0
        for i in range(12):
            result = limiter.check("user-test", "anonymous")
            if result.allowed:
                allowed_count += 1

        print(f"[网关层] 12 次请求中允许: {allowed_count} 次")
        assert allowed_count <= 10

    def test_idempotency_flow(self):
        """幂等请求处理"""
        print("\n[网关层] 进入幂等处理流程")

        handler = IdempotencyHandler()
        request_id = "req-idempotent-001"

        # 首次检查 — 应为 new
        first = handler.check_or_cache(request_id)
        print(f"[网关层] 首次检查: status={first.status}")

        # 缓存响应
        response = {"answer": "测试答案", "status": "ok"}
        handler.cache_response(request_id, response)
        print("[网关层] 已缓存响应")

        # 二次检查 — 应为 cached
        second = handler.check_or_cache(request_id)
        print(f"[网关层] 二次检查: status={second.status}, is_cached={second.is_cached}")

        assert first.is_new
        assert second.is_cached
        assert second.cached_response == response


# ============================================================
# 场景 5: 可观测性全流程
# ============================================================


class TestObservabilityE2E:
    """可观测性端到端测试"""

    def test_full_trace_with_metrics(self):
        """完整链路追踪 + 指标采集"""
        print("\n[可观测层] 进入链路追踪 + 指标采集流程")

        tracer = Tracer()
        collector = MetricsCollector()
        request_id = generate_request_id()
        print(f"[可观测层] request_id: {request_id}")

        with SpanContext(
            tracer, tracer.start_span("QueryRequest", query="核心一级资本充足率")
        ) as root:
            root.set_attribute("request_id", request_id)

            with SpanContext(
                tracer, tracer.start_span("QueryUnderstanding", parent=root)
            ) as qu:
                qu.set_attribute("intent", "threshold_query")
                collector.counter("agent_queries_total").inc()

            with SpanContext(
                tracer, tracer.start_span("Retrieval", parent=root)
            ) as ret:
                ret.set_attribute("hits", 5)
                collector.histogram("retrieval_latency_seconds").observe(0.3)
                collector.counter("retrieval_calls_total").inc()

            with SpanContext(
                tracer, tracer.start_span("Verification", parent=root)
            ) as verify:
                verify.set_attribute("passed", True)
                collector.histogram("evidence_sufficiency_score").observe(0.92)

        trace = tracer.finish()

        print(f"[可观测层] 追踪根 Span: {trace.name}")
        print(f"[可观测层] 子 Span 数: {len(trace.children)}")
        for child in trace.children:
            print(f"  - {child.name} (duration={child.duration_ms:.1f}ms)")

        export = collector.export()
        print(f"[可观测层] 指标: queries={export['agent_queries_total']['values']['']}")
        print(f"[可观测层] 指标: retrieval_calls={export['retrieval_calls_total']['values']['']}")
        print(f"[可观测层] 指标: sufficiency_avg={export['evidence_sufficiency_score']['avg']}")

        assert trace.name == "QueryRequest"
        assert len(trace.children) == 3
        assert export["agent_queries_total"]["values"][""] == 1.0
        assert export["evidence_sufficiency_score"]["avg"] == 0.92

    def test_structured_logging_and_replay(self):
        """结构化日志 + 日志重放"""
        print("\n[可观测层] 进入结构化日志 + 重放流程")

        import io

        test_logger = logging.getLogger("test.e2e.observability")
        test_logger.setLevel(logging.DEBUG)
        formatter = StructuredLogFormatter()

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        test_logger.addHandler(handler)

        test_logger.info("查询开始", extra={"request_id": "req-log-1", "span_id": "span-1"})
        test_logger.info("检索完成", extra={"request_id": "req-log-1", "span_id": "span-2"})
        test_logger.error("验证失败", extra={"request_id": "req-log-1", "span_id": "span-1"})

        output = stream.getvalue()
        test_logger.removeHandler(handler)

        log_lines = output.strip().split("\n")
        print(f"[可观测层] 日志行数: {len(log_lines)}")

        replayer = LogReplayer()
        count = replayer.load(log_lines)
        print(f"[可观测层] 重放加载: {count} 条")

        filtered = replayer.filter_by_request("req-log-1")
        print(f"[可观测层] 按 request_id 过滤: {len(filtered)} 条")

        assert count == 3
        assert len(filtered) == 3


# ============================================================
# 场景 6: 跨模块集成
# ============================================================


class TestCrossModuleE2E:
    """跨模块端到端集成测试"""

    def test_full_pipeline_gateway_to_verification(self):
        """完整流水线: 网关入口 → 记忆 → 工具 → 验证 → 可观测"""
        print("\n[集成层] 进入跨模块完整流水线")

        # 初始化所有组件
        auth = AuthHandler()
        limiter = RateLimiter()
        idempotency = IdempotencyHandler()
        memory = MemoryManager(mock=True)
        tools = create_default_platform()
        validator = AnswerValidator()
        tracer = Tracer()
        collector = MetricsCollector()

        request_id = generate_request_id()
        print(f"[集成层] request_id: {request_id}")

        # ── 步骤 1: 网关鉴权 ──
        print("[集成层] 步骤 1: 网关鉴权")
        auth_result = auth.authenticate(None)
        assert auth_result.role == "anonymous"

        # ── 步骤 2: 限流检查 ──
        print("[集成层] 步骤 2: 限流检查")
        rate_result = limiter.check("cross-user", auth_result.role)
        assert rate_result.allowed

        # ── 步骤 3: 幂等检查 ──
        print("[集成层] 步骤 3: 幂等检查")
        idem_result = idempotency.check_or_cache(request_id)
        assert idem_result.is_new  # 首次请求

        # ── 步骤 4: 多轮记忆上下文 ──
        print("[集成层] 步骤 4: 获取记忆上下文")
        session_id = memory.create_session()
        turn1 = make_turn(
            1,
            "核心一级资本充足率是多少？",
            "不得低于5%。",
            entities=[{"entity_type": "metric_name", "value": "核心一级资本充足率"}],
        )
        run_async(memory.on_turn_complete(session_id, turn1))
        ctx = run_async(memory.get_context_for_new_query(session_id))
        print(f"[集成层] 记忆上下文: metrics={ctx.mentioned_metrics}")

        # ── 步骤 5: 工具调用（计算） ──
        print("[集成层] 步骤 5: 工具调用")
        tool_result = tools.invoke(
            "calculator",
            {"expression": "5 * 1.0"},
            caller_role=auth_result.role,
        )
        assert tool_result.success
        print(f"[集成层] 计算结果: {tool_result.data}")

        # ── 步骤 6: 声明验证 ──
        print("[集成层] 步骤 6: 声明验证")
        evidence = make_evidence(
            content="核心一级资本充足率不得低于5%。",
            citation="《商业银行资本管理办法》第23条",
        )
        bundle = make_bundle([evidence])
        answer = "根据《商业银行资本管理办法》第23条，核心一级资本充足率不得低于5%。"
        validation = validator.validate_answer(answer, bundle)
        print(f"[集成层] 验证结果: valid={validation.valid}, action={validation.action}")

        # ── 步骤 7: 可观测性记录 ──
        print("[集成层] 步骤 7: 可观测性记录")
        with SpanContext(
            tracer, tracer.start_span("FullPipeline", request_id=request_id)
        ) as root:
            root.set_attribute("auth_role", auth_result.role)
            root.set_attribute("memory_metrics", len(ctx.mentioned_metrics))
            root.set_attribute("tool_success", tool_result.success)
            root.set_attribute("validation_action", validation.action)

            collector.counter("agent_queries_total").inc()
            collector.histogram("agent_query_latency_seconds").observe(1.2)
            collector.gauge("budget_consumed_ratio").set(0.5)

        trace = tracer.finish()
        export = collector.export()

        print(f"[集成层] 追踪: {trace.name}, children={len(trace.children)}")
        print(f"[集成层] 指标: queries={export['agent_queries_total']['values']['']}")

        # ── 步骤 8: 缓存响应（幂等） ──
        print("[集成层] 步骤 8: 缓存响应")
        idempotency.cache_response(
            request_id,
            {"answer": answer, "validation": validation.action},
        )

        cached = idempotency.check_or_cache(request_id)
        assert cached.is_cached
        print(f"[集成层] 幂等缓存命中: {cached.cached_response['validation']}")

        assert trace.name == "FullPipeline"
        assert export["agent_queries_total"]["values"][""] == 1.0
        assert cached.cached_response["validation"] == validation.action

        print("[集成层] 完整流水线通过")

    def test_error_recovery_flow(self):
        """错误恢复流程: 工具失败 → 降级 → 继续"""
        print("\n[集成层] 进入错误恢复流程")

        from agent_platform.tools.adapters.calculator import (
            CALCULATOR_MANIFEST,
            calculator_handler,
        )
        from agent_platform.tools.executor.executor import ToolExecutor
        from agent_platform.tools.registry.registry import ToolRegistry
        from agent_platform.tools.tool_models import RetryPolicy, ToolManifest

        registry = ToolRegistry()

        fail_manifest = ToolManifest(
            name="fail_tool",
            description="总是失败的工具",
            input_schema={},
            retry_policy=RetryPolicy(
                max_retries=2,
                retryable_errors=["RuntimeError"],
                backoff_base=0.01,
                backoff_max=0.1,
            ),
            fallback_tool="calculator",
        )

        def fail_handler(input_data):
            raise RuntimeError("模拟失败")

        registry.register(fail_manifest, fail_handler)
        registry.register(CALCULATOR_MANIFEST, calculator_handler)

        executor = ToolExecutor(registry)
        result = executor.invoke(
            "fail_tool",
            {"expression": "2+2"},
            caller_role="authenticated",
        )

        print(f"[集成层] 工具失败后降级: success={result.success}")
        print(f"[集成层] 降级使用: {result.fallback_used}")
        print(f"[集成层] 降级工具: {result.tool_name}")
        print(f"[集成层] 重试次数: {result.retries}")

        assert result.success
        assert result.fallback_used
        assert result.tool_name == "calculator"

        print("[集成层] 错误恢复通过")
