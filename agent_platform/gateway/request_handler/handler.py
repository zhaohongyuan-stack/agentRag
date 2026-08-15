"""
请求处理器 — Agent 主流程编排

完整流程:
  用户问题
    → request_handler 接收请求，创建/恢复会话
    → query_understanding 构建 QuerySpec（意图+实体+约束+歧义+复杂度）
    → query_rewriter 查询改写（指代消解+同义词扩展）    [Phase 2]
    → route_policy 综合路由（规则+复杂度+风险 → P0-P4）  [Phase 2]
    → state_machine 管理状态迁移
    → [L0/P0] → 直接回复 → 返回
    → [Phase 5] Planner 规划检索策略 + 越界检测
    → [Phase 5] Agent 协作 Loop（Retriever↔Evaluator，BudgetController 控轮次）
    → [证据不足] → 拒答 / 澄清 → 返回
    → [证据充分] → generator 生成回答（LLM 接地生成）
    → [Phase 5] Verifier 声明级验证（needs_retry → 补充检索）
    → 返回带引用的回答

Phase 2 增强:
  - LLM 接地回答生成（DeepSeek via OpenAI 兼容 API）
  - 查询改写（指代消解+同义词扩展）
  - 综合路由（风险评级+P0-P4 执行路径）
  - 证据去重和父文档聚合
  - 兼容 Phase 1 模板模式和 Mock 服务

Phase 5 增强（Agentic RAG）:
  - Planner Agent：路由后规划检索策略、越界检测
  - 检索-评估 Loop：Evaluator 输出 retrieval_suggestion 驱动多轮检索
  - BudgetController 控制检索轮次、token、耗时
  - Verifier Agent：回答生成后声明级验证，needs_retry 触发补充检索
  - V1 检索逻辑（search_by_spec + evidence_builder）作为 Retriever 实现
  - Agent 异常自动回退到 V1 线性流程，保证向后兼容
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from agent_platform.agents import (
    AgentContext,
    EvaluatorAgent,
    PlannerAgent,
    VerifierAgent,
)
from agent_platform.evidence.evidence_assembler import EvidenceBuilder
from agent_platform.generation.grounded_generator import GroundedGenerator, TemplateGenerator
from agent_platform.observability import TraceCollector, get_trace_store
from agent_platform.orchestration.budget_controller import BudgetController
from agent_platform.orchestration.state_machine import AgentState, StateMachine
from agent_platform.query_understanding import QuerySpec, QuerySpecBuilder
from agent_platform.query_understanding.query_rewriter import QueryRewriter
from agent_platform.routing.route_policy import RoutePolicy
from agent_platform.routing.rule_router import RuleRouter
from agent_platform.runtime.llm_client import get_llm_client

from .event_callback import EventCallback, LoggingEventCallback
from .models import QueryRequest, QueryResponse
from .retrieval_client import RetrievalClient
from ..session_handler.session_state import SessionManager

logger = logging.getLogger(__name__)

# ── LangGraph feature flag（阶段3）──
# USE_LANGGRAPH=true 时启用图编排流程；图流程异常自动回退手写 _run_agent_flow
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "").strip().lower() in ("1", "true", "yes", "on")


def _now_ms() -> float:
    """高精度当前时间戳（毫秒）"""
    return time.perf_counter() * 1000


class RequestHandler:
    """
    请求处理器 — Agent 主流程编排器

    接收用户查询，编排查询理解、查询改写、路由、检索、证据组装、回答生成，
    返回带引用的回答。

    Phase 2 增强:
      - 默认使用 RoutePolicy（综合路由）替代 RuleRouter
      - 默认使用 GroundedGenerator（LLM 生成）替代 TemplateGenerator
      - 新增 QueryRewriter 查询改写步骤
      - 所有组件均可通过构造函数替换，保持 Phase 1 向后兼容

    Phase 5 增强（Agentic RAG）:
      - PlannerAgent 路由后规划检索策略、越界检测
      - EvaluatorAgent 评估证据充分性、输出检索建议驱动 Loop
      - VerifierAgent 回答生成后声明级验证
      - BudgetController 控制检索轮次（默认按 path_id 分配）
      - Agent 异常自动回退 V1 线性流程（agent_fallback=True）
    """

    def __init__(
        self,
        retrieval_client: Optional[RetrievalClient] = None,
        query_spec_builder: Optional[QuerySpecBuilder] = None,
        router: Optional[Any] = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        generator: Optional[Any] = None,
        session_manager: Optional[SessionManager] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        route_policy: Optional[RoutePolicy] = None,
        planner_agent: Optional[PlannerAgent] = None,
        evaluator_agent: Optional[EvaluatorAgent] = None,
        verifier_agent: Optional[VerifierAgent] = None,
        budget_controller: Optional[BudgetController] = None,
        enable_agent_loop: bool = True,
        enable_trace: bool = True,
    ):
        """
        Args:
            retrieval_client: 检索客户端，默认使用进程内 Mock 模式
            query_spec_builder: 查询理解构建器
            router: 路由器（Phase 1 兼容参数，优先使用 route_policy）
            evidence_builder: 证据组装器
            generator: 回答生成器，默认使用 GroundedGenerator
            session_manager: 会话管理器
            query_rewriter: 查询改写器，默认使用 QueryRewriter
            route_policy: 综合路由策略，默认使用 RoutePolicy
            planner_agent: Phase 5 规划Agent，None则使用默认实例
            evaluator_agent: Phase 5 评估Agent，None则使用默认实例
            verifier_agent: Phase 5 验证Agent，None则使用默认实例
            budget_controller: Phase 5 预算控制器，None则按 path_id 动态创建
            enable_agent_loop: Phase 5 是否启用Agent协作Loop，False则回退V1线性流程
            enable_trace: Phase 6 是否启用执行追踪收集与存储
        """
        self._retrieval_client = retrieval_client or RetrievalClient(in_process=True)
        self._query_spec_builder = query_spec_builder or QuerySpecBuilder()
        # Phase 2: 优先使用 route_policy，回退到 router 参数，最后使用默认 RoutePolicy
        if route_policy is not None:
            self._route_policy = route_policy
        elif router is not None:
            # 向后兼容: 如果传入了 router（RuleRouter），包装成 RoutePolicy
            self._route_policy = RoutePolicy(rule_router=router if isinstance(router, RuleRouter) else None)
        else:
            self._route_policy = RoutePolicy()
        # 阈值 0.60: 对于 table_lookup 等结构化检索，1-2 条证据即可充分
        # (ES_DEFAULT默认 0.85 过于严格，2/3 条证据只有 0.667)
        self._evidence_builder = evidence_builder or EvidenceBuilder(sufficiency_threshold=0.60)
        # Phase 2: 默认使用 GroundedGenerator（内部自动降级到模板）
        self._generator = generator or GroundedGenerator()
        self._session_manager = session_manager or SessionManager()
        self._query_rewriter = query_rewriter or QueryRewriter()

        # Phase 5: Agent 协作层组件
        self._enable_agent_loop = enable_agent_loop
        llm_client = get_llm_client()
        self._planner_agent = planner_agent or PlannerAgent(llm_client=llm_client)
        self._evaluator_agent = evaluator_agent or EvaluatorAgent(llm_client=llm_client)
        self._verifier_agent = verifier_agent or VerifierAgent(llm_client=llm_client)
        # 预算控制器：按 path_id 分配预算，每次查询前重置
        self._budget_controller = budget_controller

        # Phase 6: 执行追踪
        self._trace_store = get_trace_store()
        self._enable_trace = enable_trace

        # ── Phase3: Redis 工具结果缓存 + 会话记忆 ──
        self._redis = None
        self._tool_cache = None
        try:
            import redis as _redis_mod
            _redis_host = os.getenv("REDIS_HOST", "localhost")
            _redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self._redis = _redis_mod.Redis(
                host=_redis_host, port=_redis_port, db=0,
                decode_responses=True, socket_timeout=2,
            )
            self._redis.ping()
            logger.info(f"[Handler] Redis已连接({_redis_host}:{_redis_port})，工具缓存+会话记忆启用")
        except Exception as _redis_err:
            logger.warning(f"[Handler] Redis不可用，工具缓存降级: {_redis_err}")
            self._redis = None

    def _safe_callback(self, method_name: str, *args, **kwargs):
        """安全调用回调方法（异常静默，不阻塞主流程）"""
        cb = getattr(self._event_callback, method_name, None)
        if cb is not None:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.warning("[EventCallback] %s 异常: %s", method_name, e, exc_info=True)

    def handle_query(
        self,
        request: QueryRequest,
        event_callback: Optional[EventCallback] = None,
    ) -> QueryResponse:
        """
        处理用户查询请求 — 主入口

        Args:
            request: 查询请求
            event_callback: 可选的事件回调，用于 SSE 流式推送

        Returns:
            查询响应
        """
        self._event_callback = event_callback or LoggingEventCallback()
        start_time = _now_ms()
        request_id = str(uuid.uuid4())

        # Phase 6: 创建执行追踪收集器（贯穿本次问答生命周期）
        trace_collector = TraceCollector(
            session_id=request.session_id or str(uuid.uuid4()),
            request_id=request_id,
        )
        trace_collector.record_query(request.query)

        # 幂等检查
        if request.idempotency_key:
            cached = self._session_manager.check_idempotency(request.idempotency_key)
            if cached:
                logger.info(f"幂等命中: {request.idempotency_key}")
                return QueryResponse(**cached["response"])

        # 创建/恢复会话
        session = self._session_manager.get_or_create(request.session_id)
        sm: StateMachine = session.state_machine

        # 每次新查询重置状态机（会话保持对话历史，状态机按查询重置）
        sm.reset()
        sm.start()

        # 空问题校验
        if not request.query or not request.query.strip():
            return QueryResponse(
                request_id=request_id,
                session_id=session.session_id,
                answer="问题不能为空，请输入您的问题。",
                intent="unknown",
                complexity="L0",
                is_refusal=True,
                refusal_reason="空问题",
                state_trace=sm.get_state_trace(),
                latency_ms=(_now_ms() - start_time),
            )

        try:
            # ── 状态机: RECEIVED → NORMALIZED ──
            sm.transition(AgentState.NORMALIZED, {"step": "normalize"})
            sm.transition(AgentState.CONTEXT_RESOLVED, {"step": "context_resolve"})

            # ── 查询理解: 构建 QuerySpec ──
            _step_t = _now_ms()
            query_spec: QuerySpec = self._query_spec_builder.build(
                request.query, session_id=session.session_id
            )
            logger.info(f"[步骤] 查询理解 → intent={query_spec.intent}, complexity={query_spec.complexity} ({_now_ms() - _step_t:.0f}ms)")
            sm.transition(AgentState.ANALYZED, {
                "step": "analyze",
                "intent": query_spec.intent,
                "complexity": query_spec.complexity,
            })

            # ── 查询改写（Phase 2）──
            _step_t = _now_ms()
            # 从会话历史构建上下文，用于指代消解
            session_context = self._build_session_context(session)
            rewritten = self._query_rewriter.rewrite(
                original_query=request.query,
                query_spec=query_spec,
                session_context=session_context,
            )
            # 使用改写后的查询替代原始查询
            search_query = rewritten.contextualized_query or request.query
            if search_query != request.query:
                logger.info(f"[步骤] 查询改写 → '{search_query[:80]}' ({_now_ms() - _step_t:.0f}ms)")
            else:
                logger.info(f"[步骤] 查询改写 → 无变化 ({_now_ms() - _step_t:.0f}ms)")

            # ── 路由决策（Phase 2: 综合路由）──
            _step_t = _now_ms()
            route_decision = self._route_policy.decide(query_spec)
            logger.info(f"[步骤] 路由决策 → level={route_decision.level}, channels={route_decision.channels} ({_now_ms() - _step_t:.0f}ms)")
            sm.transition(AgentState.ROUTED, {
                "step": "route",
                "level": route_decision.level,
                "channels": route_decision.channels,
                "path_id": route_decision.path_id,
                "risk_level": route_decision.risk_level,
            })

            # ── 分支处理 ──
            # 策略变更：先检索后澄清（不再在检索前提前拦截歧义问题）
            # 即使检测到歧义，也先尝试检索；若证据不足再考虑澄清

            # 分支1: L0 问候 → 直接回复（ROUTED → RESPONDING 合法迁移）
            if route_decision.level == "L0":
                sm.transition(AgentState.RESPONDING, {"step": "direct_respond", "reason": "L0"})
                answer = self._generator.generate(
                    intent=query_spec.intent,
                    evidence_bundle=None,
                    query_text=request.query,
                )
                response = self._build_response(
                    request_id=request_id,
                    session=session,
                    sm=sm,
                    query_spec=query_spec,
                    route_decision=route_decision,
                    answer=answer,
                    start_time=start_time,
                )
                self._finalize(session, request, response, request_id)
                return response

            # 分支2: 正常检索流程（Phase 5: Agent协作Loop / V1线性流程）
            # ──────────────────────────────────────────────────────
            # Phase 5 增量改造策略：
            #   - enable_agent_loop=True 时走 Agent 协作 Loop（Planner→Retriever↔Evaluator→Verifier）
            #   - Agent 调用异常时自动回退到 V1 线性流程，保证向后兼容
            #   - V1 检索逻辑（search_by_spec + evidence_builder）作为 Retriever 实现，不重写
            # ──────────────────────────────────────────────────────
            # table_lookup 意图：确保 table 通道在 channels 中（P2 默认不含 table）
            if query_spec.intent == "table_lookup" and "table" not in route_decision.channels:
                route_decision.channels = list(route_decision.channels) + ["table"]

            if self._enable_agent_loop:
                # ── LangGraph 图编排（USE_LANGGRAPH=true 启用）──
                # 图流程异常 → 回退手写 Agent 流程；手写流程再异常 → 下方回退 V1
                if USE_LANGGRAPH:
                    try:
                        return self._run_graph_flow(
                            request=request,
                            request_id=request_id,
                            session=session,
                            sm=sm,
                            start_time=start_time,
                            query_spec=query_spec,
                            route_decision=route_decision,
                            rewritten=rewritten,
                            search_query=search_query,
                            trace_collector=trace_collector if self._enable_trace else None,
                            event_callback=self._event_callback,
                        )
                    except Exception as graph_err:
                        logger.warning(
                            f"[LangGraph] 图流程异常，回退手写Agent流程: {graph_err}",
                            exc_info=True,
                        )
                        # 状态机可能停在中间状态，重置到 ROUTED 后重走手写流程
                        self._reset_sm_for_fallback(sm, query_spec, route_decision)
                try:
                    return self._run_agent_flow(
                        request=request,
                        request_id=request_id,
                        session=session,
                        sm=sm,
                        start_time=start_time,
                        query_spec=query_spec,
                        route_decision=route_decision,
                        rewritten=rewritten,
                        search_query=search_query,
                        trace_collector=trace_collector if self._enable_trace else None,
                        event_callback=self._event_callback,
                    )
                except Exception as agent_err:
                    # Agent 协作流程异常 → 回退 V1 线性流程
                    logger.warning(
                        f"[Phase5] Agent协作流程异常，回退V1线性流程: {agent_err}",
                        exc_info=True,
                    )
                    # 状态机可能停在中间状态，重置到 ROUTED 后重新走 V1
                    self._reset_sm_for_fallback(sm, query_spec, route_decision)

            # V1 线性检索流程（fallback 或 enable_agent_loop=False）
            return self._run_v1_linear_flow(
                request=request,
                request_id=request_id,
                session=session,
                sm=sm,
                start_time=start_time,
                query_spec=query_spec,
                route_decision=route_decision,
                rewritten=rewritten,
                search_query=search_query,
            )

        except Exception as e:
            logger.error(f"处理查询异常: {e}", exc_info=True)
            # 尝试迁移到 FAILED 状态
            try:
                if not sm.is_terminal():
                    sm.transition(AgentState.FAILED, {"step": "failed", "error": str(e)})
            except Exception:
                pass

            return QueryResponse(
                request_id=request_id,
                session_id=session.session_id,
                answer=f"处理您的问题时发生错误: {str(e)}",
                intent="unknown",
                complexity="L0",
                is_refusal=True,
                refusal_reason=f"内部错误: {str(e)}",
                state_trace=sm.get_state_trace(),
                latency_ms=(_now_ms() - start_time),
            )

    # ============================================================
    # 内部方法
    # ============================================================

    # ──────────────────────────────────────────────────────
    # Phase 6: 执行追踪
    # ──────────────────────────────────────────────────────

    def _finalize_and_save_trace(
        self,
        trace_collector: TraceCollector,
        total_latency_ms: int,
    ) -> None:
        """
        结束 trace 收集并持久化到 SQLite（容错，失败不影响主流程）

        Args:
            trace_collector: 本次问答的追踪收集器
            total_latency_ms: 总耗时
        """
        try:
            trace = trace_collector.finalize(total_latency_ms)
            self._trace_store.save(trace)
            logger.debug(
                "[Phase6] trace 已存储: session=%s, rounds=%d, latency=%dms",
                trace.get("session_id"),
                trace.get("retrieval_round_count", 0),
                total_latency_ms,
            )
        except Exception as e:
            logger.warning("[Phase6] trace 存储失败（不影响主流程）: %s", e, exc_info=True)

    # ──────────────────────────────────────────────────────
    # Phase 5: Agent 协作流程
    # ──────────────────────────────────────────────────────

    def _reset_sm_for_fallback(
        self,
        sm: StateMachine,
        query_spec: QuerySpec,
        route_decision: Any,
    ) -> None:
        """
        流程异常后将状态机重置到 ROUTED，供回退流程重新推进

        容错：重置失败不阻断回退（trace 可能不完整但功能可用）。
        """
        if sm.is_terminal():
            return
        try:
            # 从非终态回到 ROUTED 需要先重置（回退流程会自己推进状态）
            sm.reset()
            sm.start()
            sm.transition(AgentState.NORMALIZED, {"step": "normalize_fallback"})
            sm.transition(AgentState.CONTEXT_RESOLVED, {"step": "context_resolve_fallback"})
            sm.transition(AgentState.ANALYZED, {
                "step": "analyze_fallback",
                "intent": query_spec.intent,
                "complexity": query_spec.complexity,
            })
            sm.transition(AgentState.ROUTED, {
                "step": "route_fallback",
                "level": route_decision.level,
                "channels": route_decision.channels,
            })
        except Exception:
            logger.error("[Fallback] 状态机回退重置失败，继续执行回退流程")

    def _run_graph_flow(
        self,
        request: QueryRequest,
        request_id: str,
        session: Any,
        sm: StateMachine,
        start_time: float,
        query_spec: QuerySpec,
        route_decision: Any,
        rewritten: Any,
        search_query: str,
        trace_collector: Optional[TraceCollector] = None,
        event_callback: Optional[EventCallback] = None,
    ) -> QueryResponse:
        """
        LangGraph 图编排流程（阶段 3，_run_agent_flow 的等价替换）

        复用同一套组件（V1 检索/EvidenceBuilder/Generator/3个LLM Agent）与
        同一套业务硬约束，仅把编排逻辑换成 StateGraph（见 langgraph_graph 包）。
        状态机保留为 trace 记录器，事件回调/trace 记录与原流程一致。

        异常由 handle_query 捕获后回退 _run_agent_flow。
        """
        from agent_platform.orchestration.langgraph_graph import (
            GraphRuntime,
            build_graph,
        )

        # ── 初始化预算控制器（按 path_id 分配，与 _run_agent_flow 一致）──
        path_id = getattr(route_decision, "path_id", "P2") or "P2"
        budget_ctrl = self._budget_controller or BudgetController(path_id=path_id)
        # 每次查询重置预算（避免跨查询累积）
        if self._budget_controller is None:
            budget_ctrl.allocate(path_id)
        logger.info(f"[LangGraph] 预算初始化 path={path_id}, {budget_ctrl}")

        context = AgentContext(
            session_id=session.session_id,
            query=request.query,
            query_spec=query_spec,
            route_decision=route_decision,
            budget_controller=budget_ctrl,
        )

        runtime = GraphRuntime(
            handler=self,
            request=request,
            request_id=request_id,
            session=session,
            sm=sm,
            start_time=start_time,
            query_spec=query_spec,
            route_decision=route_decision,
            rewritten=rewritten,
            search_query=search_query,
            trace_collector=trace_collector,
            event_callback=event_callback,
            retrieval_client=self._retrieval_client,
            evidence_builder=self._evidence_builder,
            generator=self._generator,
            planner_agent=self._planner_agent,
            evaluator_agent=self._evaluator_agent,
            verifier_agent=self._verifier_agent,
            redis=self._redis,
            budget_controller=budget_ctrl,
            query_spec_builder=self._query_spec_builder,
        )

        # 懒加载并缓存编译后的图（进程内仅付一次编译成本）
        if getattr(self, "_langgraph_graph", None) is None:
            self._langgraph_graph = build_graph()

        final_state = self._langgraph_graph.invoke({
            "runtime": runtime,
            "context": context,
        })
        response = final_state.get("response")
        if response is None:
            raise RuntimeError(
                f"LangGraph 流程未产出响应: error={final_state.get('error')}"
            )
        return response

    def _run_agent_flow(
        self,
        request: QueryRequest,
        request_id: str,
        session: Any,
        sm: StateMachine,
        start_time: float,
        query_spec: QuerySpec,
        route_decision: Any,
        rewritten: Any,
        search_query: str,
        trace_collector: Optional[TraceCollector] = None,
        event_callback: Optional[EventCallback] = None,
    ) -> QueryResponse:
        """
        Phase 5 Agent 协作主流程

        流程:
          1. PLANNING: PlannerAgent 规划检索策略、越界检测
          2. RETRIEVING ↔ EVIDENCE_VALIDATING Loop:
             - Retriever = V1 search_by_spec + evidence_builder（不重写）
             - EvaluatorAgent 评估充分性，输出 retrieval_suggestion
             - 不充分且预算允许 → 调整策略重试
             - 充分 / 预算耗尽 → 退出 Loop
          3. GENERATING: generator 生成回答
          4. ANSWER_VALIDATING: VerifierAgent 声明级验证
             - needs_retry 且预算允许 → 补充检索后重新生成
             - 否则 → RESPONDING

        Agent 异常由上层 handle_query 捕获并回退 V1。
        """
        # ── 初始化预算控制器（按 path_id 分配）──
        path_id = getattr(route_decision, "path_id", "P2") or "P2"
        budget_ctrl = self._budget_controller or BudgetController(path_id=path_id)
        # 每次查询重置预算（避免跨查询累积）
        if self._budget_controller is None:
            budget_ctrl.allocate(path_id)
        logger.info(f"[Phase5] 预算初始化 path={path_id}, {budget_ctrl}")

        # ── 创建 AgentContext ──
        context = AgentContext(
            session_id=session.session_id,
            query=request.query,
            query_spec=query_spec,
            route_decision=route_decision,
            budget_controller=budget_ctrl,
        )

        # ── 1. PLANNING: PlannerAgent ──
        sm.transition(AgentState.PLANNING, {"step": "plan", "agent": "Planner"})
        if trace_collector:
            trace_collector.record_state("PLANNING")
        if event_callback:
            event_callback.on_agent_start("Planner", round=0, detail={
                "query": request.query,
                "intent": query_spec.intent,
                "complexity": query_spec.complexity,
            })
        _step_t = _now_ms()
        try:
            planner_result = self._planner_agent.run(context)
        except Exception as planner_err:
            # Planner LLM失败/超时 → 用默认plan继续Agent流程（不回退V1）
            # Agent流程后续有规则计算兜底,不能因Planner崩溃就放弃整个流程
            logger.warning(
                f"[Phase5][Planner] LLM调用失败，使用默认plan继续: {planner_err}"
            )
            context.retrieval_plan = {
                "is_out_of_domain": False,
                "domain_confidence": 0.5,
                "intent": query_spec.intent,
                "complexity": query_spec.complexity,
                "retrieval_plan": {
                    "strategies": list(route_decision.channels) if route_decision.channels else ["hybrid"],
                    "sub_queries": [search_query],
                },
                "sub_queries": [search_query],
                "reasoning": "Planner LLM不可用，使用路由默认检索策略",
            }
            planner_result = type("R", (), {
                "decision": "planned",
                "latency_ms": int(_now_ms() - _step_t),
            })()
            if event_callback:
                event_callback.on_agent_result(
                    "Planner",
                    decision="planned_fallback",
                    detail={
                        "is_out_of_domain": False,
                        "intent": query_spec.intent,
                        "strategies": context.retrieval_plan["retrieval_plan"]["strategies"],
                        "reasoning": "Planner LLM不可用，默认策略",
                        "fallback": True,
                    },
                    latency_ms=planner_result.latency_ms,
                    round=0,
                )
        if event_callback:
            event_callback.on_agent_result(
                "Planner",
                decision=planner_result.decision,
                detail={
                    "is_out_of_domain": context.retrieval_plan.get("is_out_of_domain"),
                    "domain_confidence": context.retrieval_plan.get("domain_confidence"),
                    "intent": context.retrieval_plan.get("intent"),
                    "complexity": context.retrieval_plan.get("complexity"),
                    "strategies": context.retrieval_plan.get("retrieval_plan", {}).get("strategies", []),
                    "sub_queries": context.retrieval_plan.get("sub_queries", []),
                    "reasoning": context.retrieval_plan.get("reasoning", ""),
                },
                latency_ms=planner_result.latency_ms,
                round=0,
            )
        logger.info(
            f"[Phase5][Planner] decision={planner_result.decision}, "
            f"out_of_domain={context.retrieval_plan.get('is_out_of_domain')}, "
            f"intent={context.retrieval_plan.get('intent')}, "
            f"strategies={context.retrieval_plan.get('retrieval_plan', {}).get('strategies')} "
            f"({planner_result.latency_ms}ms)"
        )
        # 推送规划思考过程
        if event_callback:
            _strategies = context.retrieval_plan.get('retrieval_plan', {}).get('strategies', [])
            _sub_queries = context.retrieval_plan.get('sub_queries', [])
            _thinking = f"问题类型: {context.retrieval_plan.get('intent', '?')}, 复杂度: {context.retrieval_plan.get('complexity', '?')}\n检索策略: {_strategies}\n子查询: {_sub_queries[:2]}"
            if context.retrieval_plan.get('reasoning'):
                _thinking += f"\n推理: {context.retrieval_plan['reasoning'][:120]}"
            event_callback.on_agent_thinking("Planner", _thinking, round=0)
        if trace_collector:
            trace_collector.record_planner(
                result=context.retrieval_plan,
                latency_ms=planner_result.latency_ms,
                decision=planner_result.decision,
            )

        # ── 越界检测：Planner 判定不属于本系统 → 直接拒答 ──
        # ⚠️ 必须在 PLANNING 状态下判断，PLANNING → RETRIEVING 是单向迁移，
        #    进入 RETRIEVING 后无法回到 REFUSING（状态机不允许）
        if context.retrieval_plan.get("is_out_of_domain"):
            sm.transition(AgentState.RETRIEVING, {
                "step": "retrieve_skipped",
                "reason": "out_of_domain",
            })
            sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                "step": "evidence_assemble_skipped",
                "reason": "out_of_domain",
            })
            sm.transition(AgentState.EVIDENCE_VALIDATING, {
                "step": "evidence_validate_skipped",
                "reason": "out_of_domain",
            })
            sm.transition(AgentState.REFUSING, {
                "step": "refuse",
                "reason": "领域越界（Planner判定）",
                "domain_confidence": context.retrieval_plan.get("domain_confidence", 0.5),
            })
            sm.transition(AgentState.RESPONDING, {"step": "respond"})
            logger.info("[Phase5] Planner 判定越界，生成拒答")
            answer = self._generator.generate(
                intent="out_of_domain",
                evidence_bundle=None,
                query_text=request.query,
            )
            response = self._build_response(
                request_id=request_id,
                session=session,
                sm=sm,
                query_spec=query_spec,
                route_decision=route_decision,
                answer=answer,
                start_time=start_time,
                agent_context=context,
            )
            if trace_collector:
                trace_collector.record_state("RESPONDING")
                self._finalize_and_save_trace(trace_collector, int(_now_ms() - start_time))
            if event_callback:
                event_callback.on_done(response.model_dump() if hasattr(response, 'model_dump') else response.__dict__)
            self._finalize(session, request, response, request_id)
            return response

        sm.transition(AgentState.RETRIEVING, {
            "step": "retrieve",
            "planner_decision": planner_result.decision,
            "strategies": context.retrieval_plan.get("retrieval_plan", {}).get("strategies", []),
        })

        # ── 构建初始过滤条件（复用 V1 的 _build_filters 逻辑）──
        # ⚠️ 使用去除选项后的 clean_query 重新提取实体，避免选项中的《文档名》
        #    覆盖问题主体的 doc_name
        if hasattr(rewritten, 'clean_query') and rewritten.clean_query and rewritten.clean_query.strip() != request.query.strip():
            filter_spec = self._query_spec_builder.build(
                rewritten.clean_query, session_id=session.session_id
            )
            base_filters = self._build_filters(filter_spec)
        else:
            base_filters = self._build_filters(query_spec)

        # ── 2. 检索-评估 Loop ──
        # 第一轮使用 search_query + base_filters
        # 后续轮次根据 retrieval_suggestion 调整 query/filters
        current_query = search_query
        current_filters = base_filters
        evidence_bundle = None
        accumulated_hits = []  # 跨轮次累积的检索结果（按chunk_id去重，避免覆盖）

        while True:
            # ── Retriever: 调用 V1 检索 + 证据组装 ──
            _step_t = _now_ms()
            logger.info(
                f"[Phase5][Retriever] loop={context.loop_count}, "
                f"query='{current_query[:60]}', filters={current_filters or '无'}"
            )
            if event_callback:
                event_callback.on_agent_start("Retriever", round=context.loop_count, detail={
                    "query": current_query,
                    "filters": current_filters or {},
                    "strategy": str(route_decision.channels),
                })
            if event_callback:
                event_callback.on_tool_call(
                    "Retriever", "search_regulatory_docs",
                    args={"query": current_query, "filters": current_filters or {}},
                    round=context.loop_count,
                )
            # ── Phase3: Redis工具结果缓存（相同query+filters取缓存, TTL 300s）──
            import hashlib as _hashlib
            _cache_key = f"rag:retrieval:{_hashlib.md5(f'{current_query}|{json.dumps(current_filters or {}, sort_keys=True)}|{route_decision.top_k}'.encode()).hexdigest()}"
            _cache_hit = False
            if self._redis:
                _cached_json = self._redis.get(_cache_key)
                if _cached_json:
                    try:
                        _cached_dict = json.loads(_cached_json)
                        retrieval_result = type(retrieval_result).__new__(type(retrieval_result))
                        retrieval_result.__dict__.update(_cached_dict.get("result", {}))
                        _cache_hit = True
                        logger.info(f"[Phase5][Retriever] 命中Redis缓存 → {retrieval_result.hit_count} hits")
                    except Exception:
                        _cache_hit = False
            if not _cache_hit:
                retrieval_result = self._retrieval_client.search_by_spec(
                    query_text=current_query,
                    route_decision=route_decision,
                    filters=current_filters,
                )
                # 写入缓存（仅缓存成功结果, TTL 300s）
                if self._redis and retrieval_result.success:
                    try:
                        self._redis.setex(_cache_key, 300, json.dumps({
                            "result": retrieval_result.to_dict() if hasattr(retrieval_result, 'to_dict') else {},
                            "query": current_query,
                            "filters": current_filters,
                        }))
                    except Exception as _cache_err:
                        logger.debug(f"[Phase5][Retriever] 缓存写入失败: {_cache_err}")
            retrieval_latency = int(_now_ms() - _step_t)
            logger.info(
                f"[Phase5][Retriever] 检索完成 → {retrieval_result.hit_count} hits, "
                f"{retrieval_result.latency_ms:.0f}ms, filters={current_filters or '无'}"
                f"{', 缓存命中' if _cache_hit else ''}"
            )
            if event_callback:
                event_callback.on_tool_result(
                    "Retriever", "search_regulatory_docs",
                    result={"hit_count": retrieval_result.hit_count, "latency_ms": retrieval_latency},
                    round=context.loop_count,
                )

            # 记录检索历史（供 Evaluator 参考）
            context.add_retrieval_record(
                query=current_query,
                hits=retrieval_result.hit_count,
                strategy=str(route_decision.channels),
                latency_ms=retrieval_latency,
            )
            if trace_collector:
                trace_collector.record_retrieval(
                    round_num=context.loop_count,
                    query=current_query,
                    hits=retrieval_result.hit_count,
                    latency_ms=retrieval_latency,
                    strategy=str(route_decision.channels),
                    filters=current_filters,
                )

            # ── 语义检索兜底（保留 V1 硬约束：doc_name 过滤不回退）──
            actual_filters = current_filters
            if retrieval_result.success and retrieval_result.hit_count == 0:
                if "doc_name" not in current_filters:
                    actual_filters = {}
                    logger.info(
                        f"[Phase5][Retriever] hits=0 无doc_name过滤，触发无过滤语义检索兜底"
                    )
                    _step_t = _now_ms()
                    retrieval_result = self._retrieval_client.search_by_spec(
                        query_text=current_query,
                        route_decision=route_decision,
                        filters={},
                    )
                    logger.info(
                        f"[Phase5][Retriever] 兜底完成 → {retrieval_result.hit_count} hits "
                        f"({_now_ms() - _step_t:.0f}ms)"
                    )
                else:
                    logger.warning(
                        f"[Phase5][Retriever] doc_name过滤0命中，不回退（避免从错误文档检索）"
                    )

            # ── 证据累积（跨轮次去重合并，不覆盖已有证据）──
            sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                "step": "evidence_assemble",
                "loop": context.loop_count,
                "hit_count": retrieval_result.hit_count,
                "retrieval_failed": not retrieval_result.success,
            })
            new_count = 0
            if not retrieval_result.success:
                logger.warning(
                    f"[Phase5][Retriever] 检索失败: {retrieval_result.error_code} - {retrieval_result.error}"
                )
            else:
                # 按 chunk_id 去重，将新检索结果追加到累积列表
                existing_ids = {h.get("chunk_id") for h in accumulated_hits if h.get("chunk_id")}
                for hit in retrieval_result.hits:
                    cid = hit.get("chunk_id")
                    if cid and cid not in existing_ids:
                        accumulated_hits.append(hit)
                        existing_ids.add(cid)
                        new_count += 1
                    elif cid:
                        # chunk_id 已存在：保留得分更高的
                        for i, h in enumerate(accumulated_hits):
                            if h.get("chunk_id") == cid:
                                if hit.get("score", 0) > h.get("score", 0):
                                    accumulated_hits[i] = hit
                                break
                logger.info(
                    f"[Phase5][Retriever] 证据累积 → 本轮新增{new_count}条, "
                    f"累积{len(accumulated_hits)}条 (loop={context.loop_count})"
                )
            # 始终从累积的全量证据构建 evidence_bundle（不覆盖）
            evidence_bundle = self._evidence_builder.build(
                hits=accumulated_hits,
                claims=query_spec.claims,
                query_text=request.query,
                retrieval_filters=actual_filters,
            )
            context.evidence_bundle = evidence_bundle

            # ── 充分性快速判断（V1 规则评分充分 → 直接退出，跳过 Evaluator LLM）──
            # 速度优化：SufficiencyScorer 五维规则评分达标即视为充分，
            # 不再调用 Evaluator LLM（该调用可能长达 45s+，导致前端长时间无响应）
            if evidence_bundle.is_sufficient:
                logger.info(
                    f"[Phase5] V1规则评分充分 (score={evidence_bundle.sufficiency_score:.3f})，"
                    f"跳过 Evaluator LLM，退出Loop (loop={context.loop_count})"
                )
                context.evaluation_result = {
                    "is_sufficient": True,
                    "sufficiency_score": evidence_bundle.sufficiency_score,
                    "skip_llm": True,
                }
                _loop_snip_ok = []
                if evidence_bundle.evidence_items:
                    _loop_snip_ok = [
                        (e.evidence_snippet or e.content or "")[:60]
                        for e in evidence_bundle.evidence_items[:3]
                    ]
                if event_callback:
                    event_callback.on_loop_round(
                        round=context.loop_count,
                        query=current_query,
                        hits=retrieval_result.hit_count,
                        score=evidence_bundle.sufficiency_score,
                        strategy=str(route_decision.channels),
                        latency_ms=retrieval_latency,
                        snippets=_loop_snip_ok,
                    )
                v1_sufficient = True
                agent_sufficient = True
                is_sufficient = True
                break

            # ── EVIDENCE_VALIDATING: EvaluatorAgent 评估（仅 V1 规则不足时）──
            sm.transition(AgentState.EVIDENCE_VALIDATING, {
                "step": "evidence_validate",
                "loop": context.loop_count,
                "sufficiency": evidence_bundle.sufficiency_score,
            })
            if event_callback:
                event_callback.on_agent_start("Evaluator", round=context.loop_count, detail={
                    "sufficiency_score": evidence_bundle.sufficiency_score,
                    "evidence_count": evidence_bundle.evidence_count if hasattr(evidence_bundle, 'evidence_count') else 0,
                })
            _step_t = _now_ms()
            try:
                eval_result = self._evaluator_agent.run(context)
            except Exception as eval_err:
                # Evaluator LLM 失败/超时不阻断流程：降级判定继续
                # 弱充分降级: LLM 全链路不可用时，只要有证据即视为充分
                # （Generator 侧有规则计算兜底，可脱离 LLM 正确回答取数计算题）
                weak_sufficient = False
                if evidence_bundle.evidence_count > 0:
                    weak_sufficient = True
                    logger.warning(
                        f"[Phase5][Evaluator] LLM 调用失败，证据非空→弱充分降级直接生成: {eval_err}"
                    )
                else:
                    logger.warning(
                        f"[Phase5][Evaluator] LLM 调用失败且无证据，降级用 V1 判定继续: {eval_err}"
                    )
                context.evaluation_result = {
                    "is_sufficient": weak_sufficient,
                    "sufficiency_score": evidence_bundle.sufficiency_score,
                    "skipped": True,
                    "weak_fallback": weak_sufficient,
                    "error": str(eval_err),
                }
                eval_result = None
            if event_callback and eval_result is not None:
                event_callback.on_agent_result(
                    "Evaluator",
                    decision=eval_result.decision,
                    detail={
                        "is_sufficient": context.evaluation_result.get("is_sufficient"),
                        "sufficiency_score": context.evaluation_result.get("sufficiency_score"),
                        "dimensions": context.evaluation_result.get("dimensions", {}),
                        "missing_claims": context.evaluation_result.get("missing_claims", []),
                        "retrieval_suggestion": context.retrieval_suggestion,
                    },
                    latency_ms=eval_result.latency_ms if eval_result else 0,
                    round=context.loop_count,
                )
            logger.info(
                f"[Phase5][Evaluator] loop={context.loop_count}, "
                f"decision={eval_result.decision if eval_result else 'skipped'}, "
                f"sufficient={context.evaluation_result.get('is_sufficient')}, "
                f"score={context.evaluation_result.get('sufficiency_score')} "
                f"({eval_result.latency_ms if eval_result else 0}ms)"
            )
            # 推送评估思考过程
            if event_callback:
                _missing = context.evaluation_result.get('missing_claims', [])
                _score = context.evaluation_result.get('sufficiency_score', 0)
                _suff = context.evaluation_result.get('is_sufficient', False)
                _thinking = f"证据评分: {_score:.3f}, 充分: {_suff}\n命中证据: {evidence_bundle.evidence_count}条"
                if _missing:
                    _thinking += f"\n缺失: {_missing[:3]}"
                if context.retrieval_suggestion:
                    _sug = context.retrieval_suggestion
                    _thinking += f"\n检索建议: {_sug.get('direction', '')}, query={_sug.get('suggested_query', '')[:60]}"
                event_callback.on_agent_thinking("Evaluator", _thinking, round=context.loop_count)
            if trace_collector:
                trace_collector.record_evaluation(
                    round_num=context.loop_count,
                    score=evidence_bundle.sufficiency_score,
                    dimensions=context.evaluation_result.get("dimensions", {}),
                    suggestion=context.retrieval_suggestion,
                    latency_ms=eval_result.latency_ms if eval_result else 0,
                    is_sufficient=context.evaluation_result.get("is_sufficient"),
                )

            # ── Loop 轮次事件（供前端可视化，含检索材料摘要）──
            _loop_snippets = []
            if evidence_bundle and evidence_bundle.evidence_items:
                _loop_snippets = [
                    (e.evidence_snippet or e.content or "")[:60]
                    for e in evidence_bundle.evidence_items[:3]
                ]
            if event_callback:
                event_callback.on_loop_round(
                    round=context.loop_count,
                    query=current_query,
                    hits=retrieval_result.hit_count,
                    score=evidence_bundle.sufficiency_score,
                    strategy=str(route_decision.channels),
                    latency_ms=retrieval_latency,
                    snippets=_loop_snippets,
                )

            # ── 充分性判断（V1 规则评分 + Evaluator LLM 综合判断）──
            # 双重判断逻辑：任一充分即退出 Loop（避免过度消耗预算）
            #   - v1_sufficient: SufficiencyScorer 五维度规则评分
            #   - agent_sufficient: Evaluator LLM 综合判断（可 override V1 的保守判定）
            v1_sufficient = evidence_bundle.is_sufficient
            agent_sufficient = context.evaluation_result.get("is_sufficient", v1_sufficient)
            is_sufficient = v1_sufficient or agent_sufficient

            if is_sufficient:
                logger.info(
                    f"[Phase5] 证据充分 (v1={v1_sufficient}, agent={agent_sufficient}, "
                    f"score={evidence_bundle.sufficiency_score:.3f})，退出Loop"
                )
                break

            # ── 连续空轮次保护：连续≥2轮无新证据 → 停止无意义重试 ──
            if context.loop_count >= 2 and new_count == 0:
                logger.info(
                    f"[Phase5] 本轮无新证据(new_count=0, loop={context.loop_count})，"
                    f"停止无意义重试，进入生成"
                )
                break

            # ── 不充分：检查预算，决定是否继续 Loop ──
            if not context.can_continue_loop():
                logger.info(
                    f"[Phase5] 预算耗尽，退出Loop (loop={context.loop_count}, "
                    f"score={evidence_bundle.sufficiency_score:.3f})"
                )
                break

            action = context.increment_loop()
            if action == "stop":
                logger.info(f"[Phase5] 预算STOP，退出Loop (loop={context.loop_count})")
                break

            # ── 根据 retrieval_suggestion 调整下一轮检索策略 ──
            suggestion = context.retrieval_suggestion or {}
            suggested_query = suggestion.get("suggested_query", "")
            suggested_strategy = suggestion.get("suggested_strategy", "")
            reason = suggestion.get("reason", "")

            logger.info(
                f"[Phase5] 证据不足，Loop继续 (loop={context.loop_count}, "
                f"action={action}, suggested_strategy={suggested_strategy}, reason={reason[:80]})"
            )

            # ── Layer 2: 参数槽位驱动补检索（不依赖 Evaluator 建议质量）──
            # 优先用未填充 claim 的 description 构造针对性 query，精准补检索缺失参数
            unfilled_claims = [
                cs for cs in evidence_bundle.claim_slots
                if cs.status not in ("supported",)
            ]
            if unfilled_claims:
                slot_query = " ".join(
                    cs.description for cs in unfilled_claims[:3] if cs.description
                ).strip()
                if slot_query:
                    current_query = slot_query[:200]
                    logger.info(
                        f"[Phase5][Layer2] {len(unfilled_claims)}个槽位未填充, "
                        f"针对性补检索 query='{current_query[:80]}'"
                    )
                elif suggested_query and suggested_query.strip():
                    current_query = suggested_query.strip()
            elif suggested_query and suggested_query.strip():
                current_query = suggested_query.strip()

            # ── Layer 3: 多策略兜底 — 宽松 filters（仅保留 doc_name 文档安全锁）──
            # 第2轮+证据仍不足时，去掉 table_name 等过严过滤，扩大召回避免空结果
            if context.loop_count >= 2 and base_filters:
                relaxed_filters = {}
                if "doc_name" in base_filters:
                    relaxed_filters["doc_name"] = base_filters["doc_name"]
                current_filters = relaxed_filters
                logger.info(
                    f"[Phase5][Layer3] 多策略兜底: 宽松filters={current_filters} "
                    f"(保留doc_name锁, loop={context.loop_count})"
                )
            else:
                # ⚠️ 硬约束：doc_name 过滤不可被 Evaluator 建议移除
                current_filters = base_filters

            # 状态迁移：EVIDENCE_VALIDATING → RETRIEVING（合法迁移）
            sm.transition(AgentState.RETRIEVING, {
                "step": "retrieve_retry",
                "loop": context.loop_count,
                "suggested_strategy": suggested_strategy,
            })

        # ── Loop 结束后的最终充分性判断 ──
        # 任一充分即可生成回答（V1 规则评分 或 Evaluator LLM 判断）
        final_sufficient = (
            evidence_bundle.is_sufficient
            or context.evaluation_result.get("is_sufficient", False)
        )
        if not final_sufficient:
            # 证据不足 → 检查歧义后澄清或拒答（保留 V1 先检索后澄清策略）
            if query_spec.ambiguities:
                sm.transition(AgentState.REFUSING, {
                    "step": "refuse",
                    "reason": "证据不足+歧义，请求澄清",
                    "ambiguities": len(query_spec.ambiguities),
                    "loop_count": context.loop_count,
                })
                sm.transition(AgentState.RESPONDING, {"step": "clarify"})
                logger.info(
                    f"[Phase5] 证据不足且存在歧义 → 生成澄清请求 "
                    f"(loop={context.loop_count}, ambiguities={len(query_spec.ambiguities)})"
                )
                answer = self._generator.generate_clarification(query_spec.ambiguities)
            else:
                sm.transition(AgentState.REFUSING, {
                    "step": "refuse",
                    "reason": "证据不足",
                    "loop_count": context.loop_count,
                })
                sm.transition(AgentState.RESPONDING, {"step": "respond"})
                # Layer 4: 明确记录缺失参数，便于拒答时精准报告缺失内容
                unfilled_final = [
                    cs.description for cs in evidence_bundle.claim_slots
                    if cs.status not in ("supported",) and cs.description
                ]
                logger.info(
                    f"[Phase5] 证据不足 → 生成拒答 (loop={context.loop_count}, "
                    f"score={evidence_bundle.sufficiency_score:.3f}, "
                    f"缺失参数={unfilled_final})"
                )
                answer = self._generator.generate(
                    intent=query_spec.intent,
                    evidence_bundle=evidence_bundle,
                    query_text=rewritten.clean_query or request.query,
                    options=rewritten.options,
                    prompt_mode=rewritten.prompt_mode,
                )

            response = self._build_response(
                request_id=request_id,
                session=session,
                sm=sm,
                query_spec=query_spec,
                route_decision=route_decision,
                answer=answer,
                evidence_bundle=evidence_bundle,
                start_time=start_time,
                agent_context=context,
            )
            if trace_collector:
                trace_collector.record_state("RESPONDING")
                trace_collector.record_loop_count(context.loop_count)
                self._finalize_and_save_trace(trace_collector, int(_now_ms() - start_time))
            if event_callback:
                event_callback.on_done(response.model_dump() if hasattr(response, 'model_dump') else response.__dict__)
            self._finalize(session, request, response, request_id)
            return response

        # ── Agent判定充分时，覆盖evidence_bundle的V1评分标志 ──
        # 确保 Generator 不会因 V1 规则评分不足而拒绝生成
        if not evidence_bundle.is_sufficient:
            evidence_bundle.is_sufficient = True
            logger.info(
                "[Phase5] Agent判定充分，覆盖evidence_bundle.is_sufficient=True "
                f"(v1_score={evidence_bundle.sufficiency_score:.3f})"
            )

        # ── 3. GENERATING: 回答生成 ──
        sm.transition(AgentState.GENERATING, {"step": "generate"})
        _step_t = _now_ms()
        logger.info(
            f"[Phase5] 回答生成中 → intent={query_spec.intent}, "
            f"evidence_count={evidence_bundle.evidence_count}, loop={context.loop_count}"
        )
        if event_callback:
            event_callback.on_agent_start("Generator", round=0, detail={
                "intent": query_spec.intent,
                "evidence_count": evidence_bundle.evidence_count if hasattr(evidence_bundle, 'evidence_count') else 0,
                "loop_count": context.loop_count,
            })
        # ── 流式生成：有 event_callback 时逐 token 推送，失败回退非流式 ──
        answer = None
        if event_callback:
            try:
                gen = self._generator.generate_stream(
                    intent=query_spec.intent,
                    evidence_bundle=evidence_bundle,
                    query_text=rewritten.clean_query or request.query,
                    options=rewritten.options,
                    prompt_mode=rewritten.prompt_mode,
                    on_thinking=lambda text: event_callback.on_agent_thinking("Generator", text, round=context.loop_count) if event_callback else None,
                )
                try:
                    while True:
                        token = next(gen)
                        event_callback.on_answer_token(token)
                except StopIteration as e:
                    answer = e.value
                if answer is None:
                    raise RuntimeError("generate_stream 未返回 GeneratedAnswer")
                logger.info("[Phase5] 流式生成完成")
            except Exception as stream_err:
                logger.warning(
                    f"[Phase5] 流式生成失败，回退非流式: {stream_err}"
                )
                answer = self._generator.generate(
                    intent=query_spec.intent,
                    evidence_bundle=evidence_bundle,
                    query_text=rewritten.clean_query or request.query,
                    ambiguities=query_spec.ambiguities,
                    options=rewritten.options,
                    prompt_mode=rewritten.prompt_mode,
                )
        else:
            answer = self._generator.generate(
                intent=query_spec.intent,
                evidence_bundle=evidence_bundle,
                query_text=rewritten.clean_query or request.query,
                ambiguities=query_spec.ambiguities,
                options=rewritten.options,
                prompt_mode=rewritten.prompt_mode,
            )
        gen_latency = int(_now_ms() - _step_t)
        if event_callback:
            event_callback.on_agent_result(
                "Generator",
                decision="generated",
                latency_ms=gen_latency,
                round=context.loop_count,
            )
        logger.info(f"[Phase5] 回答生成完成 ({gen_latency}ms)")
        if trace_collector:
            # confidence 从 GeneratedAnswer 提取（若可用）
            _conf = getattr(answer, "confidence", None)
            trace_collector.record_generation(
                model=getattr(self._generator, "model_name", "unknown"),
                tokens=getattr(answer, "tokens", {}) or {},
                latency_ms=gen_latency,
                confidence=_conf,
            )

        # ── 4. ANSWER_VALIDATING: VerifierAgent 声明级验证 ──
        # 将生成的回答写入 context 供 Verifier 读取
        # GroundedGenerator 返回 GeneratedAnswer 对象，直接赋值
        context.generated_answer = answer

        # Verifier 补充检索预算（独立于检索Loop，受总预算约束）
        verifier_retry_done = False
        max_verifier_retries = 1  # Verifier 只允许一次补充检索，避免无限循环

        # 使用 while True + break 控制，确保补充检索后重新验证
        # （状态机要求 GENERATING → ANSWER_VALIDATING → RESPONDING 合法路径）
        while True:
            # ── Verifier 时间闸：总耗时超限时跳过验证（LLM慢时避免再等45s）──
            # 验证是锦上添花，不阻塞回答输出（回答已验证过证据充分）
            if _now_ms() - start_time > 65000:
                logger.warning(
                    f"[Phase5][Verifier] 总耗时已超65s，跳过验证直接回复 "
                    f"(elapsed={int(_now_ms() - start_time) / 1000:.1f}s)"
                )
                context.verification_result = {
                    "verified": False,
                    "needs_retry": False,
                    "unverified_count": 0,
                    "claims": [],
                    "skipped": True,
                    "reason": "time_gate",
                }
                # 状态机要求 GENERATING → ANSWER_VALIDATING → RESPONDING
                sm.transition(AgentState.ANSWER_VALIDATING, {
                    "step": "answer_validate_skip",
                    "agent": "Verifier",
                    "reason": "time_gate",
                })
                break
            sm.transition(AgentState.ANSWER_VALIDATING, {
                "step": "answer_validate",
                "agent": "Verifier",
                "verifier_retry": verifier_retry_done,
            })
            if event_callback:
                event_callback.on_agent_start("Verifier", round=verifier_retry_done, detail={
                    "answer_length": len(answer.answer_text) if hasattr(answer, 'answer_text') else 0,
                    "evidence_count": evidence_bundle.evidence_count if hasattr(evidence_bundle, 'evidence_count') else 0,
                    "is_retry": verifier_retry_done,
                })
            _step_t = _now_ms()
            try:
                verify_result = self._verifier_agent.run(context)
            except Exception as verifier_err:
                # Verifier 失败（如 LLM 超时）不阻断流程：跳过验证，保留已生成的回答
                logger.warning(
                    f"[Phase5][Verifier] 验证失败，跳过验证继续回复: {verifier_err}",
                    exc_info=True,
                )
                context.verification_result = {
                    "verified": False,
                    "needs_retry": False,
                    "unverified_count": 0,
                    "claims": [],
                    "skipped": True,
                    "error": str(verifier_err),
                }
                if event_callback:
                    event_callback.on_error(
                        "Verifier",
                        f"验证超时或失败，已跳过（回答将直接返回）: {verifier_err}",
                        round=verifier_retry_done,
                    )
                break
            logger.info(
                f"[Phase5][Verifier] decision={verify_result.decision}, "
                f"verified={context.verification_result.get('verified')}, "
                f"needs_retry={context.verification_result.get('needs_retry')}, "
                f"unverified_count={context.verification_result.get('unverified_count')} "
                f"({verify_result.latency_ms}ms)"
            )
            if event_callback:
                vres = context.verification_result
                claims = vres.get("claims", [])
                verified_cnt = sum(1 for c in claims if isinstance(c, dict) and c.get("status") == "verified")
                event_callback.on_agent_result(
                    "Verifier",
                    decision=verify_result.decision,
                    detail={
                        "verified": vres.get("verified"),
                        "needs_retry": vres.get("needs_retry"),
                        "verified_count": verified_cnt,
                        "unverified_count": vres.get("unverified_count", 0),
                        "claims": claims,
                        "retry_query": vres.get("retry_query", ""),
                    },
                    latency_ms=verify_result.latency_ms,
                    round=verifier_retry_done,
                )
            if trace_collector:
                vres = context.verification_result
                claims = vres.get("claims", [])
                verified_cnt = sum(1 for c in claims if isinstance(c, dict) and c.get("status") == "verified")
                trace_collector.record_verification(
                    verified_count=verified_cnt,
                    unverified_count=vres.get("unverified_count", 0),
                    needs_retry=vres.get("needs_retry", False),
                    claims=claims,
                    latency_ms=verify_result.latency_ms,
                )

            vres = context.verification_result
            if vres.get("verified") and not vres.get("needs_retry"):
                # 验证通过 → 直接回复
                logger.info("[Phase5] Verifier验证通过，进入回复")
                break

            if not vres.get("needs_retry"):
                # partial_verified（部分验证）但不要求重试 → 直接回复
                logger.info("[Phase5] Verifier部分验证，不重试，进入回复")
                break

            # needs_retry=True：检查是否可补充检索
            if verifier_retry_done:
                logger.info("[Phase5] Verifier补充检索已执行过，不再重试，直接回复")
                break

            if not context.can_continue_loop():
                logger.info("[Phase5] Verifier要求重试但预算耗尽，直接回复")
                break

            retry_query = vres.get("retry_query", "").strip()
            if not retry_query:
                logger.info("[Phase5] Verifier未提供retry_query，跳过补充检索")
                break

            # ── 触发补充检索：用 retry_query 重新检索+生成 ──
            context.increment_loop()
            verifier_retry_done = True
            logger.info(
                f"[Phase5] Verifier触发补充检索 (loop={context.loop_count}, "
                f"retry_query='{retry_query[:60]}')"
            )

            # 状态迁移：ANSWER_VALIDATING → RETRYING → RETRIEVING（合法路径）
            sm.transition(AgentState.RETRYING, {
                "step": "verifier_retry",
                "retry_query": retry_query,
            })
            sm.transition(AgentState.RETRIEVING, {
                "step": "retrieve_verifier_retry",
                "loop": context.loop_count,
            })

            # 补充检索（保留 doc_name 文档安全锁）
            retry_filters = base_filters
            _step_t = _now_ms()
            retry_result = self._retrieval_client.search_by_spec(
                query_text=retry_query,
                route_decision=route_decision,
                filters=retry_filters,
            )
            context.add_retrieval_record(
                query=retry_query,
                hits=retry_result.hit_count,
                strategy="verifier_retry",
                latency_ms=int(_now_ms() - _step_t),
            )

            # 合并证据：用新检索结果重建证据包
            sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                "step": "evidence_assemble_retry",
                "hit_count": retry_result.hit_count,
            })
            if retry_result.success and retry_result.hit_count > 0:
                # 合并新旧 hits（去重靠 evidence_builder 内部处理）
                merged_hits = list(retrieval_result.hits) + list(retry_result.hits)
                evidence_bundle = self._evidence_builder.build(
                    hits=merged_hits,
                    claims=query_spec.claims,
                    query_text=request.query,
                    retrieval_filters=retry_filters,
                )
                context.evidence_bundle = evidence_bundle

                # 状态机合法路径：EVIDENCE_ASSEMBLING → EVIDENCE_VALIDATING → GENERATING
                sm.transition(AgentState.EVIDENCE_VALIDATING, {
                    "step": "evidence_validate_retry",
                    "sufficiency": evidence_bundle.sufficiency_score,
                })
                # 重新生成回答（确保 is_sufficient 被覆盖，避免 Generator 拒答）
                if not evidence_bundle.is_sufficient:
                    evidence_bundle.is_sufficient = True
                    logger.info(
                        "[Phase5] Verifier重试: 覆盖evidence_bundle.is_sufficient=True"
                    )
                sm.transition(AgentState.GENERATING, {"step": "regenerate"})
                _step_t = _now_ms()
                answer = self._generator.generate(
                    intent=query_spec.intent,
                    evidence_bundle=evidence_bundle,
                    query_text=rewritten.clean_query or request.query,
                    ambiguities=query_spec.ambiguities,
                    options=rewritten.options,
                    prompt_mode=rewritten.prompt_mode,
                )
                context.generated_answer = answer
                logger.info(f"[Phase5] 补充检索后重新生成完成 ({_now_ms() - _step_t:.0f}ms)")
                # 重新生成后继续循环 → 回到 ANSWER_VALIDATING 做最终验证
                # （状态机合法：GENERATING → ANSWER_VALIDATING）
            else:
                logger.info("[Phase5] 补充检索无新证据，保留原回答")
                # 无新证据时，状态停在 EVIDENCE_ASSEMBLING，
                # 需要走到 ANSWER_VALIDATING 才能 RESPONDING
                sm.transition(AgentState.EVIDENCE_VALIDATING, {
                    "step": "evidence_validate_no_new",
                    "sufficiency": evidence_bundle.sufficiency_score,
                })
                sm.transition(AgentState.GENERATING, {"step": "keep_answer"})
                # 继续循环 → ANSWER_VALIDATING → RESPONDING
                break

        # ── 5. RESPONDING ──
        # 状态机合法路径：ANSWER_VALIDATING → RESPONDING
        sm.transition(AgentState.RESPONDING, {"step": "respond"})
        if trace_collector:
            trace_collector.record_state("RESPONDING")
            trace_collector.record_loop_count(context.loop_count)

        response = self._build_response(
            request_id=request_id,
            session=session,
            sm=sm,
            query_spec=query_spec,
            route_decision=route_decision,
            answer=answer,
            evidence_bundle=evidence_bundle,
            start_time=start_time,
            agent_context=context,
        )
        # Phase 6: finalize trace + 持久化（容错，失败不影响主流程）
        if trace_collector:
            self._finalize_and_save_trace(trace_collector, int(_now_ms() - start_time))
        if event_callback:
            event_callback.on_done(response.model_dump() if hasattr(response, 'model_dump') else response.__dict__)
        self._finalize(session, request, response, request_id)
        return response

    # ──────────────────────────────────────────────────────
    # V1 线性流程（fallback / enable_agent_loop=False）
    # ──────────────────────────────────────────────────────

    def _run_v1_linear_flow(
        self,
        request: QueryRequest,
        request_id: str,
        session: Any,
        sm: StateMachine,
        start_time: float,
        query_spec: QuerySpec,
        route_decision: Any,
        rewritten: Any,
        search_query: str,
    ) -> QueryResponse:
        """
        V1 线性检索流程（原 Phase 2 实现，保持向后兼容）

        流程: 检索 → 证据组装 → 验证 → 生成/拒答 → 回复
        无 Loop，无 Agent 协作。
        """
        sm.transition(AgentState.RETRIEVING, {
            "step": "retrieve",
            "strategy": route_decision.channels,
        })

        # 构建检索过滤条件
        if hasattr(rewritten, 'clean_query') and rewritten.clean_query and rewritten.clean_query.strip() != request.query.strip():
            filter_spec = self._query_spec_builder.build(
                rewritten.clean_query, session_id=session.session_id
            )
            filters = self._build_filters(filter_spec)
            logger.info(f"[步骤] 过滤条件构建（from clean_query）→ filters={filters or '无'}")
        else:
            filters = self._build_filters(query_spec)
        actual_filters = filters

        # 调用检索
        _step_t = _now_ms()
        logger.info(f"[步骤] 检索中 → query='{search_query[:60]}', filters={filters or '无'}")
        retrieval_result = self._retrieval_client.search_by_spec(
            query_text=search_query,
            route_decision=route_decision,
            filters=filters,
        )
        logger.info(f"[步骤] 检索完成 → {retrieval_result.hit_count} hits, {retrieval_result.latency_ms:.0f}ms, filters={filters or '无'}")

        # ── 语义检索兜底（V1 硬约束：doc_name 过滤不回退）──
        if retrieval_result.success and retrieval_result.hit_count == 0:
            if "doc_name" not in filters:
                actual_filters = {}
                logger.info(
                    f"[步骤] 首次检索证据不足 (hits=0, filters={filters or '无'})，触发无过滤语义检索兜底"
                )
                _step_t = _now_ms()
                retrieval_result = self._retrieval_client.search_by_spec(
                    query_text=search_query,
                    route_decision=route_decision,
                    filters={},
                )
                logger.info(
                    f"[步骤] 无过滤语义检索兜底完成 → {retrieval_result.hit_count} hits, "
                    f"{retrieval_result.latency_ms:.0f}ms (总 {_now_ms() - _step_t:.0f}ms)"
                )
            else:
                logger.warning(
                    f"[步骤] doc_name 过滤无命中 (filters={filters})，"
                    f"文档可能不存在于知识库，不执行无过滤回退（避免从错误文档检索）"
                )

        # 检索失败处理
        if not retrieval_result.success:
            logger.warning(f"检索失败: {retrieval_result.error_code} - {retrieval_result.error}")
            sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                "step": "evidence_assemble",
                "retrieval_failed": True,
            })
            evidence_bundle = self._evidence_builder.build(
                hits=[], claims=query_spec.claims, query_text=request.query,
                retrieval_filters=actual_filters,
            )
        else:
            sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                "step": "evidence_assemble",
                "hit_count": retrieval_result.hit_count,
            })
            evidence_bundle = self._evidence_builder.build(
                hits=retrieval_result.hits,
                claims=query_spec.claims,
                query_text=request.query,
                retrieval_filters=actual_filters,
            )

        # ── 证据验证 ──
        logger.info(f"[步骤] 证据组装 → 充分性={evidence_bundle.sufficiency_score:.3f}, sufficient={evidence_bundle.is_sufficient}")
        sm.transition(AgentState.EVIDENCE_VALIDATING, {
            "step": "evidence_validate",
            "sufficiency": evidence_bundle.sufficiency_score,
        })

        # 证据不足 → 检查歧义后澄清或拒答（先检索后澄清策略）
        if not evidence_bundle.is_sufficient:
            if query_spec.ambiguities:
                sm.transition(AgentState.REFUSING, {
                    "step": "refuse",
                    "reason": "证据不足+歧义，请求澄清",
                    "ambiguities": len(query_spec.ambiguities),
                })
                sm.transition(AgentState.RESPONDING, {"step": "clarify"})
                logger.info(f"[步骤] 证据不足且存在歧义 → 生成澄清请求 (ambiguities={len(query_spec.ambiguities)})")
                answer = self._generator.generate_clarification(query_spec.ambiguities)
            else:
                sm.transition(AgentState.REFUSING, {
                    "step": "refuse",
                    "reason": "证据不足",
                })
                sm.transition(AgentState.RESPONDING, {"step": "respond"})
                logger.info("[步骤] 回答生成 → 证据不足，生成拒答")
                answer = self._generator.generate(
                    intent=query_spec.intent,
                    evidence_bundle=evidence_bundle,
                    query_text=rewritten.clean_query or request.query,
                    options=rewritten.options,
                    prompt_mode=rewritten.prompt_mode,
                )

            response = self._build_response(
                request_id=request_id,
                session=session,
                sm=sm,
                query_spec=query_spec,
                route_decision=route_decision,
                answer=answer,
                evidence_bundle=evidence_bundle,
                start_time=start_time,
            )
            self._finalize(session, request, response, request_id)
            return response

        # ── 回答生成 ──
        sm.transition(AgentState.GENERATING, {"step": "generate"})
        _step_t = _now_ms()
        logger.info(f"[步骤] 回答生成中 → intent={query_spec.intent}, evidence_count={evidence_bundle.evidence_count}")
        answer = self._generator.generate(
            intent=query_spec.intent,
            evidence_bundle=evidence_bundle,
            query_text=rewritten.clean_query or request.query,
            ambiguities=query_spec.ambiguities,
            options=rewritten.options,
            prompt_mode=rewritten.prompt_mode,
        )
        logger.info(f"[步骤] 回答生成完成 ({_now_ms() - _step_t:.0f}ms)")

        # ── 回答验证 ──
        sm.transition(AgentState.ANSWER_VALIDATING, {"step": "answer_validate"})
        sm.transition(AgentState.RESPONDING, {"step": "respond"})

        response = self._build_response(
            request_id=request_id,
            session=session,
            sm=sm,
            query_spec=query_spec,
            route_decision=route_decision,
            answer=answer,
            evidence_bundle=evidence_bundle,
            start_time=start_time,
        )
        self._finalize(session, request, response, request_id)
        return response

    def _build_session_context(self, session: Any) -> Any:
        """从会话历史构建查询改写所需的上下文"""
        try:
            from agent_platform.query_understanding.query_rewriter import SessionContext

            # 从会话历史提取信息
            previous_queries = []
            mentioned_metrics = []
            mentioned_docs = []
            previous_entities = []

            # 兼容不同会话对象接口
            turns = getattr(session, "turns", [])
            for turn in turns[-5:]:  # 最近5轮
                query = turn.get("query", "") if isinstance(turn, dict) else getattr(turn, "query", "")
                if query:
                    previous_queries.append(query)

                # 从 turn metadata 中提取实体
                metadata = turn.get("metadata", {}) if isinstance(turn, dict) else getattr(turn, "metadata", {})
                entities = metadata.get("entities", []) if isinstance(metadata, dict) else []
                for ent in entities:
                    previous_entities.append(ent)
                    etype = ent.get("entity_type", "")
                    value = ent.get("value", "")
                    if etype == "metric_name" and value:
                        mentioned_metrics.append(value)
                    elif etype == "doc_name" and value:
                        mentioned_docs.append(value)

            return SessionContext(
                previous_queries=previous_queries,
                previous_entities=previous_entities,
                mentioned_metrics=mentioned_metrics,
                mentioned_docs=mentioned_docs,
            )
        except Exception:
            return None

    def _build_filters(self, query_spec: QuerySpec) -> dict:
        """从 QuerySpec 构建检索过滤条件

        注意：对于 table_lookup 意图，"applicable_scope"（如"大型商业银行"）
        是表格内行标签而非文档级元数据，不应作为 metadata 过滤条件。
        metadata 过滤仅用于限定文档来源，表格内的行/列值应通过 table 检索器匹配。
        """
        filters = {}
        constraints = query_spec.constraints

        # table_lookup 意图：applicable_scope 是表格行数据，不作为 metadata 过滤
        if query_spec.intent != "table_lookup":
            if constraints.get("applicable_scope"):
                filters["applicable_scope"] = constraints["applicable_scope"]

        # 收集搜索关键词（用于 table_lookup 的 pattern）
        search_terms = []

        # 从实体中提取过滤条件
        # ⚠️ doc_name 只取第一个（问题主体中的），避免被后续《》覆盖
        doc_name_set = False
        for entity in query_spec.entities:
            etype = entity.get("entity_type")
            value = entity.get("value")
            if etype == "doc_name" and value:
                if not doc_name_set:
                    filters["doc_name"] = value
                    doc_name_set = True
                    # table_lookup 意图：doc_name 同时也映射为 table_name（用于 table 检索器）
                    if query_spec.intent == "table_lookup":
                        filters["table_name"] = value
            elif etype == "clause_number" and value:
                filters["clause_number"] = f"第{value}条"
            elif etype == "chapter_number" and value:
                filters["chapter_number"] = f"第{value}章"
            elif etype == "table_name" and value:
                filters["table_name"] = value
            elif etype == "attachment_no" and value:
                filters["attachment_no"] = f"附件{value}"
            elif etype == "metric_name" and value:
                search_terms.append(value)
            elif etype == "scope" and value:
                search_terms.append(value)

        # table_lookup 意图：构建精简搜索 pattern
        # 交叉表结构：行=指标名，列=机构类型。
        # 优先从引号中提取目标指标名（用户明确询问的），
        # 避免从文档名《》中连带提取的非目标指标污染搜索词。
        if query_spec.intent == "table_lookup":
            import re
            quoted = re.findall(
                r'[“”「」\"]([^“”「」\"]+)[“”「」\"]',
                query_spec.raw_query
            )
            # 排除口径/时间修饰词，剩余的作为目标指标名
            scope_patterns = ["截至", "累计", "当期", "同比", "环比", "账面余额", "规模占比", "年-季度", "口径"]
            quoted_metrics = []
            if quoted:
                seen = set()
                for q in quoted:
                    if q not in seen and not any(p in q for p in scope_patterns):
                        seen.add(q)
                        quoted_metrics.append(q)

            if quoted_metrics:
                # 优先：引号内指标名 = 用户明确询问的目标
                filters["pattern"] = quoted_metrics[0]
                logger.info(f"[步骤] table_lookup 从引号提取搜索词: '{filters['pattern']}'")
            else:
                # 回退：从实体中取指标名
                metric_terms = [
                    e.get("value", "")
                    for e in query_spec.entities
                    if e.get("entity_type") == "metric_name"
                ]
                if metric_terms:
                    filters["pattern"] = " ".join(metric_terms)
                elif search_terms:
                    filters["pattern"] = " ".join(search_terms)

        return filters

    def _build_response(
        self,
        request_id: str,
        session: Any,
        sm: StateMachine,
        query_spec: QuerySpec,
        route_decision: Any,
        answer: Any,
        evidence_bundle: Optional[Any] = None,
        start_time: float = 0.0,
        agent_context: Optional[AgentContext] = None,
    ) -> QueryResponse:
        """构建响应"""
        latency_ms = _now_ms() - start_time

        # Phase 5/6: 从 AgentContext 提取 Agent 决策数据（仅 Agent 流程有）
        loop_count = 0
        agent_decisions: List[Dict[str, Any]] = []
        if agent_context is not None:
            loop_count = agent_context.loop_count
            # 从 retrieval_history 构建 Retriever 决策记录
            for rec in agent_context.retrieval_history:
                agent_decisions.append({
                    "agent": "Retriever",
                    "decision": f"hits={rec.get('hits', 0)}",
                    "latency_ms": rec.get("latency_ms", 0),
                    "round": rec.get("round", 0),
                    "detail": {
                        "query": rec.get("query", ""),
                        "hits": rec.get("hits", 0),
                        "strategy": rec.get("strategy", ""),
                    },
                })
            # Planner 决策
            if agent_context.retrieval_plan:
                plan = agent_context.retrieval_plan
                agent_decisions.append({
                    "agent": "Planner",
                    "decision": "out_of_domain" if plan.get("is_out_of_domain") else "proceed",
                    "latency_ms": 0,
                    "round": 0,
                    "detail": {
                        "intent": plan.get("intent", ""),
                        "complexity": plan.get("complexity", ""),
                        "domain_confidence": plan.get("domain_confidence", 0),
                        "strategies": plan.get("retrieval_plan", {}).get("strategies", []),
                    },
                })
            # Evaluator 决策
            if agent_context.evaluation_result:
                ev = agent_context.evaluation_result
                agent_decisions.append({
                    "agent": "Evaluator",
                    "decision": "sufficient" if ev.get("is_sufficient") else "insufficient",
                    "latency_ms": 0,
                    "round": loop_count,
                    "detail": {
                        "sufficiency_score": ev.get("sufficiency_score", 0),
                        "dimensions": ev.get("dimensions", {}),
                        "missing_claims": ev.get("missing_claims", []),
                        "suggestion": ev.get("retrieval_suggestion", {}),
                    },
                })
            # Verifier 决策
            if agent_context.verification_result:
                vr = agent_context.verification_result
                if vr.get("skipped"):
                    _v_decision = "skipped"
                elif vr.get("verified"):
                    _v_decision = "verified"
                elif vr.get("needs_retry"):
                    _v_decision = "needs_retry"
                else:
                    _v_decision = "partial"
                agent_decisions.append({
                    "agent": "Verifier",
                    "decision": _v_decision,
                    "latency_ms": 0,
                    "round": loop_count,
                    "detail": {
                        "verified_count": vr.get("verified_count", 0),
                        "unverified_count": vr.get("unverified_count", 0),
                        "needs_retry": vr.get("needs_retry", False),
                        "claims": vr.get("claims", []),
                        "skipped": vr.get("skipped", False),
                        "error": vr.get("error", ""),
                    },
                })

        return QueryResponse(
            request_id=request_id,
            session_id=session.session_id,
            answer=answer.answer_text,
            citations=answer.citations,
            intent=query_spec.intent,
            complexity=route_decision.level,
            is_refusal=answer.is_refusal,
            refusal_reason=answer.refusal_reason,
            confidence=answer.confidence,
            state_trace=sm.get_state_trace(),
            evidence_count=evidence_bundle.evidence_count if evidence_bundle else 0,
            sufficiency_score=evidence_bundle.sufficiency_score if evidence_bundle else 0.0,
            latency_ms=latency_ms,
            claims_with_evidence=answer.claims_with_evidence,
            ambiguities=query_spec.ambiguities,
            loop_count=loop_count,
            agent_decisions=agent_decisions,
        )

    def _finalize(
        self,
        session: Any,
        request: QueryRequest,
        response: QueryResponse,
        request_id: str,
    ):
        """会话收尾：保存历史、缓存幂等"""
        session.add_turn(
            query=request.query,
            answer=response.answer,
            metadata={
                "request_id": request_id,
                "intent": response.intent,
                "complexity": response.complexity,
            },
        )

        # 幂等缓存
        if request.idempotency_key:
            self._session_manager.cache_response(
                request.idempotency_key,
                response.model_dump(),
            )
