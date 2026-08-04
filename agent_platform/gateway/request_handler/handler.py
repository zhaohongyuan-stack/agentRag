"""
请求处理器 — Agent 主流程编排

完整流程:
  用户问题
    → request_handler 接收请求，创建/恢复会话
    → query_understanding 构建 QuerySpec（意图+实体+约束+歧义+复杂度）
    → query_rewriter 查询改写（指代消解+同义词扩展）    [Phase 2]
    → route_policy 综合路由（规则+复杂度+风险 → P0-P4）  [Phase 2]
    → state_machine 管理状态迁移
    → [歧义] → 澄清请求 → 返回
    → [L0/P0] → 直接回复 → 返回
    → [L1-L4] → retrieval_client 调用检索
    → evidence_builder 组装证据包（去重+父聚合+排序）    [Phase 2]
    → [证据不足] → 拒答 → 返回
    → [证据充分] → generator 生成回答（LLM 接地生成）    [Phase 2]
    → 返回带引用的回答

Phase 2 增强:
  - LLM 接地回答生成（DeepSeek via OpenAI 兼容 API）
  - 查询改写（指代消解+同义词扩展）
  - 综合路由（风险评级+P0-P4 执行路径）
  - 证据去重和父文档聚合
  - 兼容 Phase 1 模板模式和 Mock 服务
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from agent_platform.evidence.evidence_assembler import EvidenceBuilder
from agent_platform.generation.grounded_generator import GroundedGenerator, TemplateGenerator
from agent_platform.orchestration.state_machine import AgentState, StateMachine
from agent_platform.query_understanding import QuerySpec, QuerySpecBuilder
from agent_platform.query_understanding.query_rewriter import QueryRewriter
from agent_platform.routing.route_policy import RoutePolicy
from agent_platform.routing.rule_router import RuleRouter

from .models import QueryRequest, QueryResponse
from .retrieval_client import RetrievalClient
from ..session_handler.session_state import SessionManager

logger = logging.getLogger(__name__)


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
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        # Phase 2: 默认使用 GroundedGenerator（内部自动降级到模板）
        self._generator = generator or GroundedGenerator()
        self._session_manager = session_manager or SessionManager()
        self._query_rewriter = query_rewriter or QueryRewriter()

    def handle_query(self, request: QueryRequest) -> QueryResponse:
        """
        处理用户查询请求 — 主入口

        Args:
            request: 查询请求

        Returns:
            查询响应
        """
        start_time = _now_ms()
        request_id = str(uuid.uuid4())

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
            query_spec: QuerySpec = self._query_spec_builder.build(
                request.query, session_id=session.session_id
            )
            sm.transition(AgentState.ANALYZED, {
                "step": "analyze",
                "intent": query_spec.intent,
                "complexity": query_spec.complexity,
            })

            # ── 查询改写（Phase 2）──
            # 从会话历史构建上下文，用于指代消解
            session_context = self._build_session_context(session)
            rewritten = self._query_rewriter.rewrite(
                original_query=request.query,
                query_spec=query_spec,
                session_context=session_context,
            )
            # 使用改写后的查询替代原始查询
            search_query = rewritten.contextualized_query or request.query

            # ── 路由决策（Phase 2: 综合路由）──
            route_decision = self._route_policy.decide(query_spec)
            sm.transition(AgentState.ROUTED, {
                "step": "route",
                "level": route_decision.level,
                "channels": route_decision.channels,
                "path_id": route_decision.path_id,
                "risk_level": route_decision.risk_level,
            })

            # ── 分支处理 ──

            # 分支1: 需要澄清 — 在 CLARIFYING 状态返回澄清请求
            if route_decision.need_clarification and query_spec.ambiguities:
                sm.transition(AgentState.CLARIFYING, {
                    "step": "clarify",
                    "ambiguities": len(query_spec.ambiguities),
                })
                answer = self._generator.generate_clarification(query_spec.ambiguities)

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

            # 分支2: L0 问候 → 直接回复（ROUTED → RESPONDING 合法迁移）
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

            # 分支3: 正常检索流程
            sm.transition(AgentState.RETRIEVING, {
                "step": "retrieve",
                "strategy": route_decision.channels,
            })

            # 构建检索过滤条件
            filters = self._build_filters(query_spec)

            # 调用检索（使用改写后的查询）
            retrieval_result = self._retrieval_client.search_by_spec(
                query_text=search_query,
                route_decision=route_decision,
                filters=filters,
            )
            logger.info(f"[步骤] 检索完成 → {retrieval_result.hit_count} hits, {retrieval_result.latency_ms:.0f}ms, filters={filters or '无'}")

            # ── 语义检索兜底：检索成功但证据为0时（通常因元数据过滤过严，
            #    如 doc_name/工作表名等过滤条件未命中），去掉所有过滤条件重试，
            #    确保语义检索执行，避免0.1s级快速空拒答 ──
            if retrieval_result.success and retrieval_result.hit_count == 0:
                logger.info(
                    f"[步骤] 首次检索证据不足 (hits=0, filters={filters or '无'})，触发无过滤语义检索兜底"
                )
                _step_t = _now_ms()
                retrieval_result = self._retrieval_client.search_by_spec(
                    query_text=search_query,
                    route_decision=route_decision,
                    filters={},  # 去掉所有过滤条件，强制走全库语义检索
                )
                logger.info(
                    f"[步骤] 无过滤语义检索兜底完成 → {retrieval_result.hit_count} hits, "
                    f"{retrieval_result.latency_ms:.0f}ms (总 {_now_ms() - _step_t:.0f}ms)"
                )

            # 检索失败处理
            if not retrieval_result.success:
                logger.warning(f"检索失败: {retrieval_result.error_code} - {retrieval_result.error}")
                sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                    "step": "evidence_assemble",
                    "retrieval_failed": True,
                })
                evidence_bundle = self._evidence_builder.build(
                    hits=[], claims=query_spec.claims, query_text=request.query
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
                )

            # ── 证据验证 ──
            sm.transition(AgentState.EVIDENCE_VALIDATING, {
                "step": "evidence_validate",
                "sufficiency": evidence_bundle.sufficiency_score,
            })

            # 证据不足 → 拒答
            if not evidence_bundle.is_sufficient:
                sm.transition(AgentState.REFUSING, {
                    "step": "refuse",
                    "reason": "证据不足",
                })
                sm.transition(AgentState.RESPONDING, {"step": "respond"})
                answer = self._generator.generate(
                    intent=query_spec.intent,
                    evidence_bundle=evidence_bundle,
                    query_text=request.query,
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
            answer = self._generator.generate(
                intent=query_spec.intent,
                evidence_bundle=evidence_bundle,
                query_text=request.query,
                ambiguities=query_spec.ambiguities,
            )

            # ── 回答验证 ──
            sm.transition(AgentState.ANSWER_VALIDATING, {"step": "answer_validate"})

            # Phase 1 跳过详细验证，直接回复
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
        """从 QuerySpec 构建检索过滤条件"""
        filters = {}
        constraints = query_spec.constraints

        if constraints.get("applicable_scope"):
            filters["applicable_scope"] = constraints["applicable_scope"]

        # 从实体中提取过滤条件
        for entity in query_spec.entities:
            etype = entity.get("entity_type")
            value = entity.get("value")
            if etype == "doc_name" and value:
                filters["doc_name"] = value
            elif etype == "clause_number" and value:
                filters["clause_number"] = f"第{value}条"
            elif etype == "chapter_number" and value:
                filters["chapter_number"] = f"第{value}章"
            elif etype == "table_name" and value:
                filters["table_name"] = value
            elif etype == "attachment_no" and value:
                filters["attachment_no"] = f"附件{value}"

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
    ) -> QueryResponse:
        """构建响应"""
        latency_ms = _now_ms() - start_time

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
