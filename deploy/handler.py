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
        # 阈值 0.60: 对于 table_lookup 等结构化检索，1-2 条证据即可充分
        # (ES_DEFAULT默认 0.85 过于严格，2/3 条证据只有 0.667)
        self._evidence_builder = evidence_builder or EvidenceBuilder(sufficiency_threshold=0.60)
        # Phase 2: 默认使用 GroundedGenerator（内部自动降级到模板）
        self._generator = generator or GroundedGenerator()
        self._session_manager = session_manager or SessionManager()
        self._query_rewriter = query_rewriter or QueryRewriter()
        # 文档名称解析器（延迟初始化，首次使用时从检索服务加载文档名）
        self._doc_name_resolver = None

    def _resolve_doc_name(self, raw_name: str) -> str:
        """归一化文档名称：去后缀、统一标点、尝试别名解析

        如果能解析到标准名称则返回标准名称，否则返回归一化后的原始名称。
        """
        import re

        # 1. 基本归一化：去文件后缀、统一标点
        normalized = re.sub(r"\.(pdf|docx|xlsx|xls|doc)$", "", raw_name, flags=re.IGNORECASE)
        normalized = normalized.replace("：", ":").replace("（", "(").replace("）", ")")

        # 2. 尝试从检索服务获取文档名列表，构建别名映射
        if self._doc_name_resolver is None:
            try:
                from agent_platform.query_understanding.context_anchor import ContextAnchorExtractor
                # 延迟初始化 DocNameResolver
                from pathlib import Path
                import sys
                # 尝试从检索服务获取文档名
                doc_names = self._fetch_doc_names()
                if doc_names:
                    resolver_code = Path(__file__).parent.parent.parent / "knowledge_platform" / "retrieval" / "retrieval_service" / "doc_name_resolver.py"
                    # 直接内联实现，避免文件依赖
                    from knowledge_platform.retrieval.retrieval_service.doc_name_resolver import DocNameResolver
                    self._doc_name_resolver = DocNameResolver()
                    self._doc_name_resolver.build_from_doc_names(doc_names)
                else:
                    self._doc_name_resolver = False  # 标记为不可用
            except Exception as e:
                logger.debug(f"DocNameResolver 初始化失败，使用基本归一化: {e}")
                self._doc_name_resolver = False

        # 3. 别名解析
        if self._doc_name_resolver:
            resolved = self._doc_name_resolver.resolve(normalized)
            if resolved:
                logger.info(f"[doc_name解析] '{raw_name}' → '{resolved}'")
                return resolved

        return normalized

    def _fetch_doc_names(self) -> list:
        """从检索服务获取所有文档名"""
        try:
            import urllib.request
            import json
            req = urllib.request.Request("http://retrieval:8000/api/v1/documents")
            with urllib.request.urlopen(req, timeout=5) as resp:
                docs = json.loads(resp.read())
                return [d.get("doc_name", "") for d in docs if d.get("doc_name")]
        except Exception:
            return []

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

            # 分支1: 需要澄清 — 先尝试强语境优先检索，再决定是否澄清
            if route_decision.need_clarification and query_spec.ambiguities:
                # 强语境优先检索：如果存在高权重锚点，先尝试检索而非直接澄清
                from agent_platform.query_understanding.context_anchor import ContextAnchorExtractor
                anchor_extractor = ContextAnchorExtractor()
                anchors = anchor_extractor.extract(
                    request.query, entities=query_spec.entities
                )

                if anchor_extractor.has_strong_context(anchors):
                    # 有强语境锚点，先尝试检索
                    logger.info(f"[步骤] 歧义检测到但存在强语境锚点 ({len(anchors)} 个)，尝试优先检索")
                    enhanced_query = anchor_extractor.build_enhanced_query(
                        request.query, anchors
                    )
                    anchor_filters = anchor_extractor.build_anchor_filters(anchors)

                    sm.transition(AgentState.RETRIEVING, {
                        "step": "retrieve",
                        "strategy": "anchor_priority",
                        "anchors": len(anchors),
                    })

                    _step_t = _now_ms()
                    logger.info(f"[步骤] 强语境检索 → query='{enhanced_query[:60]}', filters={anchor_filters or '无'}")
                    anchor_result = self._retrieval_client.search_by_spec(
                        query_text=enhanced_query,
                        route_decision=route_decision,
                        filters=anchor_filters,
                    )
                    logger.info(f"[步骤] 强语境检索完成 → {anchor_result.hit_count} hits ({_now_ms() - _step_t:.0f}ms)")

                    if anchor_result.success and anchor_result.hit_count > 0:
                        sm.transition(AgentState.EVIDENCE_ASSEMBLING, {
                            "step": "evidence_assemble",
                            "hit_count": anchor_result.hit_count,
                        })
                        anchor_evidence = self._evidence_builder.build(
                            hits=anchor_result.hits,
                            claims=query_spec.claims,
                            query_text=request.query,
                        )

                        if anchor_evidence.is_sufficient:
                            # 证据充分，跳过澄清，直接生成回答
                            logger.info(f"[步骤] 强语境检索证据充分 → sufficiency={anchor_evidence.sufficiency_score:.3f}")
                            sm.transition(AgentState.GENERATING, {"step": "generate"})
                            answer = self._generator.generate(
                                intent=query_spec.intent,
                                evidence_bundle=anchor_evidence,
                                query_text=request.query,
                                ambiguities=query_spec.ambiguities,
                            )
                            sm.transition(AgentState.RESPONDING, {"step": "respond"})
                            response = self._build_response(
                                request_id=request_id,
                                session=session,
                                sm=sm,
                                query_spec=query_spec,
                                route_decision=route_decision,
                                answer=answer,
                                evidence_bundle=anchor_evidence,
                                start_time=start_time,
                            )
                            self._finalize(session, request, response, request_id)
                            return response

                    # 强语境检索仍不足，继续走澄清流程
                    logger.info("[步骤] 强语境检索证据不足，回退到澄清流程")

                # 无强语境或检索不足，返回澄清请求
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

            # table_lookup 意图：确保 table 通道在 channels 中（P2 默认不含 table）
            if query_spec.intent == "table_lookup" and "table" not in route_decision.channels:
                route_decision.channels = list(route_decision.channels) + ["table"]

            sm.transition(AgentState.RETRIEVING, {
                "step": "retrieve",
                "strategy": route_decision.channels,
            })

            # 构建检索过滤条件
            filters = self._build_filters(query_spec)

            # 查询分解检索：如果检测到子问题，对每个子问题独立检索，合并结果
            if query_spec.sub_queries:
                logger.info(f"[步骤] 查询分解检索 → {len(query_spec.sub_queries)} 个子问题")
                all_hits = []
                # 子问题检索时不带 doc_name 过滤（已修复多值覆盖问题）
                sub_filters = {k: v for k, v in filters.items() if k != "doc_name"}
                first_result = None
                for sq in query_spec.sub_queries:
                    sq_text = sq.get("text", "")
                    sq_label = sq.get("option_label", "")
                    _sq_t = _now_ms()
                    sq_result = self._retrieval_client.search_by_spec(
                        query_text=sq_text,
                        route_decision=route_decision,
                        filters=sub_filters,
                    )
                    logger.info(f"[步骤] 子问题 {sq_label} 检索 → {sq_result.hit_count} hits ({_now_ms() - _sq_t:.0f}ms)")
                    if first_result is None:
                        first_result = sq_result
                    if sq_result.success and sq_result.hits:
                        for hit in sq_result.hits:
                            if isinstance(hit, dict):
                                hit["_sub_query_label"] = sq_label
                        all_hits.extend(sq_result.hits)

                # 去重（按 chunk_id）
                seen_ids = set()
                deduped_hits = []
                for hit in all_hits:
                    chunk_id = hit.get("chunk_id", str(id(hit))) if isinstance(hit, dict) else str(id(hit))
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        deduped_hits.append(hit)

                # 用第一个结果作为基础，替换 hits（hit_count 是 property，自动更新）
                if first_result is not None:
                    retrieval_result = first_result
                    retrieval_result.hits = deduped_hits
                else:
                    retrieval_result = self._retrieval_client.search_by_spec(
                        query_text=search_query,
                        route_decision=route_decision,
                        filters=filters,
                    )
                logger.info(f"[步骤] 查询分解检索完成 → 合并 {len(deduped_hits)} hits (去重前 {len(all_hits)})")
            else:
                # 正常单次检索
                _step_t = _now_ms()
                logger.info(f"[步骤] 检索中 → query='{search_query[:60]}', filters={filters or '无'}")
                retrieval_result = self._retrieval_client.search_by_spec(
                    query_text=search_query,
                    route_decision=route_decision,
                    filters=filters,
                )
                logger.info(f"[步骤] 检索完成 → {retrieval_result.hit_count} hits, {retrieval_result.latency_ms:.0f}ms")

            # ── 语义检索兜底：检索成功但证据为0时（通常因元数据过滤过严），
            #    去掉所有过滤条件重试，确保语义检索执行，避免7ms级快速空拒答 ──
            if retrieval_result.success and retrieval_result.hit_count == 0:
                original_filters = filters
                logger.info(
                    f"[步骤] 首次检索证据不足 (hits=0, filters={original_filters or '无'})，触发无过滤语义检索兜底"
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
            logger.info(f"[步骤] 证据组装 → 充分性={evidence_bundle.sufficiency_score:.3f}, sufficient={evidence_bundle.is_sufficient}")
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
                logger.info("[步骤] 回答生成 → 证据不足，生成拒答")
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
            _step_t = _now_ms()
            logger.info(f"[步骤] 回答生成中 → intent={query_spec.intent}, evidence_count={evidence_bundle.evidence_count}")
            # DEBUG: 打印证据内容，排查 LLM 返回"依据不足"
            for ev in evidence_bundle.evidence_items[:3]:
                logger.info(f"[DEBUG证据] {ev.chunk_type} | {ev.source_doc}\n{getattr(ev, 'content', '')[:500]}")
            answer = self._generator.generate(
                intent=query_spec.intent,
                evidence_bundle=evidence_bundle,
                query_text=request.query,
                ambiguities=query_spec.ambiguities,
            )
            logger.info(f"[步骤] 回答生成完成 ({_now_ms() - _step_t:.0f}ms)")

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
        # 修复：多个 doc_name 时不设过滤（多选题场景避免误限定检索范围）
        doc_names = []
        for entity in query_spec.entities:
            etype = entity.get("entity_type")
            value = entity.get("value")
            if etype == "doc_name" and value:
                doc_names.append(value)
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

        # 仅当单个 doc_name 时设为过滤条件（多个 doc_name 不设过滤）
        if len(doc_names) == 1:
            # 归一化 doc_name：去 .pdf 后缀、统一标点
            resolved_name = self._resolve_doc_name(doc_names[0])
            filters["doc_name"] = resolved_name
            if query_spec.intent == "table_lookup":
                filters["table_name"] = resolved_name

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
