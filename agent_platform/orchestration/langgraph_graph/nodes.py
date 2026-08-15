"""
LangGraph 节点函数（阶段 2）

每个节点对应 handler._run_agent_flow 中的一个阶段，逻辑逐条移植，
业务硬约束（doc_name 安全锁、语义兜底、预算控制、空轮次保护、65s 时间闸、
Verifier 最多 1 次补充检索等）全部保留。

节点拓扑（由 graph.py 装配）:

  planner ──越界──> refuse ──────────────────────────┐
     │                                               │
  retriever <── loop_control <── evaluator           │
     │                            │ 充分/空轮/预算    │
  evidence ──V1充分──> generator <┘                  │
                        │                            │
                     verifier ──通过/不重试──> respond│
                        │ needs_retry                │
                  verifier_retry ──有新证据──> verifier
                        │ 无新证据
                     verifier                        │
                                                     v
                                                    END

状态机（sm）保留为 trace 记录器：节点内按原流程做状态迁移，
保证 trace 记录与 V2 手写编排完全一致。
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict

from agent_platform.orchestration.state_machine import AgentState

logger = logging.getLogger(__name__)


def _now_ms() -> float:
    """高精度当前时间戳（毫秒）"""
    return time.perf_counter() * 1000


def _emit_done(runtime: Any, response: Any) -> None:
    """推送 on_done 事件（与 handler 一致：优先 model_dump）"""
    if runtime.event_callback:
        runtime.event_callback.on_done(
            response.model_dump() if hasattr(response, "model_dump") else response.__dict__
        )


def _answer_lacks_option_letter(query: str, answer_text: str) -> bool:
    """规则快检：选项题但回答未给出选项字母 → 必须保留验证机会（时间闸不得跳过）

    Verifier 的核心职责之一是校验回答是否满足问题形式要求（如 ABCD 选项），
    若规则层已发现回答缺选项字母，即使总耗时超限也要让 Verifier 运行以触发纠正重试。
    """
    if not re.search(r"选项\s*[A-D]", query or ""):
        return False
    text = answer_text or ""
    if re.search(r"答案\s*[:：]?\s*\**\s*[A-D]", text):
        return False
    if re.search(r"选\s*[A-D]|选项\s*为?\s*\**\s*[A-D]", text):
        return False
    return True


# ============================================================
# 1. planner_node — PLANNING: PlannerAgent 规划 + 越界检测
# ============================================================


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm, trace_collector, event_callback = rt.sm, rt.trace_collector, rt.event_callback
    request, query_spec, route_decision = rt.request, rt.query_spec, rt.route_decision

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
    # 流式思考回调：Planner 的 JSON 输出过程逐字推送到前端
    def _planner_thinking(text: str) -> None:
        if event_callback:
            event_callback.on_agent_thinking("Planner", text, round=0)
    try:
        planner_result = rt.planner_agent.run(context, thinking_callback=_planner_thinking)
    except Exception as planner_err:
        # Planner LLM失败/超时 → 用默认plan继续Agent流程（不回退V1）
        # Agent流程后续有规则计算兜底,不能因Planner崩溃就放弃整个流程
        logger.warning(
            f"[LangGraph][Planner] LLM调用失败，使用默认plan继续: {planner_err}"
        )
        context.retrieval_plan = {
            "is_out_of_domain": False,
            "domain_confidence": 0.5,
            "intent": query_spec.intent,
            "complexity": query_spec.complexity,
            "retrieval_plan": {
                "strategies": list(route_decision.channels) if route_decision.channels else ["hybrid"],
                "sub_queries": [rt.search_query],
            },
            "sub_queries": [rt.search_query],
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
        f"[LangGraph][Planner] decision={planner_result.decision}, "
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

    is_out_of_domain = bool(context.retrieval_plan.get("is_out_of_domain"))
    if is_out_of_domain:
        # 越界拒答路径不进入 RETRIEVING，base_filters 无需构建
        return {
            "retrieval_plan": context.retrieval_plan,
            "is_out_of_domain": True,
            "refuse_reason": "out_of_domain",
        }

    sm.transition(AgentState.RETRIEVING, {
        "step": "retrieve",
        "planner_decision": planner_result.decision,
        "strategies": context.retrieval_plan.get("retrieval_plan", {}).get("strategies", []),
    })

    # ── 构建初始过滤条件（复用 V1 的 _build_filters 逻辑）──
    # ⚠️ 使用去除选项后的 clean_query 重新提取实体，避免选项中的《文档名》
    #    覆盖问题主体的 doc_name
    rewritten, session = rt.rewritten, rt.session
    if (
        hasattr(rewritten, "clean_query")
        and rewritten.clean_query
        and rewritten.clean_query.strip() != request.query.strip()
    ):
        filter_spec = rt.query_spec_builder.build(
            rewritten.clean_query, session_id=session.session_id
        )
        base_filters = rt.handler._build_filters(filter_spec)
    else:
        base_filters = rt.handler._build_filters(query_spec)

    return {
        "retrieval_plan": context.retrieval_plan,
        "is_out_of_domain": False,
        "base_filters": base_filters,
        "current_query": rt.search_query,
        "current_filters": base_filters,
        "accumulated_hits": [],
    }


def route_after_planner(state: Dict[str, Any]) -> str:
    return "refuse" if state.get("is_out_of_domain") else "retriever"


# ============================================================
# 2. refuse_node — 拒答/澄清统一出口
# ============================================================


def refuse_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm, trace_collector, event_callback = rt.sm, rt.trace_collector, rt.event_callback
    request, query_spec, route_decision = rt.request, rt.query_spec, rt.route_decision
    reason = state.get("refuse_reason", "")
    handler = rt.handler

    if reason == "out_of_domain":
        # ── 越界检测：Planner 判定不属于本系统 → 直接拒答 ──
        # ⚠️ PLANNING → RETRIEVING 是单向迁移，越界必须在此处经 4 个 skip
        #    状态到达 REFUSING（状态机不允许从 RETRIEVING 回到 REFUSING）
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
        logger.info("[LangGraph] Planner 判定越界，生成拒答")
        answer = rt.generator.generate(
            intent="out_of_domain",
            evidence_bundle=None,
            query_text=request.query,
        )
        response = handler._build_response(
            request_id=rt.request_id,
            session=rt.session,
            sm=sm,
            query_spec=query_spec,
            route_decision=route_decision,
            answer=answer,
            start_time=rt.start_time,
            agent_context=context,
        )
        if trace_collector:
            trace_collector.record_state("RESPONDING")
            handler._finalize_and_save_trace(trace_collector, int(_now_ms() - rt.start_time))
        _emit_done(rt, response)
        handler._finalize(rt.session, request, response, rt.request_id)
        return {"response": response}

    # ── Loop 结束后证据不足 → 检查歧义后澄清或拒答（保留 V1 先检索后澄清策略）──
    evidence_bundle = state["evidence_bundle"]
    rewritten = rt.rewritten
    if reason == "clarify":
        sm.transition(AgentState.REFUSING, {
            "step": "refuse",
            "reason": "证据不足+歧义，请求澄清",
            "ambiguities": len(query_spec.ambiguities),
            "loop_count": context.loop_count,
        })
        sm.transition(AgentState.RESPONDING, {"step": "clarify"})
        logger.info(
            f"[LangGraph] 证据不足且存在歧义 → 生成澄清请求 "
            f"(loop={context.loop_count}, ambiguities={len(query_spec.ambiguities)})"
        )
        answer = rt.generator.generate_clarification(query_spec.ambiguities)
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
            f"[LangGraph] 证据不足 → 生成拒答 (loop={context.loop_count}, "
            f"score={evidence_bundle.sufficiency_score:.3f}, "
            f"缺失参数={unfilled_final})"
        )
        answer = rt.generator.generate(
            intent=query_spec.intent,
            evidence_bundle=evidence_bundle,
            query_text=rewritten.clean_query or request.query,
            options=rewritten.options,
            prompt_mode=rewritten.prompt_mode,
        )

    response = handler._build_response(
        request_id=rt.request_id,
        session=rt.session,
        sm=sm,
        query_spec=query_spec,
        route_decision=route_decision,
        answer=answer,
        evidence_bundle=evidence_bundle,
        start_time=rt.start_time,
        agent_context=context,
    )
    if trace_collector:
        trace_collector.record_state("RESPONDING")
        trace_collector.record_loop_count(context.loop_count)
        handler._finalize_and_save_trace(trace_collector, int(_now_ms() - rt.start_time))
    _emit_done(rt, response)
    handler._finalize(rt.session, request, response, rt.request_id)
    return {"response": response}


# ============================================================
# 3. retriever_node — V1 检索 + Redis 缓存 + 语义兜底
# ============================================================


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    trace_collector, event_callback = rt.trace_collector, rt.event_callback
    route_decision = rt.route_decision
    current_query = state["current_query"]
    current_filters = state["current_filters"]

    _step_t = _now_ms()
    logger.info(
        f"[LangGraph][Retriever] loop={context.loop_count}, "
        f"query='{current_query[:60]}', filters={current_filters or '无'}"
    )
    if event_callback:
        event_callback.on_agent_start("Retriever", round=context.loop_count, detail={
            "query": current_query,
            "filters": current_filters or {},
            "strategy": str(route_decision.channels),
        })
        # 思考过程流式输出：检索策略与目标通道
        event_callback.on_agent_thinking(
            "Retriever",
            f"检索目标: {current_query[:80]}\n通道: {', '.join(route_decision.channels) if route_decision.channels else '默认'}"
            + (f"\n过滤条件: {current_filters}" if current_filters else ""),
            round=context.loop_count,
        )
    if event_callback:
        event_callback.on_tool_call(
            "Retriever", "search_regulatory_docs",
            args={"query": current_query, "filters": current_filters or {}},
            round=context.loop_count,
        )
    # ── Phase3: Redis工具结果缓存（相同query+filters取缓存, TTL 300s）──
    _cache_key = f"rag:retrieval:{hashlib.md5(f'{current_query}|{json.dumps(current_filters or {}, sort_keys=True)}|{route_decision.top_k}'.encode()).hexdigest()}"
    _cache_hit = False
    if rt.redis:
        _cached_json = rt.redis.get(_cache_key)
        if _cached_json:
            try:
                _cached_dict = json.loads(_cached_json)
                retrieval_result = type(retrieval_result).__new__(type(retrieval_result))
                retrieval_result.__dict__.update(_cached_dict.get("result", {}))
                _cache_hit = True
                logger.info(f"[LangGraph][Retriever] 命中Redis缓存 → {retrieval_result.hit_count} hits")
            except Exception:
                _cache_hit = False
    if not _cache_hit:
        retrieval_result = rt.retrieval_client.search_by_spec(
            query_text=current_query,
            route_decision=route_decision,
            filters=current_filters,
        )
        # 写入缓存（仅缓存成功结果, TTL 300s）
        if rt.redis and retrieval_result.success:
            try:
                rt.redis.setex(_cache_key, 300, json.dumps({
                    "result": retrieval_result.to_dict() if hasattr(retrieval_result, 'to_dict') else {},
                    "query": current_query,
                    "filters": current_filters,
                }))
            except Exception as _cache_err:
                logger.debug(f"[LangGraph][Retriever] 缓存写入失败: {_cache_err}")
    retrieval_latency = int(_now_ms() - _step_t)
    logger.info(
        f"[LangGraph][Retriever] 检索完成 → {retrieval_result.hit_count} hits, "
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
                f"[LangGraph][Retriever] hits=0 无doc_name过滤，触发无过滤语义检索兜底"
            )
            _step_t = _now_ms()
            retrieval_result = rt.retrieval_client.search_by_spec(
                query_text=current_query,
                route_decision=route_decision,
                filters={},
            )
            logger.info(
                f"[LangGraph][Retriever] 兜底完成 → {retrieval_result.hit_count} hits "
                f"({_now_ms() - _step_t:.0f}ms)"
            )
        else:
            logger.warning(
                f"[LangGraph][Retriever] doc_name过滤0命中，不回退（避免从错误文档检索）"
            )

    return {
        "retrieval_result": retrieval_result,
        "actual_filters": actual_filters,
        "last_retrieval_latency": retrieval_latency,
    }


# ============================================================
# 4. evidence_node — 证据累积 + 组装 + V1 充分性快速判断
# ============================================================


def evidence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm, trace_collector, event_callback = rt.sm, rt.trace_collector, rt.event_callback
    request, query_spec, route_decision = rt.request, rt.query_spec, rt.route_decision
    retrieval_result = state["retrieval_result"]
    accumulated_hits = list(state.get("accumulated_hits", []))
    current_query = state["current_query"]
    actual_filters = state["actual_filters"]
    retrieval_latency = state.get("last_retrieval_latency", 0)

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
            f"[LangGraph][Retriever] 检索失败: {retrieval_result.error_code} - {retrieval_result.error}"
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
            f"[LangGraph][Retriever] 证据累积 → 本轮新增{new_count}条, "
            f"累积{len(accumulated_hits)}条 (loop={context.loop_count})"
        )
    # 始终从累积的全量证据构建 evidence_bundle（不覆盖）
    evidence_bundle = rt.evidence_builder.build(
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
            f"[LangGraph] V1规则评分充分 (score={evidence_bundle.sufficiency_score:.3f})，"
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
        return {
            "accumulated_hits": accumulated_hits,
            "new_hit_count": new_count,
            "evidence_bundle": evidence_bundle,
            "evaluation_result": context.evaluation_result,
            "is_sufficient": True,
        }

    # ── EVIDENCE_VALIDATING: 进入 Evaluator 评估（仅 V1 规则不足时）──
    sm.transition(AgentState.EVIDENCE_VALIDATING, {
        "step": "evidence_validate",
        "loop": context.loop_count,
        "sufficiency": evidence_bundle.sufficiency_score,
    })
    return {
        "accumulated_hits": accumulated_hits,
        "new_hit_count": new_count,
        "evidence_bundle": evidence_bundle,
        "is_sufficient": False,
    }


def route_after_evidence(state: Dict[str, Any]) -> str:
    return "generator" if state.get("is_sufficient") else "evaluator"


# ============================================================
# 5. evaluator_node — EvaluatorAgent 评估（仅 V1 规则不足时）
# ============================================================


def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    trace_collector, event_callback = rt.trace_collector, rt.event_callback
    route_decision = rt.route_decision
    evidence_bundle = state["evidence_bundle"]
    retrieval_result = state["retrieval_result"]
    current_query = state["current_query"]
    retrieval_latency = state.get("last_retrieval_latency", 0)

    if event_callback:
        event_callback.on_agent_start("Evaluator", round=context.loop_count, detail={
            "sufficiency_score": evidence_bundle.sufficiency_score,
            "evidence_count": evidence_bundle.evidence_count if hasattr(evidence_bundle, 'evidence_count') else 0,
        })
    _step_t = _now_ms()
    # 流式思考回调：Evaluator 的 JSON 输出过程逐字推送到前端
    def _eval_thinking(text: str) -> None:
        if event_callback:
            event_callback.on_agent_thinking("Evaluator", text, round=context.loop_count)
    try:
        eval_result = rt.evaluator_agent.run(context, thinking_callback=_eval_thinking)
    except Exception as eval_err:
        # Evaluator LLM 失败/超时不阻断流程：降级判定继续
        # 弱充分降级: LLM 全链路不可用时，只要有证据即视为充分
        # （Generator 侧有规则计算兜底，可脱离 LLM 正确回答取数计算题）
        weak_sufficient = False
        if evidence_bundle.evidence_count > 0:
            weak_sufficient = True
            logger.warning(
                f"[LangGraph][Evaluator] LLM 调用失败，证据非空→弱充分降级直接生成: {eval_err}"
            )
        else:
            logger.warning(
                f"[LangGraph][Evaluator] LLM 调用失败且无证据，降级用 V1 判定继续: {eval_err}"
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
        f"[LangGraph][Evaluator] loop={context.loop_count}, "
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
            f"[LangGraph] 证据充分 (v1={v1_sufficient}, agent={agent_sufficient}, "
            f"score={evidence_bundle.sufficiency_score:.3f})，退出Loop"
        )

    return {
        "evaluation_result": context.evaluation_result,
        "retrieval_suggestion": context.retrieval_suggestion,
        "is_sufficient": bool(is_sufficient),
    }


def route_after_evaluator(state: Dict[str, Any]) -> str:
    context = state["context"]
    if state.get("is_sufficient"):
        return "generator"
    # ── 连续空轮次保护：连续≥2轮无新证据 → 停止无意义重试 ──
    if context.loop_count >= 2 and state.get("new_hit_count", 0) == 0:
        logger.info(
            f"[LangGraph] 本轮无新证据(new_count=0, loop={context.loop_count})，"
            f"停止无意义重试，进入最终判断"
        )
        return "final_judge"
    # ── 不充分：检查预算，决定是否继续 Loop ──
    if not context.can_continue_loop():
        logger.info(
            f"[LangGraph] 预算耗尽，退出Loop (loop={context.loop_count}, "
            f"score={state['evidence_bundle'].sufficiency_score:.3f})"
        )
        return "final_judge"
    return "loop_control"


# ============================================================
# 6. loop_control_node — 预算消耗 + 下一轮检索策略调整
# ============================================================


def loop_control_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm = rt.sm
    evidence_bundle = state["evidence_bundle"]
    base_filters = state["base_filters"]

    action = context.increment_loop()
    if action == "stop":
        logger.info(f"[LangGraph] 预算STOP，退出Loop (loop={context.loop_count})")
        return {"loop_stopped_reason": "budget_stop"}

    # ── 根据 retrieval_suggestion 调整下一轮检索策略 ──
    suggestion = context.retrieval_suggestion or {}
    suggested_query = suggestion.get("suggested_query", "")
    suggested_strategy = suggestion.get("suggested_strategy", "")
    reason = suggestion.get("reason", "")

    logger.info(
        f"[LangGraph] 证据不足，Loop继续 (loop={context.loop_count}, "
        f"action={action}, suggested_strategy={suggested_strategy}, reason={reason[:80]})"
    )

    current_query = state["current_query"]
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
                f"[LangGraph][Layer2] {len(unfilled_claims)}个槽位未填充, "
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
            f"[LangGraph][Layer3] 多策略兜底: 宽松filters={current_filters} "
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

    return {
        "current_query": current_query,
        "current_filters": current_filters,
    }


def route_after_loop_control(state: Dict[str, Any]) -> str:
    return "final_judge" if state.get("loop_stopped_reason") == "budget_stop" else "retriever"


# ============================================================
# 6.5 final_judge_node — Loop 结束后的最终充分性判断
# ============================================================


def final_judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loop 因空轮次保护/预算耗尽退出后的最终判断

    任一充分即可生成回答（V1 规则评分 或 Evaluator LLM 判断）；
    不足时检查歧义后澄清或拒答（保留 V1 先检索后澄清策略）。
    （充分路径不经过本节点：V1快路径/Evaluator充分直接进 generator）
    """
    context = state["context"]
    evidence_bundle = state["evidence_bundle"]
    query_spec = state["runtime"].query_spec

    final_sufficient = (
        evidence_bundle.is_sufficient
        or context.evaluation_result.get("is_sufficient", False)
    )
    if final_sufficient:
        return {}

    if query_spec.ambiguities:
        logger.info(
            f"[LangGraph] 证据不足且存在歧义 → 澄清 "
            f"(loop={context.loop_count}, ambiguities={len(query_spec.ambiguities)})"
        )
        return {"refuse_reason": "clarify"}

    logger.info(
        f"[LangGraph] 证据不足 → 拒答 (loop={context.loop_count}, "
        f"score={evidence_bundle.sufficiency_score:.3f})"
    )
    return {"refuse_reason": "insufficient"}


def route_after_final_judge(state: Dict[str, Any]) -> str:
    return "refuse" if state.get("refuse_reason") else "generator"


# ============================================================
# 7. generator_node — GENERATING: 回答生成（流式优先）
# ============================================================


def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm, trace_collector, event_callback = rt.sm, rt.trace_collector, rt.event_callback
    request, query_spec = rt.request, rt.query_spec
    rewritten = rt.rewritten
    evidence_bundle = state["evidence_bundle"]

    # ── Agent判定充分时，覆盖evidence_bundle的V1评分标志 ──
    # 确保 Generator 不会因 V1 规则评分不足而拒绝生成
    if not evidence_bundle.is_sufficient:
        evidence_bundle.is_sufficient = True
        logger.info(
            "[LangGraph] Agent判定充分，覆盖evidence_bundle.is_sufficient=True "
            f"(v1_score={evidence_bundle.sufficiency_score:.3f})"
        )

    sm.transition(AgentState.GENERATING, {"step": "generate"})
    _step_t = _now_ms()
    logger.info(
        f"[LangGraph] 回答生成中 → intent={query_spec.intent}, "
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
            gen = rt.generator.generate_stream(
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
            logger.info("[LangGraph] 流式生成完成")
        except Exception as stream_err:
            logger.warning(
                f"[LangGraph] 流式生成失败，回退非流式: {stream_err}"
            )
            answer = rt.generator.generate(
                intent=query_spec.intent,
                evidence_bundle=evidence_bundle,
                query_text=rewritten.clean_query or request.query,
                ambiguities=query_spec.ambiguities,
                options=rewritten.options,
                prompt_mode=rewritten.prompt_mode,
            )
    else:
        answer = rt.generator.generate(
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
    logger.info(f"[LangGraph] 回答生成完成 ({gen_latency}ms)")
    if trace_collector:
        # confidence 从 GeneratedAnswer 提取（若可用）
        _conf = getattr(answer, "confidence", None)
        trace_collector.record_generation(
            model=getattr(rt.generator, "model_name", "unknown"),
            tokens=getattr(answer, "tokens", {}) or {},
            latency_ms=gen_latency,
            confidence=_conf,
        )

    # 将生成的回答写入 context 供 Verifier 读取
    context.generated_answer = answer
    return {"generated_answer": answer}


# ============================================================
# 8. verifier_node — ANSWER_VALIDATING: VerifierAgent 声明级验证
# ============================================================


def verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm, trace_collector, event_callback = rt.sm, rt.trace_collector, rt.event_callback
    evidence_bundle = state["evidence_bundle"]
    answer = state["generated_answer"]
    verifier_retry_done = state.get("verifier_retry_done", False)

    # ── Verifier 时间闸：总耗时超限时跳过验证（LLM慢时避免再等45s）──
    # 验证是锦上添花，不阻塞回答输出（回答已验证过证据充分）
    # 阈值放宽到 120s：表格取数类问题前置阶段耗时较长，仍需保留验证机会
    # 例外：选项题回答缺选项字母时不跳过（形式校验必须执行，可触发纠正重试）
    _answer_text = answer.answer_text if hasattr(answer, "answer_text") else ""
    _must_verify = _answer_lacks_option_letter(rt.request.query, _answer_text)
    if _now_ms() - rt.start_time > 120000 and not _must_verify:
        logger.warning(
            f"[LangGraph][Verifier] 总耗时已超120s，跳过验证直接回复 "
            f"(elapsed={int(_now_ms() - rt.start_time) / 1000:.1f}s)"
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
        if event_callback:
            # 补发 start/result 事件，避免前端块停留在“已中断”状态
            event_callback.on_agent_start("Verifier", round=verifier_retry_done, detail={"skipped": True})
            event_callback.on_agent_result(
                "Verifier",
                decision="skipped",
                detail=context.verification_result,
                latency_ms=0,
                round=verifier_retry_done,
            )
        return {
            "verification_result": context.verification_result,
            "verifier_retry_done": verifier_retry_done,
            "needs_retry": False,
        }

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
    # 流式思考回调：Verifier 的 JSON 输出过程逐字推送到前端
    def _verifier_thinking(text: str) -> None:
        if event_callback:
            event_callback.on_agent_thinking("Verifier", text, round=verifier_retry_done)
    try:
        verify_result = rt.verifier_agent.run(context, thinking_callback=_verifier_thinking)
    except Exception as verifier_err:
        # Verifier 失败（如 LLM 超时）不阻断流程：跳过验证，保留已生成的回答
        logger.warning(
            f"[LangGraph][Verifier] 验证失败，跳过验证继续回复: {verifier_err}",
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
            # 以 result 事件收尾（而非仅 error），前端块可正常转为完成态并展示原因
            event_callback.on_agent_result(
                "Verifier",
                decision="skipped",
                detail=context.verification_result,
                latency_ms=_now_ms() - _step_t,
                round=verifier_retry_done,
            )
        return {
            "verification_result": context.verification_result,
            "verifier_retry_done": verifier_retry_done,
            "needs_retry": False,
        }
    logger.info(
        f"[LangGraph][Verifier] decision={verify_result.decision}, "
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
                "query_conformance": vres.get("query_conformance", {"conforms": True, "issue": ""}),
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

    return {
        "verification_result": context.verification_result,
        "verifier_retry_done": verifier_retry_done,
        "needs_retry": bool(context.verification_result.get("needs_retry")),
    }


def route_after_verifier(state: Dict[str, Any]) -> str:
    context = state["context"]
    vres = state.get("verification_result", {})
    if vres.get("verified") and not vres.get("needs_retry"):
        # 验证通过 → 直接回复
        logger.info("[LangGraph] Verifier验证通过，进入回复")
        return "respond"
    if not vres.get("needs_retry"):
        # partial_verified（部分验证）但不要求重试 → 直接回复
        logger.info("[LangGraph] Verifier部分验证，不重试，进入回复")
        return "respond"
    # needs_retry=True：检查是否可补充检索（最多 1 次，避免无限循环）
    if state.get("verifier_retry_done"):
        logger.info("[LangGraph] Verifier补充检索已执行过，不再重试，直接回复")
        return "respond"
    if not context.can_continue_loop():
        logger.info("[LangGraph] Verifier要求重试但预算耗尽，直接回复")
        return "respond"
    retry_query = vres.get("retry_query", "").strip()
    if not retry_query:
        logger.info("[LangGraph] Verifier未提供retry_query，跳过补充检索")
        return "respond"
    return "verifier_retry"


# ============================================================
# 9. verifier_retry_node — Verifier 补充检索 + 重新生成
# ============================================================


def verifier_retry_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm = rt.sm
    request, query_spec, route_decision = rt.request, rt.query_spec, rt.route_decision
    rewritten = rt.rewritten
    evidence_bundle = state["evidence_bundle"]
    retrieval_result = state["retrieval_result"]
    base_filters = state["base_filters"]
    retry_query = state.get("verification_result", {}).get("retry_query", "").strip()

    # ── 触发补充检索：用 retry_query 重新检索+生成 ──
    context.increment_loop()
    logger.info(
        f"[LangGraph] Verifier触发补充检索 (loop={context.loop_count}, "
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
    retry_result = rt.retrieval_client.search_by_spec(
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
        # ⚠️ 保留原行为：仅合并最后一轮 hits，不用跨轮累积列表
        merged_hits = list(retrieval_result.hits) + list(retry_result.hits)
        evidence_bundle = rt.evidence_builder.build(
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
                "[LangGraph] Verifier重试: 覆盖evidence_bundle.is_sufficient=True"
            )
        sm.transition(AgentState.GENERATING, {"step": "regenerate"})
        _step_t = _now_ms()
        # 若重试由“回答不符合问题要求”触发，把校验反馈注入重新生成，纠正回答形式
        _vres = state.get("verification_result", {}) or {}
        _conf = _vres.get("query_conformance") or {}
        _feedback = "" if _conf.get("conforms", True) else (_conf.get("issue") or "")
        answer = rt.generator.generate(
            intent=query_spec.intent,
            evidence_bundle=evidence_bundle,
            query_text=rewritten.clean_query or request.query,
            ambiguities=query_spec.ambiguities,
            options=rewritten.options,
            prompt_mode=rewritten.prompt_mode,
            verification_feedback=_feedback,
        )
        context.generated_answer = answer
        logger.info(f"[LangGraph] 补充检索后重新生成完成 ({_now_ms() - _step_t:.0f}ms)")
        # 重新生成后回到 ANSWER_VALIDATING 做最终验证
        # （状态机合法：GENERATING → ANSWER_VALIDATING）
        return {
            "evidence_bundle": evidence_bundle,
            "generated_answer": answer,
            "retrieval_result": retry_result,
            "verifier_retry_done": True,
            "needs_retry": True,  # 标记有新证据 → 路由回 verifier 重新验证
        }

    logger.info("[LangGraph] 补充检索无新证据，保留原回答")
    # 无新证据时，需要走到 ANSWER_VALIDATING 才能 RESPONDING
    sm.transition(AgentState.EVIDENCE_VALIDATING, {
        "step": "evidence_validate_no_new",
        "sufficiency": evidence_bundle.sufficiency_score,
    })
    sm.transition(AgentState.GENERATING, {"step": "keep_answer"})
    # needs_retry=False → 路由到 respond（不重跑 Verifier，复刻原 break；
    # 原代码此处后续 GENERATING→RESPONDING 为非法迁移，异常由外层
    # _run_graph_flow 捕获后回退 _run_agent_flow，行为与原流程一致）
    return {
        "verifier_retry_done": True,
        "needs_retry": False,
    }


def route_after_verifier_retry(state: Dict[str, Any]) -> str:
    # 有新证据 → 回 verifier 重新验证（verifier_retry_done=True 保证不再重试）
    # 无新证据 → 直接 respond（复刻原 break，不重跑 Verifier）
    return "verifier" if state.get("needs_retry") else "respond"


# ============================================================
# 10. respond_node — RESPONDING: 收尾出口
# ============================================================


def respond_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rt = state["runtime"]
    context = state["context"]
    sm, trace_collector, event_callback = rt.sm, rt.trace_collector, rt.event_callback
    handler = rt.handler
    evidence_bundle = state["evidence_bundle"]
    answer = state["generated_answer"]

    # 状态机合法路径：ANSWER_VALIDATING → RESPONDING
    sm.transition(AgentState.RESPONDING, {"step": "respond"})
    if trace_collector:
        trace_collector.record_state("RESPONDING")
        trace_collector.record_loop_count(context.loop_count)

    response = handler._build_response(
        request_id=rt.request_id,
        session=rt.session,
        sm=sm,
        query_spec=rt.query_spec,
        route_decision=rt.route_decision,
        answer=answer,
        evidence_bundle=evidence_bundle,
        start_time=rt.start_time,
        agent_context=context,
    )
    # Phase 6: finalize trace + 持久化（容错，失败不影响主流程）
    if trace_collector:
        handler._finalize_and_save_trace(trace_collector, int(_now_ms() - rt.start_time))
    _emit_done(rt, response)
    handler._finalize(rt.session, rt.request, response, rt.request_id)
    return {"response": response}
