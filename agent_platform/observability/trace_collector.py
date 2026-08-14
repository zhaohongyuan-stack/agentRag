"""
执行追踪收集器 — Phase 6

为单次问答收集完整执行轨迹，结构化记录:
  - Planner 决策（越界判断、意图、策略）
  - 检索轮次（query、命中数、耗时）
  - 评估明细（充分性分数、各维度得分、检索建议）
  - 生成信息（模型、token、耗时）
  - 验证结果（已验证/未验证声明数、是否触发补充检索）
  - 状态机迁移轨迹
  - 总耗时

设计原则:
  - 收集器与存储解耦：finalize() 仅返回 dict，不直接写库
  - 轻量：所有 record_xxx 仅内存操作，不阻塞主流程
  - 可读：finalize() 输出的 JSON 结构清晰，业务人员可读
  - 容错：单个 record 失败不影响主流程（调用方自行处理）
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    单次问答的执行追踪收集器

    生命周期:
        collector = TraceCollector(session_id)
        collector.record_query(query)
        # ... 各 Agent 执行时调用 record_xxx ...
        trace = collector.finalize(total_latency_ms)
        # 由 handler 调用 TraceStore.save(trace)
    """

    def __init__(self, session_id: str, request_id: Optional[str] = None):
        self.session_id = session_id
        self.request_id = request_id or session_id
        self._start_ts = time.time()
        self._trace: Dict[str, Any] = {
            "session_id": session_id,
            "request_id": self.request_id,
            "query": "",
            "planner": None,
            "retrieval_rounds": [],
            "evaluations": [],
            "generation": None,
            "verification": None,
            "state_trace": [],
            "loop_count": 0,
            "total_latency_ms": 0,
            "start_ts": self._start_ts,
        }

    # ──────────────────────────────────────────────────────
    # 基础信息
    # ──────────────────────────────────────────────────────

    def record_query(self, query: str) -> None:
        """记录原始用户问题"""
        self._trace["query"] = query

    def record_state(self, state: str) -> None:
        """记录状态机迁移（按发生顺序追加）"""
        self._trace["state_trace"].append(state)

    def record_loop_count(self, loop_count: int) -> None:
        """记录最终 Loop 轮次"""
        self._trace["loop_count"] = loop_count

    # ──────────────────────────────────────────────────────
    # Agent 执行记录
    # ──────────────────────────────────────────────────────

    def record_planner(
        self,
        result: Dict[str, Any],
        latency_ms: int,
        decision: Optional[str] = None,
    ) -> None:
        """
        记录 Planner 决策

        Args:
            result: Planner 输出的结构化数据（is_out_of_domain, intent, retrieval_plan 等）
            latency_ms: 本次调用耗时
            decision: Agent 决策标签（proceed/refuse/out_of_domain）
        """
        self._trace["planner"] = {
            "decision": decision,
            "is_out_of_domain": result.get("is_out_of_domain"),
            "domain_confidence": result.get("domain_confidence"),
            "intent": result.get("intent"),
            "complexity": result.get("complexity"),
            "sub_queries": result.get("sub_queries", []),
            "retrieval_plan": result.get("retrieval_plan", {}),
            "latency_ms": latency_ms,
        }

    def record_retrieval(
        self,
        round_num: int,
        query: str,
        hits: int,
        latency_ms: int,
        strategy: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录一轮检索

        Args:
            round_num: 检索轮次（0 表示首轮）
            query: 实际用于检索的查询文本
            hits: 命中数
            latency_ms: 检索耗时
            strategy: 使用的检索策略
            filters: 检索过滤条件
        """
        self._trace["retrieval_rounds"].append({
            "round": round_num,
            "query": query,
            "hits": hits,
            "strategy": strategy,
            "filters": filters or {},
            "latency_ms": latency_ms,
        })

    def record_evaluation(
        self,
        round_num: int,
        score: float,
        dimensions: Dict[str, Any],
        suggestion: Dict[str, Any],
        latency_ms: int,
        is_sufficient: Optional[bool] = None,
    ) -> None:
        """
        记录一轮评估

        Args:
            round_num: 评估对应的检索轮次
            score: 充分性分数（0-1）
            dimensions: 各维度得分
            suggestion: 检索建议
            latency_ms: 评估耗时
            is_sufficient: 是否充分
        """
        self._trace["evaluations"].append({
            "round": round_num,
            "is_sufficient": is_sufficient,
            "sufficiency_score": score,
            "dimensions": dimensions,
            "missing_claims": suggestion.get("missing_claims", []) if isinstance(suggestion, dict) else [],
            "retrieval_suggestion": suggestion,
            "latency_ms": latency_ms,
        })

    def record_generation(
        self,
        model: str,
        tokens: Dict[str, int],
        latency_ms: int,
        confidence: Optional[float] = None,
    ) -> None:
        """
        记录回答生成

        Args:
            model: 使用的模型名称
            tokens: token 使用（prompt_tokens/completion_tokens/total_tokens）
            latency_ms: 生成耗时
            confidence: 置信度
        """
        self._trace["generation"] = {
            "model": model,
            "tokens": tokens,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }

    def record_verification(
        self,
        verified_count: int,
        unverified_count: int,
        needs_retry: bool,
        claims: Optional[List[Dict[str, Any]]] = None,
        latency_ms: Optional[int] = None,
    ) -> None:
        """
        记录验证结果

        Args:
            verified_count: 已验证声明数
            unverified_count: 未验证声明数
            needs_retry: 是否触发补充检索
            claims: 各声明的验证详情
            latency_ms: 验证耗时
        """
        self._trace["verification"] = {
            "verified_count": verified_count,
            "unverified_count": unverified_count,
            "needs_retry": needs_retry,
            "claims": claims or [],
            "latency_ms": latency_ms,
        }

    # ──────────────────────────────────────────────────────
    # 最终化
    # ──────────────────────────────────────────────────────

    def finalize(self, total_latency_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        结束收集，返回完整 trace 字典

        Args:
            total_latency_ms: 总耗时，None 则按启动时间计算
        """
        if total_latency_ms is None:
            total_latency_ms = int((time.time() - self._start_ts) * 1000)
        self._trace["total_latency_ms"] = total_latency_ms
        # 检索轮次 = 已记录的 retrieval_rounds 长度
        self._trace["retrieval_round_count"] = len(self._trace["retrieval_rounds"])
        return self._trace

    def to_dict(self) -> Dict[str, Any]:
        """返回当前 trace（未 finalize 也可读取）"""
        return dict(self._trace)
