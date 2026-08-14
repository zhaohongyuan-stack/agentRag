"""
Agent 间共享上下文 - AgentContext

Agent通过AgentContext直接通信：
  - Planner 写入 retrieval_plan
  - Retriever 读取 retrieval_plan，写入 evidence_bundle
  - Evaluator 读取 evidence_bundle，写入 evaluation_result 和 retrieval_suggestion
  - Retriever 读取 retrieval_suggestion 调整下一轮检索策略
  - Generator 写入 generated_answer
  - Verifier 读取 generated_answer，写入 verification_result

AgentContext由handler创建，贯穿整个问答生命周期。
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_platform.evidence.evidence_assembler.builder import EvidenceBundle
from agent_platform.generation.grounded_generator.generator import GeneratedAnswer
from agent_platform.orchestration.budget_controller.controller import BudgetController
from agent_platform.query_understanding import QuerySpec

if TYPE_CHECKING:
    from agent_platform.observability.trace_collector import TraceCollector

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """
    Agent 统一结构化输出

    所有Agent的run方法返回此对象，保证输出格式统一。
    handler和TraceCollector通过此格式获取Agent决策数据。
    """

    agent_name: str
    decision: str          # Agent的决策摘要（如"proceed"/"retry"/"refuse"/"verified"）
    data: Dict[str, Any]   # Agent输出的完整结构化数据
    latency_ms: int        # Agent执行耗时（毫秒）
    success: bool = True   # 是否执行成功
    error: str = ""        # 失败时的错误信息

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "decision": self.decision,
            "data": self.data,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class AgentContext:
    """
    Agent 间共享上下文

    贯穿整个问答生命周期，4个Agent通过此对象传递信息。
    handler创建后传入每个Agent的run方法。

    属性说明：
        session_id: 会话ID
        query: 原始用户问题
        query_spec: V1查询理解结果（QuerySpecBuilder产出）
        route_decision: V1路由决策结果

        retrieval_plan: Planner输出的检索计划
        evidence_bundle: Retriever输出的证据包（每轮覆盖）
        evaluation_result: Evaluator输出的评估结果
        retrieval_suggestion: Evaluator给Retriever的检索建议（驱动Loop）
        generated_answer: Generator输出
        verification_result: Verifier输出的验证结果

        loop_count: 当前检索Loop轮次（从0开始）
        max_loops: 最大允许Loop轮次（由BudgetController控制）
        budget_controller: 预算控制器
    """

    session_id: str
    query: str = ""
    query_spec: Optional[QuerySpec] = None
    route_decision: Optional[Any] = None

    # Agent产出
    retrieval_plan: Dict[str, Any] = field(default_factory=dict)
    evidence_bundle: Optional[EvidenceBundle] = None
    evaluation_result: Dict[str, Any] = field(default_factory=dict)
    retrieval_suggestion: Dict[str, Any] = field(default_factory=dict)
    generated_answer: Optional[GeneratedAnswer] = None
    verification_result: Dict[str, Any] = field(default_factory=dict)

    # Loop控制
    loop_count: int = 0
    budget_controller: Optional[BudgetController] = None

    # 每轮检索历史（供trace和Evaluator参考）
    retrieval_history: List[Dict[str, Any]] = field(default_factory=list)

    # Phase 6: 执行追踪收集器（由 handler 创建并注入，各执行点调用 record_xxx）
    trace_collector: Optional["TraceCollector"] = None

    def increment_loop(self) -> int:
        """递增Loop轮次并检查预算"""
        self.loop_count += 1
        if self.budget_controller:
            action = self.budget_controller.consume_retrieval_round()
            logger.info(
                "Loop轮次递增: %d, 预算动作: %s",
                self.loop_count,
                action.value,
            )
            return action.value  # "continue" / "stop" / "downgrade"
        return "continue"

    def can_continue_loop(self) -> bool:
        """检查是否还可以继续Loop"""
        if self.budget_controller is None:
            return self.loop_count < 3  # 默认上限
        remaining = self.budget_controller.get_remaining()
        return remaining.get("retrieval_rounds", 0) > 0

    def add_retrieval_record(self, query: str, hits: int, strategy: str, latency_ms: int):
        """记录一轮检索历史"""
        self.retrieval_history.append({
            "round": len(self.retrieval_history) + 1,
            "query": query,
            "hits": hits,
            "strategy": strategy,
            "latency_ms": latency_ms,
        })

    def to_trace_dict(self) -> dict:
        """序列化为trace可用的字典（排除不可序列化的对象）"""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "loop_count": self.loop_count,
            "retrieval_plan": self.retrieval_plan,
            "evaluation_result": self.evaluation_result,
            "retrieval_suggestion": self.retrieval_suggestion,
            "verification_result": self.verification_result,
            "retrieval_history": self.retrieval_history,
            "evidence_bundle": self.evidence_bundle.to_dict() if self.evidence_bundle else None,
            "generated_answer": self.generated_answer.to_dict() if self.generated_answer else None,
        }
