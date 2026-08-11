"""
DAG 执行器 — 子任务模型定义

定义 DAG 中子任务的数据结构与执行状态枚举。任务分解器（task decomposer）
将复杂查询拆分为多个 DagTask，由 DagExecutor 按依赖关系编排执行。

设计要点:
  1. 状态显式: 每个任务携带 TaskStatus，便于追踪与检查点恢复
  2. 依赖声明: 通过 dependencies 列表表达任务间的前置关系，形成 DAG
  3. 可序列化: DagTask / DagState 均提供 to_dict，DagState 额外提供
     from_dict 用于检查点恢复
  4. 结果自包含: 任务执行结果写入 result 字段，供后继任务消费

使用方式:
    from agent_platform.orchestration.dag_executor import DagTask, TaskStatus

    task = DagTask(
        task_id="t1",
        description="检索相关文档",
        dependencies=[],
        retrieval_strategy={"strategy": "vector", "top_k": 5},
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    """
    子任务状态枚举 — 描述任务在 DAG 执行过程中的生命周期阶段

    继承 str 以便于序列化与日志输出。
    """

    PENDING = "pending"  # 待执行（尚未满足执行条件）
    READY = "ready"  # 就绪（依赖已完成，可被调度）
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 已取消（通常因上游任务失败）
    BLOCKED = "blocked"  # 已阻塞（需要用户澄清）


@dataclass
class DagTask:
    """
    DAG 子任务 — 任务分解后的最小执行单元

    每个子任务描述一项独立的检索或处理工作，通过 dependencies 声明
    与其他任务的前置关系。DagExecutor 根据依赖图按层并行执行。

    Attributes:
        task_id: 任务唯一标识
        description: 任务描述（自然语言，供执行器理解意图）
        input_constraints: 输入约束（如检索范围、时间过滤等）
        dependencies: 依赖的 task_id 列表，全部完成后本任务才可执行
        retrieval_strategy: 检索策略（如 vector/keyword/hybrid 及参数）
        completion_condition: 完成条件（自然语言描述，供校验）
        evidence_ids: 关联的证据 ID 列表
        result: 任务执行结果
        failure_reason: 失败原因（失败时填充）
        allow_parallel: 是否允许与其他任务并行执行
        requires_clarification: 是否需要用户澄清
        status: 当前任务状态
        started_at: 开始时间（ISO 字符串）
        completed_at: 完成时间（ISO 字符串）
    """

    task_id: str
    description: str
    input_constraints: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retrieval_strategy: Dict[str, Any] = field(default_factory=dict)
    completion_condition: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    allow_parallel: bool = True
    requires_clarification: bool = False
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典，用于日志输出与序列化"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "input_constraints": self.input_constraints,
            "dependencies": list(self.dependencies),
            "retrieval_strategy": self.retrieval_strategy,
            "completion_condition": self.completion_condition,
            "evidence_ids": list(self.evidence_ids),
            "result": self.result,
            "failure_reason": self.failure_reason,
            "allow_parallel": self.allow_parallel,
            "requires_clarification": self.requires_clarification,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DagTask":
        """从字典恢复 DagTask（用于检查点恢复）"""
        # 兼容缺失字段的情形，仅取已知字段
        status_raw = data.get("status", TaskStatus.PENDING.value)
        status = (
            status_raw
            if isinstance(status_raw, TaskStatus)
            else TaskStatus(status_raw)
        )
        return cls(
            task_id=data["task_id"],
            description=data.get("description", ""),
            input_constraints=dict(data.get("input_constraints", {})),
            dependencies=list(data.get("dependencies", [])),
            retrieval_strategy=dict(data.get("retrieval_strategy", {})),
            completion_condition=data.get("completion_condition", ""),
            evidence_ids=list(data.get("evidence_ids", [])),
            result=dict(data.get("result", {})),
            failure_reason=data.get("failure_reason"),
            allow_parallel=data.get("allow_parallel", True),
            requires_clarification=data.get("requires_clarification", False),
            status=status,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class DagState:
    """
    DAG 执行状态快照 — 记录一次 DAG 执行的全局状态

    包含所有任务及其最终状态、实际执行顺序与完成情况。可序列化为
    字典用于持久化检查点，并通过 from_dict 恢复以支持故障恢复。

    Attributes:
        tasks: 全部子任务列表
        execution_order: 实际执行顺序（task_id 列表）
        is_complete: 是否全部任务已完成
        has_failure: 是否存在失败或取消的任务
    """

    tasks: List[DagTask] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    is_complete: bool = False
    has_failure: bool = False

    def to_dict(self) -> dict:
        """转换为字典，用于日志输出与序列化"""
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "execution_order": list(self.execution_order),
            "is_complete": self.is_complete,
            "has_failure": self.has_failure,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DagState":
        """
        从字典恢复（用于检查点恢复）

        Args:
            data: to_dict 产出的字典

        Returns:
            重建的 DagState 实例
        """
        tasks = [DagTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            execution_order=list(data.get("execution_order", [])),
            is_complete=data.get("is_complete", False),
            has_failure=data.get("has_failure", False),
        )

    def __repr__(self) -> str:
        return (
            f"DagState(tasks={len(self.tasks)}, "
            f"order={len(self.execution_order)}, "
            f"complete={self.is_complete}, failure={self.has_failure})"
        )
