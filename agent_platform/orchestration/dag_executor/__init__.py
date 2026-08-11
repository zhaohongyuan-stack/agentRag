"""
DAG 执行器模块（M4.2）

提供基于依赖图的子任务编排能力：将任务分解器产出的子任务按拓扑分层，
逐层并行执行，并处理失败传播（上游失败则递归取消后继任务）。

核心导出:
    DagExecutor         — DAG 核心执行器（编排 + 失败传播）
    DagTask             — 子任务数据模型
    DagState            — DAG 执行状态快照（可序列化/恢复）
    TaskStatus          — 子任务状态枚举
    DependencyResolver  — 依赖解析与拓扑排序
    ParallelScheduler   — 基于 asyncio 的并行调度器
"""

from .dependency_resolver import DependencyResolver
from .executor import DagExecutor
from .parallel_scheduler import ParallelScheduler
from .task_models import DagState, DagTask, TaskStatus

__all__ = [
    # 核心执行器
    "DagExecutor",
    # 数据模型
    "DagTask",
    "DagState",
    "TaskStatus",
    # 组件
    "DependencyResolver",
    "ParallelScheduler",
]
