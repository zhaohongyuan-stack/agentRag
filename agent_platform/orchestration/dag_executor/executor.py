"""
DAG 执行器 — 核心编排引擎

DagExecutor 是 M4.2 的核心组件，负责按依赖关系编排子任务的执行。
它将任务分解器产出的 DagTask 列表按拓扑分层，逐层并行执行，并处理
失败传播：上游任务失败时，其所有后继任务被标记为 CANCELLED。

设计要点:
  1. 分层并行: 通过 DependencyResolver.topological_sort 得到分层计划，
     同层任务由 ParallelScheduler 并发执行，最大化并行度
  2. 失败传播: 某任务 FAILED 时，递归将其后继标记为 CANCELLED，
     避免执行无意义的下游任务
  3. 可插拔执行: executor_func 由调用方注入，默认提供 mock 执行器
     便于测试与集成；真实场景注入检索/生成执行函数
  4. 同步/异步双入口: execute 为原生 async，execute_sync 用 asyncio.run
     包装，适配不便使用 async 的调用方
  5. 状态快照: 执行结束返回 DagState，可序列化用于检查点恢复

使用方式:
    from agent_platform.orchestration.dag_executor import DagExecutor, DagTask

    executor = DagExecutor()
    state = executor.execute_sync(tasks)            # 同步
    state = await executor.execute(tasks, my_func)  # 异步
"""

import asyncio
import logging
from typing import Dict, List, Optional

from .dependency_resolver import DependencyResolver
from .parallel_scheduler import ExecutorFunc, ParallelScheduler
from .task_models import DagState, DagTask, TaskStatus

logger = logging.getLogger(__name__)


class DagExecutor:
    """
    DAG 核心执行器 — 按依赖关系编排子任务的并行执行

    Attributes:
        _resolver: 依赖关系解析器
        _scheduler: 并行任务调度器
    """

    def __init__(self):
        """初始化 DAG 执行器，组装解析器与调度器"""
        self._resolver = DependencyResolver()
        self._scheduler = ParallelScheduler()

    # ============================================================
    # 默认执行函数（mock）
    # ============================================================

    @staticmethod
    async def _default_executor(task: DagTask) -> DagTask:
        """
        默认 mock 执行器 — 直接标记任务为 COMPLETED

        当调用方未提供 executor_func 时使用，便于测试与冒烟验证。
        执行结果写入 result 字段。
        """
        task.status = TaskStatus.COMPLETED
        task.result = {"status": "mock_completed"}
        return task

    # ============================================================
    # 异步执行入口
    # ============================================================

    async def execute(
        self,
        tasks: List[DagTask],
        executor_func: Optional[ExecutorFunc] = None,
    ) -> DagState:
        """
        执行 DAG

        流程:
          1. 检测循环依赖，存在则抛出 ValueError
          2. 拓扑排序得到分层执行计划
          3. 按层并行执行（同层任务并发）
          4. 每层完成后，将失败任务的后继递归标记为 CANCELLED
          5. 汇总结果构造 DagState 返回

        Args:
            tasks: 待执行的子任务列表
            executor_func: 异步执行函数 (DagTask) -> DagTask；
                           为 None 时使用默认 mock 执行器

        Returns:
            DAG 执行状态快照 DagState

        Raises:
            ValueError: 检测到循环依赖
        """
        if not tasks:
            logger.debug("任务列表为空，直接返回空 DagState")
            return DagState()

        func = executor_func or self._default_executor

        # 1. 检测循环依赖
        if self._resolver.has_cycle(tasks):
            cycle_tasks = [t.task_id for t in tasks]
            logger.error("检测到循环依赖，任务: %s", cycle_tasks)
            raise ValueError(
                f"检测到循环依赖，无法执行 DAG，涉及任务: {cycle_tasks}"
            )

        # 2. 拓扑排序
        layers = self._resolver.topological_sort(tasks)
        logger.info(
            "DAG 执行开始，共 %d 个任务，%d 层",
            len(tasks),
            len(layers),
        )

        # 构建 task_id -> DagTask 索引（执行过程中通过索引更新引用）
        task_map: Dict[str, DagTask] = {t.task_id: t for t in tasks}
        execution_order: List[str] = []

        # 3. 按层并行执行
        for layer_idx, layer in enumerate(layers):
            # 过滤掉因上游失败已 CANCELLED 的任务
            runnable = [
                task_map[tid]
                for tid in layer
                if task_map[tid].status == TaskStatus.PENDING
            ]

            if not runnable:
                logger.debug(
                    "第 %d 层无可执行任务（均已被取消）: %s",
                    layer_idx,
                    layer,
                )
                continue

            # 标记为 READY（依赖已就绪）
            for task in runnable:
                task.status = TaskStatus.READY

            logger.debug(
                "执行第 %d 层，任务: %s",
                layer_idx,
                [t.task_id for t in runnable],
            )

            # 并发执行本层任务
            executed = await self._scheduler.execute_parallel(runnable, func)

            # 更新索引与执行顺序
            for task in executed:
                task_map[task.task_id] = task
                execution_order.append(task.task_id)

            # 4. 失败任务的后继标记为 CANCELLED
            failed = [
                t for t in executed if t.status == TaskStatus.FAILED
            ]
            for failed_task in failed:
                logger.warning(
                    "任务失败 task_id=%s reason=%s，取消其后继任务",
                    failed_task.task_id,
                    failed_task.failure_reason,
                )
                self._cancel_dependents(failed_task.task_id, task_map)

        # 5. 构造 DagState
        all_tasks = list(task_map.values())
        is_complete = all(
            t.status == TaskStatus.COMPLETED for t in all_tasks
        )
        has_failure = any(
            t.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in all_tasks
        )

        state = DagState(
            tasks=all_tasks,
            execution_order=execution_order,
            is_complete=is_complete,
            has_failure=has_failure,
        )

        logger.info(
            "DAG 执行完成 complete=%s has_failure=%s executed=%d/%d",
            is_complete,
            has_failure,
            len(execution_order),
            len(all_tasks),
        )
        return state

    # ============================================================
    # 同步执行入口
    # ============================================================

    def execute_sync(
        self,
        tasks: List[DagTask],
        executor_func: Optional[ExecutorFunc] = None,
    ) -> DagState:
        """
        同步执行 DAG（用于不方便使用 async 的场景）

        内部用 asyncio.run 包装 execute。注意：若当前已运行事件循环，
        asyncio.run 会抛出 RuntimeError，此时应直接 await execute。

        Args:
            tasks: 待执行的子任务列表
            executor_func: 异步执行函数 (DagTask) -> DagTask；
                           为 None 时使用默认 mock 执行器

        Returns:
            DAG 执行状态快照 DagState

        Raises:
            ValueError: 检测到循环依赖
            RuntimeError: 当前已有运行中的事件循环
        """
        return asyncio.run(self.execute(tasks, executor_func))

    # ============================================================
    # 内部辅助
    # ============================================================

    def _cancel_dependents(
        self, failed_task_id: str, task_map: Dict[str, DagTask]
    ) -> None:
        """
        递归取消失败任务的所有后继任务

        将直接与间接依赖该失败任务的后继标记为 CANCELLED，并记录原因。
        仅取消尚未执行（PENDING / READY）的任务，已进入终态的任务保持不变。

        Args:
            failed_task_id: 失败任务的 ID
            task_map: task_id -> DagTask 索引
        """
        all_tasks = list(task_map.values())
        dependents = self._resolver.get_dependents(
            failed_task_id, all_tasks
        )
        for dep_id in dependents:
            dep_task = task_map.get(dep_id)
            if dep_task is None:
                continue
            # 仅取消尚未执行的任务
            if dep_task.status in (TaskStatus.PENDING, TaskStatus.READY):
                dep_task.status = TaskStatus.CANCELLED
                dep_task.failure_reason = (
                    f"上游任务 {failed_task_id} 失败，已取消执行"
                )
                logger.debug(
                    "取消后继任务 task_id=%s（上游 %s 失败）",
                    dep_id,
                    failed_task_id,
                )
                # 递归取消该后继的后续任务
                self._cancel_dependents(dep_id, task_map)
