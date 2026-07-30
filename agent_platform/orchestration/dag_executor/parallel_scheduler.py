"""
DAG 执行器 — 基于 asyncio 的并行任务调度器

负责将同一层级内的多个子任务并发提交执行，并提供单任务超时控制。
通过 asyncio.gather 实现并发，单任务异常不会中断整体调度，
异常会被捕获并转化为任务失败状态。

设计要点:
  1. 并发而非并行: 同层任务通过 asyncio.gather 并发提交，I/O 密集型
     检索任务可充分利用等待时间
  2. 故障隔离: 单个任务抛出异常不影响同层其他任务，异常被捕获后
     标记该任务为 FAILED 并记录原因
  3. 超时保护: execute_with_timeout 对单任务施加时限，超时标记 FAILED
  4. 状态自洽: 调度器负责将任务置为 RUNNING，执行函数负责置为终态；
     若执行函数未正确设置状态，调度器会兜底处理

使用方式:
    from agent_platform.orchestration.dag_executor import ParallelScheduler

    scheduler = ParallelScheduler()
    done = await scheduler.execute_parallel(tasks, my_async_executor)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, List

from .task_models import DagTask, TaskStatus

logger = logging.getLogger(__name__)

# 执行函数类型：接收 DagTask，返回更新后的 DagTask
ExecutorFunc = Callable[[DagTask], Awaitable[DagTask]]


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串"""
    return datetime.now(timezone.utc).isoformat()


class ParallelScheduler:
    """
    并行任务调度器 — 基于 asyncio 的同层任务并发执行

    提供 execute_parallel（批量并发）与 execute_with_timeout（单任务超时）
    两种调度方式。executor_func 为异步函数，接收 DagTask 并返回
    更新状态后的 DagTask。
    """

    async def execute_parallel(
        self,
        tasks: List[DagTask],
        executor_func: ExecutorFunc,
    ) -> List[DagTask]:
        """
        并发执行多个任务

        将所有任务置为 RUNNING 后并发提交执行，使用 return_exceptions=True
        确保单任务异常不会中断其他任务。异常会被捕获并转化为 FAILED 状态。

        Args:
            tasks: 待执行的任务列表（同一层级，互不依赖）
            executor_func: 异步执行函数 (DagTask) -> DagTask

        Returns:
            执行完成后的任务列表（顺序与输入一致），每个任务状态为
            COMPLETED 或 FAILED
        """
        if not tasks:
            return []

        # 标记为执行中并记录开始时间
        for task in tasks:
            task.status = TaskStatus.RUNNING
            task.started_at = _now_iso()

        logger.debug("并行提交 %d 个任务: %s", len(tasks), [t.task_id for t in tasks])

        # 并发执行，单任务异常不中断整体
        results = await asyncio.gather(
            *[executor_func(task) for task in tasks],
            return_exceptions=True,
        )

        final: List[DagTask] = []
        for task, result in zip(tasks, results):
            if isinstance(result, BaseException):
                # 执行函数抛出异常 -> 标记失败
                task.status = TaskStatus.FAILED
                task.failure_reason = f"{type(result).__name__}: {result}"
                task.completed_at = _now_iso()
                logger.warning(
                    "任务执行抛出异常 task_id=%s reason=%s",
                    task.task_id,
                    task.failure_reason,
                    exc_info=result,
                )
                final.append(task)
            elif isinstance(result, DagTask):
                # 执行函数返回了（可能更新过的）任务对象
                if result.completed_at is None:
                    result.completed_at = _now_iso()
                final.append(result)
            else:
                # 执行函数返回 None 或非预期类型，兜底使用原任务
                logger.warning(
                    "执行函数返回非 DagTask 类型 task_id=%s return=%r，"
                    "使用原任务对象",
                    task.task_id,
                    result,
                )
                if task.status not in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                ):
                    task.status = TaskStatus.COMPLETED
                task.completed_at = _now_iso()
                final.append(task)

        return final

    async def execute_with_timeout(
        self,
        task: DagTask,
        executor_func: ExecutorFunc,
        timeout_ms: int,
    ) -> DagTask:
        """
        带超时执行单个任务

        将任务置为 RUNNING 后提交执行，若在 timeout_ms 内未完成则取消
        并标记为 FAILED。超时与异常均不会向外抛出，而是写入任务状态。

        Args:
            task: 待执行的任务
            executor_func: 异步执行函数 (DagTask) -> DagTask
            timeout_ms: 超时阈值（毫秒）

        Returns:
            执行完成后的任务（状态为 COMPLETED 或 FAILED）
        """
        task.status = TaskStatus.RUNNING
        task.started_at = _now_iso()

        timeout_s = timeout_ms / 1000.0
        try:
            result = await asyncio.wait_for(
                executor_func(task), timeout=timeout_s
            )
            if isinstance(result, DagTask):
                if result.completed_at is None:
                    result.completed_at = _now_iso()
                return result
            # 兜底：返回非 DagTask 时使用原任务
            logger.warning(
                "执行函数返回非 DagTask 类型 task_id=%s return=%r",
                task.task_id,
                result,
            )
            if task.status not in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                task.status = TaskStatus.COMPLETED
            task.completed_at = _now_iso()
            return task
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.failure_reason = f"任务执行超时（{timeout_ms}ms）"
            task.completed_at = _now_iso()
            logger.warning(
                "任务执行超时 task_id=%s timeout_ms=%d", task.task_id, timeout_ms
            )
            return task
        except Exception as exc:  # noqa: BLE001 - 调度器需兜底所有异常
            task.status = TaskStatus.FAILED
            task.failure_reason = f"{type(exc).__name__}: {exc}"
            task.completed_at = _now_iso()
            logger.warning(
                "任务执行异常 task_id=%s reason=%s",
                task.task_id,
                task.failure_reason,
                exc_info=exc,
            )
            return task
