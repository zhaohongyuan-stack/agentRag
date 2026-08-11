"""
检查点恢复 — 故障恢复逻辑

从存储中读取检查点，恢复状态机状态、DAG 任务状态（已完成的不重做，
未完成的重新执行）与预算消耗，使 Agent 能从断点继续执行而非从头开始。

恢复流程:
  1. 从存储读取最新检查点
  2. 恢复状态机到检查点状态
  3. 恢复 DAG 任务状态（已完成的不重做，未完成的重新执行）
  4. 恢复预算消耗
  5. 从检查点状态继续执行

设计要点:
  1. 非破坏性: recover 只读取并返回恢复结果，不直接修改运行时状态，
     由调用方据此重建状态机、DAG 与预算控制器
  2. 损坏容错: recover_from_corrupt 在最新版本损坏时回退到历史版本
  3. DAG 感知: get_resume_tasks 依据任务状态区分已完成与待执行任务
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .checkpoint_models import Checkpoint
from .manager import CheckpointManager

logger = logging.getLogger(__name__)

# DAG 任务状态中被视为「已完成」的状态值（小写匹配）
_COMPLETED_TASK_STATUSES = {"completed", "done", "success", "succeeded", "finished"}


@dataclass
class RecoveryResult:
    """
    恢复结果 — 描述一次故障恢复的产出

    Attributes:
        success: 是否恢复成功
        checkpoint: 恢复所依据的检查点（失败时为 None）
        recovered_state: 恢复到的状态机状态
        completed_task_ids: 已完成、无需重做的 DAG 任务 ID 列表
        pending_task_ids: 待重新执行的 DAG 任务 ID 列表
        error: 失败原因（成功时为 None）
    """

    success: bool
    checkpoint: Optional[Checkpoint] = None
    recovered_state: str = ""
    completed_task_ids: List[str] = field(default_factory=list)
    pending_task_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None


class RecoveryManager:
    """
    故障恢复管理器

    依据 CheckpointManager 中持久化的检查点，重建可恢复的执行上下文。
    调用方拿到 RecoveryResult 后，负责将其映射到具体的状态机、DAG
    执行器与预算控制器实例上。
    """

    def __init__(self, checkpoint_manager: CheckpointManager = None):
        """
        初始化恢复管理器

        Args:
            checkpoint_manager: 检查点管理器实例，为 None 时创建默认实例
                                （使用内存后端）
        """
        self._checkpoint_manager: CheckpointManager = (
            checkpoint_manager or CheckpointManager()
        )

    # ============================================================
    # 恢复入口
    # ============================================================

    def recover(self, session_id: str, request_id: str) -> RecoveryResult:
        """
        从检查点恢复

        如果找到检查点，返回恢复结果（包含已完成和待执行的任务）。
        如果没有检查点，返回 success=False。

        Args:
            session_id: 会话 ID
            request_id: 请求 ID

        Returns:
            RecoveryResult
        """
        checkpoint = self._checkpoint_manager.load_latest(session_id, request_id)
        if checkpoint is None:
            logger.info(
                "未找到检查点，无法恢复 sid=%s rid=%s", session_id, request_id
            )
            return RecoveryResult(success=False, error="未找到检查点")

        completed_task_ids, pending_task_ids = self.get_resume_tasks(
            checkpoint.dag_state or {}
        )

        logger.info(
            "恢复成功 sid=%s rid=%s state=%s completed=%d pending=%d",
            session_id, request_id, checkpoint.state,
            len(completed_task_ids), len(pending_task_ids),
        )
        return RecoveryResult(
            success=True,
            checkpoint=checkpoint,
            recovered_state=checkpoint.state,
            completed_task_ids=completed_task_ids,
            pending_task_ids=pending_task_ids,
        )

    def recover_from_corrupt(self, session_id: str, request_id: str) -> RecoveryResult:
        """
        从损坏的检查点恢复

        如果最新检查点损坏，尝试加载上一版本；如果所有版本都损坏，
        返回从头开始（success=False）。

        Args:
            session_id: 会话 ID
            request_id: 请求 ID

        Returns:
            RecoveryResult（成功时 error 字段标注回退到的版本）
        """
        versions = self._checkpoint_manager.list_versions(session_id, request_id)
        if not versions:
            logger.info(
                "无可用检查点版本，从头开始 sid=%s rid=%s", session_id, request_id
            )
            return RecoveryResult(success=False, error="无可用检查点版本")

        # 从最新版本向前逐个尝试，跳过反序列化失败的损坏版本
        for version in sorted(versions, reverse=True):
            checkpoint = self._checkpoint_manager.load_by_version(
                session_id, request_id, version
            )
            if checkpoint is not None:
                completed_task_ids, pending_task_ids = self.get_resume_tasks(
                    checkpoint.dag_state or {}
                )
                logger.info(
                    "从版本 %d 恢复成功 sid=%s rid=%s", version, session_id, request_id
                )
                return RecoveryResult(
                    success=True,
                    checkpoint=checkpoint,
                    recovered_state=checkpoint.state,
                    completed_task_ids=completed_task_ids,
                    pending_task_ids=pending_task_ids,
                    error=f"最新版本可能损坏，已回退到版本 {version}",
                )

        logger.warning(
            "所有检查点版本均损坏 sid=%s rid=%s", session_id, request_id
        )
        return RecoveryResult(success=False, error="所有检查点版本均损坏")

    # ============================================================
    # DAG 任务状态解析
    # ============================================================

    def get_resume_tasks(
        self, dag_state: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        从 DAG 状态中提取已完成和待执行的任务

        约定 dag_state 形如:
            {
                "tasks": [
                    {"task_id": "t1", "status": "completed"},
                    {"task_id": "t2", "status": "pending"},
                    ...
                ]
            }

        Args:
            dag_state: DAG 任务状态快照

        Returns:
            (completed_task_ids, pending_task_ids)
            已完成任务不重做，其余任务（pending/running/failed/未知）待执行。
        """
        completed: List[str] = []
        pending: List[str] = []

        if not isinstance(dag_state, dict):
            return completed, pending

        tasks = dag_state.get("tasks", [])
        if not isinstance(tasks, list):
            return completed, pending

        for task in tasks:
            if not isinstance(task, dict):
                continue
            # 兼容 task_id / id 两种字段命名
            task_id = task.get("task_id") or task.get("id")
            if not task_id:
                continue
            status = str(task.get("status", "")).lower()
            if status in _COMPLETED_TASK_STATUSES:
                completed.append(str(task_id))
            else:
                pending.append(str(task_id))

        return completed, pending
