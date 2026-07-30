"""
检查点管理器模块

提供执行检查点的保存、版本管理、加载与故障恢复能力，支持长流程在
进程崩溃或异常中断后从断点继续执行。

核心导出:
    Checkpoint        — 执行检查点数据结构（完整可恢复快照）
    CheckpointManager — 检查点管理器（保存/加载/版本/清理）
    RecoveryManager   — 故障恢复管理器
    RecoveryResult    — 恢复结果
"""

from .checkpoint_models import Checkpoint
from .manager import CheckpointManager
from .recovery import RecoveryManager, RecoveryResult

__all__ = [
    "Checkpoint",
    "CheckpointManager",
    "RecoveryManager",
    "RecoveryResult",
]
