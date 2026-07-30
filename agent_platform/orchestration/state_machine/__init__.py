"""
Agent 状态机模块

提供状态机核心功能：状态枚举、合法迁移校验、事件记录、检查点恢复。

核心导出:
    AgentState      — 状态枚举
    StateMachine    — 状态机核心
    StateEvent      — 状态迁移事件
    Checkpoint      — 检查点快照
    IllegalTransition — 非法迁移异常
"""

from .exceptions import (
    IllegalTransition,
    StateMachineError,
    StateNotInitializedError,
    TerminalStateError,
)
from .machine import Checkpoint, StateEvent, StateMachine
from .states import (
    STATE_LABELS,
    TERMINAL_STATES,
    TRANSITIONS,
    AgentState,
    get_valid_targets,
    is_terminal,
    is_valid_transition,
)

__all__ = [
    # 状态枚举
    "AgentState",
    "TRANSITIONS",
    "TERMINAL_STATES",
    "STATE_LABELS",
    # 状态机核心
    "StateMachine",
    "StateEvent",
    "Checkpoint",
    # 异常
    "StateMachineError",
    "IllegalTransition",
    "TerminalStateError",
    "StateNotInitializedError",
    # 工具函数
    "is_valid_transition",
    "is_terminal",
    "get_valid_targets",
]
