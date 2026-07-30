"""
状态机异常定义

非法状态迁移、终态操作等异常在此定义。
"""

from typing import Optional


class StateMachineError(Exception):
    """状态机基础异常"""

    def __init__(self, message: str, current_state: Optional[str] = None,
                 target_state: Optional[str] = None):
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(message)


class IllegalTransition(StateMachineError):
    """
    非法状态迁移异常

    当尝试执行不在合法迁移表中的状态迁移时抛出。
    """

    def __init__(self, current_state: str, target_state: str):
        message = (
            f"非法状态迁移: {current_state} → {target_state}。"
            f"当前状态 {current_state} 的合法目标状态请查阅 TRANSITIONS 表。"
        )
        super().__init__(message, current_state, target_state)


class TerminalStateError(StateMachineError):
    """
    终态操作异常

    当尝试从终态（如 RESPONDING、FAILED）执行迁移时抛出。
    """

    def __init__(self, state: str):
        message = f"状态 {state} 是终态，不允许任何迁移操作。"
        super().__init__(message, state, None)


class StateNotInitializedError(StateMachineError):
    """状态机未初始化异常 — 未设置初始状态就尝试迁移"""

    def __init__(self):
        super().__init__("状态机尚未初始化，请先调用 start() 设置初始状态。")
