"""编排模块 — 状态机、DAG执行器、预算控制、检查点恢复"""

from .budget_controller import (
    BudgetAction,
    BudgetConsumed,
    BudgetController,
    BudgetEnforcer,
    ExecutionBudget,
)
from .checkpoint_manager import (
    CheckpointManager,
    RecoveryManager,
    RecoveryResult,
)
from .state_machine import (
    AgentState,
    Checkpoint,
    IllegalTransition,
    StateEvent,
    StateMachine,
    StateMachineError,
)

__all__ = [
    # 状态机
    "StateMachine",
    "AgentState",
    "StateEvent",
    "Checkpoint",
    "StateMachineError",
    "IllegalTransition",
    # 预算控制器
    "BudgetController",
    "BudgetEnforcer",
    "ExecutionBudget",
    "BudgetConsumed",
    "BudgetAction",
    # 检查点恢复
    # 注: 此处的 Checkpoint 为状态机内存快照；面向持久化的完整执行检查点
    # 请从 agent_platform.orchestration.checkpoint_manager 导入。
    "CheckpointManager",
    "RecoveryManager",
    "RecoveryResult",
]
