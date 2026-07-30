"""
预算控制器 — 预算模型定义

定义执行预算的数据结构与按执行路径（P0-P4）的预算分配。

设计要点:
  1. 预算隔离: 每条执行路径 P0-P4 拥有独立的预算上限，防止高复杂度路径
     耗尽整体资源
  2. 多维度计量: 检索轮次、改写次数、子任务数、工具调用数、token、耗时
     均纳入预算管理
  3. 硬限制与降级: 超出硬限制返回 STOP，接近软限制可返回 DOWNGRADE
  4. 环境变量覆盖: 默认值可通过 .env 中的 AGENT_* 配置项覆盖

配置来源: .env 中 AGENT_MAX_RETRIEVAL_ROUNDS 等配置项
路径定义参考: agent_platform/routing/route_policy/path_table.py
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class BudgetAction(str, Enum):
    """
    预算动作枚举 — 检查预算后给出的执行决策

    与 AgentState 类似，继承 str 以便于序列化。
    """

    CONTINUE = "continue"  # 预算充足，继续执行
    STOP = "stop"  # 硬限制触发，必须停止
    DOWNGRADE = "downgrade"  # 临近预算上限，建议降级（如从 P3 降到 P2 策略）


# ============================================================
# 环境变量辅助读取
# ============================================================


def _env_int(key: str, default: int) -> int:
    """从环境变量读取整数，读取失败时回退到默认值"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# ============================================================
# 默认预算参数（可被环境变量覆盖）
# ============================================================

_DEFAULT_MAX_RETRIEVAL_ROUNDS = _env_int("AGENT_MAX_RETRIEVAL_ROUNDS", 3)
_DEFAULT_MAX_REWRITES = _env_int("AGENT_MAX_REWRITES", 2)
_DEFAULT_MAX_SUBTASKS = _env_int("AGENT_MAX_SUBTASKS", 10)
_DEFAULT_MAX_TOOL_CALLS = _env_int("AGENT_MAX_TOOL_CALLS", 5)
_DEFAULT_TOTAL_TIMEOUT_MS = _env_int("AGENT_TOTAL_TIMEOUT_MS", 30000)
_DEFAULT_TOTAL_TOKEN_BUDGET = _env_int("AGENT_TOTAL_TOKEN_BUDGET", 50000)
_DEFAULT_PER_STEP_TIMEOUT_MS = _env_int("AGENT_PER_STEP_TIMEOUT_MS", 5000)


@dataclass
class BudgetConsumed:
    """
    预算消耗记录 — 记录当前已消耗的各维度预算

    每个字段对应一种资源消耗，由 BudgetController 在执行过程中累加。
    """

    retrieval_rounds: int = 0  # 已消耗的检索轮次
    rewrites: int = 0  # 已消耗的查询改写次数
    subtasks: int = 0  # 已执行的子任务数
    tool_calls: int = 0  # 已发起的工具调用数
    tokens: int = 0  # 已消耗的 token 数
    elapsed_ms: int = 0  # 已消耗的耗时（毫秒）

    def to_dict(self) -> dict:
        """转换为字典，用于日志输出与序列化"""
        return {
            "retrieval_rounds": self.retrieval_rounds,
            "rewrites": self.rewrites,
            "subtasks": self.subtasks,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class ExecutionBudget:
    """
    执行预算 — 描述一次执行允许使用的资源上限

    包含各维度的上限阈值和已消耗记录。BudgetController 在每次资源
    消耗时更新 consumed 字段，并据此判断是否触发预算动作。

    Attributes:
        max_retrieval_rounds: 最大检索轮次
        max_rewrites: 最大查询改写次数
        max_subtasks: 最大子任务数
        max_tool_calls: 最大工具调用数
        total_timeout_ms: 总超时（毫秒）
        total_token_budget: 总 token 预算
        per_step_timeout_ms: 单步超时（毫秒）
        consumed: 已消耗预算记录
    """

    max_retrieval_rounds: int = _DEFAULT_MAX_RETRIEVAL_ROUNDS
    max_rewrites: int = _DEFAULT_MAX_REWRITES
    max_subtasks: int = _DEFAULT_MAX_SUBTASKS
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS
    total_timeout_ms: int = _DEFAULT_TOTAL_TIMEOUT_MS
    total_token_budget: int = _DEFAULT_TOTAL_TOKEN_BUDGET
    per_step_timeout_ms: int = _DEFAULT_PER_STEP_TIMEOUT_MS
    consumed: BudgetConsumed = field(default_factory=BudgetConsumed)

    def to_dict(self) -> dict:
        """转换为字典，用于日志输出与序列化"""
        return {
            "max_retrieval_rounds": self.max_retrieval_rounds,
            "max_rewrites": self.max_rewrites,
            "max_subtasks": self.max_subtasks,
            "max_tool_calls": self.max_tool_calls,
            "total_timeout_ms": self.total_timeout_ms,
            "total_token_budget": self.total_token_budget,
            "per_step_timeout_ms": self.per_step_timeout_ms,
            "consumed": self.consumed.to_dict(),
        }


def _build_default_budget(
    *,
    max_retrieval_rounds: int,
    total_timeout_ms: int,
    max_rewrites: int = _DEFAULT_MAX_REWRITES,
    max_subtasks: int = _DEFAULT_MAX_SUBTASKS,
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
    total_token_budget: int = _DEFAULT_TOTAL_TOKEN_BUDGET,
    per_step_timeout_ms: int = _DEFAULT_PER_STEP_TIMEOUT_MS,
) -> ExecutionBudget:
    """
    构建一条执行路径的预算

    仅显式指定各路径差异化的参数（检索轮次、超时），其余维度沿用
    全局默认值（环境变量或内置默认），保持路径间配置的一致性。
    """
    return ExecutionBudget(
        max_retrieval_rounds=max_retrieval_rounds,
        max_rewrites=max_rewrites,
        max_subtasks=max_subtasks,
        max_tool_calls=max_tool_calls,
        total_timeout_ms=total_timeout_ms,
        total_token_budget=total_token_budget,
        per_step_timeout_ms=per_step_timeout_ms,
    )


# ============================================================
# 按执行路径（P0-P4）的预算分配
# 与 routing/route_policy/path_table.py 的 DEFAULT_EXECUTION_PATHS 对齐
# ============================================================
BUDGET_BY_PATH: Dict[str, ExecutionBudget] = {
    "P0": _build_default_budget(
        max_retrieval_rounds=0,
        total_timeout_ms=100,
        max_rewrites=0,
        max_subtasks=1,
        max_tool_calls=0,
        total_token_budget=2000,
    ),
    "P1": _build_default_budget(
        max_retrieval_rounds=1,
        total_timeout_ms=2000,
        max_rewrites=1,
        max_subtasks=3,
        max_tool_calls=2,
        total_token_budget=10000,
    ),
    "P2": _build_default_budget(
        max_retrieval_rounds=2,
        total_timeout_ms=5000,
        max_rewrites=2,
        max_subtasks=5,
        max_tool_calls=3,
        total_token_budget=20000,
    ),
    "P3": _build_default_budget(
        max_retrieval_rounds=3,
        total_timeout_ms=15000,
        max_rewrites=2,
        max_subtasks=8,
        max_tool_calls=4,
        total_token_budget=35000,
    ),
    "P4": _build_default_budget(
        max_retrieval_rounds=5,
        total_timeout_ms=30000,
        max_rewrites=3,
        max_subtasks=10,
        max_tool_calls=5,
        total_token_budget=50000,
    ),
}
