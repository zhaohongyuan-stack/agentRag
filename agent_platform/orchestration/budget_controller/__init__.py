"""
预算控制器模块

提供执行预算的分配、消耗跟踪与执行策略决策。

按执行路径（P0-P4）分配预算上限，在执行过程中计量检索轮次、改写次数、
子任务数、工具调用数、token 与耗时，并据此给出 CONTINUE / STOP / DOWNGRADE
三种预算动作，由 BudgetEnforcer 转化为执行信号。

核心导出:
    BudgetController  — 预算控制器（分配与消耗跟踪）
    BudgetEnforcer    — 预算执行器（动作转策略信号）
    ExecutionBudget   — 执行预算数据结构
    BudgetConsumed    — 预算消耗记录
    BudgetAction      — 预算动作枚举
    BUDGET_BY_PATH    — 按路径 P0-P4 的预算分配表
"""

from .budget_enforcer import BudgetEnforcer
from .budget_models import (
    BUDGET_BY_PATH,
    BudgetAction,
    BudgetConsumed,
    ExecutionBudget,
)
from .controller import BudgetController

__all__ = [
    # 控制器与执行器
    "BudgetController",
    "BudgetEnforcer",
    # 数据模型
    "ExecutionBudget",
    "BudgetConsumed",
    "BudgetAction",
    # 配置表
    "BUDGET_BY_PATH",
]
