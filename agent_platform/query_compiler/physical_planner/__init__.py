"""物理计划生成模块 — M3.4 / M4.1

将逻辑计划转换为物理执行计划（通道 / top_k / 超时 / 停止条件），
并提供成本估计与多计划优化。

主要组件:
  - PhysicalPlanner: 物理计划生成器
  - PhysicalPlan / PlanStage: 物理计划数据结构
  - PLAN_TEMPLATES: 物理计划模板（第一版固定模板）
  - CostEstimator: 算子成本估计器（M4.1）
  - PlanOptimizer: 计划优化器，多候选生成与选择（M4.1）
"""

from .cost_estimator import OPERATOR_COSTS, CostEstimator, PlanCost
from .planner import (
    DEFAULT_STOP_CONDITIONS,
    DEFAULT_TEMPLATE,
    PLAN_TEMPLATES,
    PhysicalPlan,
    PhysicalPlanner,
    PlanStage,
)
from .plan_optimizer import PlanOptimizer

__all__ = [
    "PhysicalPlanner",
    "PhysicalPlan",
    "PlanStage",
    "PLAN_TEMPLATES",
    "DEFAULT_TEMPLATE",
    "DEFAULT_STOP_CONDITIONS",
    "CostEstimator",
    "PlanCost",
    "OPERATOR_COSTS",
    "PlanOptimizer",
]
