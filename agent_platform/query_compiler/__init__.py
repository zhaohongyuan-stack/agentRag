"""查询编译模块 — M3.4 / M4.1

将查询规格编译为可执行的物理检索计划。

编译流水线:
  QuerySpec + Claims → IRBuilder → QueryIR
    → LogicalPlanner → LogicalPlan
    → PhysicalPlanner → PhysicalPlan
    → PlanValidator → ValidationResult

增强（M4.1）:
  → CostEstimator → PlanCost            （成本估计）
  → PlanOptimizer → PhysicalPlan        （多候选生成与择优）
  → PlanCache / CacheKeyGenerator       （计划缓存与键生成）

主要导出:
  - IRBuilder: 查询 IR 构建器
  - LogicalPlanner: 逻辑计划生成器
  - PhysicalPlanner: 物理计划生成器
  - PlanValidator: 计划校验器
  - CostEstimator: 算子成本估计器
  - PlanOptimizer: 计划优化器
  - PlanCache / CacheKeyGenerator / CacheContext: 计划缓存
"""

from .logical_planner.planner import LogicalPlan, LogicalPlanner
from .physical_planner.cost_estimator import CostEstimator, PlanCost
from .physical_planner.plan_optimizer import PlanOptimizer
from .physical_planner.planner import PhysicalPlan, PhysicalPlanner
from .plan_cache.cache import PlanCache
from .plan_cache.cache_key import CacheContext, CacheKeyGenerator
from .plan_validator.validator import PlanValidator, ValidationResult
from .query_ir.ir_builder import (
    AnswerShape,
    Dependency,
    IRBuilder,
    Operator,
    QueryIR,
    StopCondition,
)

__all__ = [
    "IRBuilder",
    "LogicalPlanner",
    "PhysicalPlanner",
    "PlanValidator",
    "QueryIR",
    "Operator",
    "Dependency",
    "AnswerShape",
    "StopCondition",
    "LogicalPlan",
    "PhysicalPlan",
    "ValidationResult",
    # M4.1 增强
    "CostEstimator",
    "PlanCost",
    "PlanOptimizer",
    "PlanCache",
    "CacheKeyGenerator",
    "CacheContext",
]
