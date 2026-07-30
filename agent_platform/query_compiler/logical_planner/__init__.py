"""逻辑计划生成模块 — M3.4

将 Query IR 编排为逻辑执行计划（阶段 + 依赖）。

主要组件:
  - LogicalPlanner: 逻辑计划生成器
  - LogicalPlan / PlanStage: 逻辑计划数据结构
"""

from .planner import LogicalPlan, LogicalPlanner, PlanStage

__all__ = [
    "LogicalPlanner",
    "LogicalPlan",
    "PlanStage",
]
