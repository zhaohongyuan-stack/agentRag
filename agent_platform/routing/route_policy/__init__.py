"""
综合路由策略模块

组合规则路由 + 风险路由，输出执行路径（P0-P4）的综合路由决策。

核心导出:
    RoutePolicy               — 综合路由策略主类
    ExecutionPath             — 执行路径定义
    PolicyLoader              — 策略加载器
    ComprehensiveRouteDecision — 综合路由决策结果
"""

from .path_table import DEFAULT_EXECUTION_PATHS, ExecutionPath
from .policy_loader import PolicyLoader
from .route_policy import ComprehensiveRouteDecision, RoutePolicy

__all__ = [
    "RoutePolicy",
    "ExecutionPath",
    "PolicyLoader",
    "ComprehensiveRouteDecision",
    "DEFAULT_EXECUTION_PATHS",
]
