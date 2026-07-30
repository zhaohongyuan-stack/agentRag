"""
规则路由器模块

提供基于规则的请求路由，输出 RouteDecision。

核心导出:
    RuleRouter        — 规则路由器主类
    RouteDecision     — 路由决策结果
    RouteTable        — 路由决策表
    ComplexityRouter  — 复杂度路由器
"""

from .complexity_router import ComplexityRouter
from .route_table import DEFAULT_ROUTE_TABLE, RouteDecision, RouteTable
from .router import RuleRouter

__all__ = [
    "RuleRouter",
    "RouteDecision",
    "RouteTable",
    "ComplexityRouter",
    "DEFAULT_ROUTE_TABLE",
]
