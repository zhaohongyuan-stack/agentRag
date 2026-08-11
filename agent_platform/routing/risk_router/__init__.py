"""
风险路由器模块

提供基于规则的风险评级，输出 RiskAssessment。

核心导出:
    RiskRouter      — 风险路由器主类
    RiskLevel       — 风险级别枚举
    RiskAssessment  — 风险评估结果
"""

from .risk_router import (
    INTENT_TO_BASE_RISK,
    PROHIBITIVE_KEYWORDS,
    PENALTY_KEYWORDS,
    COMPLIANCE_CHECK_KEYWORDS,
    THRESHOLD_KEYWORDS,
    RiskAssessment,
    RiskLevel,
    RiskRouter,
)

__all__ = [
    "RiskRouter",
    "RiskLevel",
    "RiskAssessment",
    "INTENT_TO_BASE_RISK",
    "PROHIBITIVE_KEYWORDS",
    "PENALTY_KEYWORDS",
    "COMPLIANCE_CHECK_KEYWORDS",
    "THRESHOLD_KEYWORDS",
]
