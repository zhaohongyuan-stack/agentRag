"""证据充分性评分模块 — 证据充分性综合评分

职责:
  1. 计算声明槽位覆盖率、来源权威性、版本有效性等维度分数
  2. 计算冲突惩罚与缺失条件惩罚
  3. 综合各维度分数与惩罚，输出充分性评分

主要接口:
  - SufficiencyScorer: 充分性评分器，score() 返回 SufficiencyScore
  - SufficiencyScore: 评分结果数据结构
  - calculate_coverage / calculate_required_coverage: 覆盖率计算
  - calculate_authority / calculate_version_validity: 质量分计算
  - calculate_condition_completeness / calculate_channel_consistency: 完整性与一致性计算
  - calculate_conflict_penalty / calculate_missing_penalty: 惩罚计算
"""

from .conflict_penalty import (
    CONFLICT_PENALTIES,
    DEFAULT_CONFLICT_PENALTY,
    calculate_conflict_penalty,
    calculate_missing_penalty,
)
from .coverage_calculator import (
    AUTHORITY_WEIGHTS,
    VERSION_WEIGHTS,
    calculate_authority,
    calculate_channel_consistency,
    calculate_condition_completeness,
    calculate_coverage,
    calculate_required_coverage,
    calculate_version_validity,
)
from .scorer import SufficiencyScore, SufficiencyScorer

__all__ = [
    # 评分器
    "SufficiencyScorer",
    "SufficiencyScore",
    # 覆盖率计算
    "calculate_coverage",
    "calculate_required_coverage",
    "calculate_authority",
    "calculate_version_validity",
    "calculate_condition_completeness",
    "calculate_channel_consistency",
    # 惩罚计算
    "calculate_conflict_penalty",
    "calculate_missing_penalty",
    # 常量
    "AUTHORITY_WEIGHTS",
    "VERSION_WEIGHTS",
    "CONFLICT_PENALTIES",
    "DEFAULT_CONFLICT_PENALTY",
]
