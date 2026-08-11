"""
充分性评分器 — 证据充分性综合评分

职责:
  1. 综合各维度分数计算证据充分性评分
  2. 判定证据是否充分（达到阈值）
  3. 输出各维度分数与惩罚明细，便于追溯

评分公式:
  score = coverage * 0.30
        + authority * 0.15
        + version_validity * 0.20
        + condition_completeness * 0.15
        + multi_channel_consistency * 0.20
        - conflict_penalty
        - missing_penalty

设计要点:
  - 复用 coverage_calculator 的各维度计算函数
  - 复用 conflict_penalty 的惩罚计算函数
  - 阈值可通过构造函数配置，默认 0.85
  - SufficiencyScore 使用 dataclass，提供 to_dict 方法
  - 所有日志使用 logging.getLogger(__name__)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

from agent_platform.evidence.evidence_assembler.builder import EvidenceBundle

from .conflict_penalty import calculate_conflict_penalty, calculate_missing_penalty
from .coverage_calculator import (
    calculate_authority,
    calculate_channel_consistency,
    calculate_condition_completeness,
    calculate_coverage,
    calculate_required_coverage,
    calculate_version_validity,
)

logger = logging.getLogger(__name__)


# ============================================================
# 评分权重
# ============================================================
WEIGHT_COVERAGE: float = 0.30
WEIGHT_AUTHORITY: float = 0.15
WEIGHT_VERSION_VALIDITY: float = 0.20
WEIGHT_CONDITION_COMPLETENESS: float = 0.15
WEIGHT_CHANNEL_CONSISTENCY: float = 0.20


@dataclass
class SufficiencyScore:
    """
    充分性评分结果

    Attributes:
        score: 综合充分性评分，取值范围 [0.0, 1.0]
        is_sufficient: 是否达到充分性阈值
        components: 各维度分数明细
        penalties: 各项惩罚明细
    """

    score: float
    is_sufficient: bool
    components: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典，便于日志输出与前端展示"""
        return {
            "score": round(self.score, 4),
            "is_sufficient": self.is_sufficient,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
        }


class SufficiencyScorer:
    """
    证据充分性评分器

    综合声明覆盖率、来源权威性、版本有效性、条件完整性、多通道一致性
    五个维度，并扣除冲突惩罚与缺失惩罚，计算证据充分性评分。

    评分公式:
        score = coverage * 0.30
              + authority * 0.15
              + version_validity * 0.20
              + condition_completeness * 0.15
              + multi_channel_consistency * 0.20
              - conflict_penalty
              - missing_penalty

    阈值默认 0.85，可通过构造函数配置。
    """

    def __init__(self, threshold: float = 0.85):
        """
        Args:
            threshold: 充分性阈值，评分达到或超过此值判定为证据充分
        """
        self._threshold = threshold

    def score(self, bundle: EvidenceBundle) -> SufficiencyScore:
        """
        计算证据充分性评分

        依次计算各维度分数与惩罚项，按加权公式汇总为综合评分。

        Args:
            bundle: 证据包

        Returns:
            SufficiencyScore 评分结果
        """
        # 各维度分数
        coverage = calculate_coverage(bundle.claim_slots)
        required_coverage = calculate_required_coverage(bundle.claim_slots)
        authority = calculate_authority(bundle.evidence_items)
        version_validity = calculate_version_validity(bundle.evidence_items)
        condition_completeness = calculate_condition_completeness(bundle)
        multi_channel_consistency = calculate_channel_consistency(bundle)

        # 惩罚项
        conflict_penalty = calculate_conflict_penalty(bundle.conflicts)
        missing_penalty = calculate_missing_penalty(bundle.missing_conditions)

        # 加权汇总
        score = (
            coverage * WEIGHT_COVERAGE
            + authority * WEIGHT_AUTHORITY
            + version_validity * WEIGHT_VERSION_VALIDITY
            + condition_completeness * WEIGHT_CONDITION_COMPLETENESS
            + multi_channel_consistency * WEIGHT_CHANNEL_CONSISTENCY
            - conflict_penalty
            - missing_penalty
        )
        # 限制在 [0.0, 1.0] 范围内
        score = max(0.0, min(1.0, score))

        is_sufficient = score >= self._threshold

        components: Dict[str, float] = {
            "coverage": coverage,
            "required_coverage": required_coverage,
            "authority": authority,
            "version_validity": version_validity,
            "condition_completeness": condition_completeness,
            "multi_channel_consistency": multi_channel_consistency,
        }

        penalties: Dict[str, float] = {
            "conflict_penalty": conflict_penalty,
            "missing_penalty": missing_penalty,
        }

        result = SufficiencyScore(
            score=score,
            is_sufficient=is_sufficient,
            components=components,
            penalties=penalties,
        )

        logger.debug(
            "充分性评分: %.4f (阈值: %.2f, 充分: %s)",
            score,
            self._threshold,
            is_sufficient,
        )
        return result

    def __repr__(self) -> str:
        return f"SufficiencyScorer(threshold={self._threshold})"
