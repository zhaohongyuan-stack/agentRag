"""
覆盖率计算器 — 证据充分性各维度覆盖率与质量分计算

职责:
  1. 声明槽位覆盖率计算（总体覆盖率与必填槽位覆盖率）
  2. 来源权威性计算（基于 normative_level 效力层级）
  3. 版本有效性计算（active / superseded 加权）
  4. 条件完整性计算（缺失条件越多分越低）
  5. 多通道一致性计算（证据来源通道多样性）

设计要点:
  - 复用 evidence_assembler/builder.py 中的 ClaimSlot、EvidenceItem、EvidenceBundle
  - 权威性映射表定义在本模块内，便于独立调整
  - 所有函数为纯函数，无副作用，便于单元测试
  - 所有日志使用 logging.getLogger(__name__)
"""

import logging
from typing import Dict, List

from agent_platform.evidence.evidence_assembler.builder import (
    ClaimSlot,
    EvidenceBundle,
    EvidenceItem,
)

logger = logging.getLogger(__name__)


# ============================================================
# 权威性映射表 — normative_level 到权重的映射
# 效力层级从高到低: 法律 > 行政法规 > 部门规章 > 规范性文件 > 其他
# ============================================================
AUTHORITY_WEIGHTS: Dict[str, float] = {
    "法律": 1.0,
    "行政法规": 0.9,
    "部门规章": 0.75,
    "规范性文件": 0.6,
    "其他": 0.4,
}

# 默认权威性权重（normative_level 为空或无法识别时）
DEFAULT_AUTHORITY_WEIGHT: float = 0.4


# ============================================================
# 版本状态权重映射
# ============================================================
VERSION_WEIGHTS: Dict[str, float] = {
    "active": 1.0,
    "superseded": 0.3,
}

# 默认版本权重（未识别的版本状态）
DEFAULT_VERSION_WEIGHT: float = 0.5


# ============================================================
# 公共函数
# ============================================================

def calculate_coverage(claim_slots: List[ClaimSlot]) -> float:
    """
    计算声明槽位覆盖率

    覆盖率 = supported_claims / total_claims

    无声明槽位时返回 1.0（有证据即可，覆盖率不构成限制因素）。

    Args:
        claim_slots: 声明槽位列表

    Returns:
        覆盖率，取值范围 [0.0, 1.0]
    """
    if not claim_slots:
        logger.debug("无声明槽位，覆盖率返回 1.0")
        return 1.0

    supported = sum(1 for c in claim_slots if c.status == "supported")
    coverage = supported / len(claim_slots)
    logger.debug(
        "声明覆盖率: %d/%d = %.4f",
        supported,
        len(claim_slots),
        coverage,
    )
    return coverage


def calculate_required_coverage(claim_slots: List[ClaimSlot]) -> float:
    """
    计算必填槽位覆盖率

    只计算 required 槽位（通过 slot_type 字段判断，包含 "required" 字样）。
    无必填槽位时返回 1.0。

    Args:
        claim_slots: 声明槽位列表

    Returns:
        必填槽位覆盖率，取值范围 [0.0, 1.0]
    """
    required_slots = [
        c for c in claim_slots if "required" in (c.slot_type or "").lower()
    ]
    if not required_slots:
        logger.debug("无必填槽位，必填覆盖率返回 1.0")
        return 1.0

    supported = sum(1 for c in required_slots if c.status == "supported")
    coverage = supported / len(required_slots)
    logger.debug(
        "必填槽位覆盖率: %d/%d = %.4f",
        supported,
        len(required_slots),
        coverage,
    )
    return coverage


def calculate_authority(evidence_items: List[EvidenceItem]) -> float:
    """
    计算来源权威性

    基于 normative_level 效力层级计算，权威性映射表:
      法律(1.0) > 行政法规(0.9) > 部门规章(0.75) > 规范性文件(0.6) > 其他(0.4)

    匹配策略:
      1. 精确匹配 normative_level
      2. 关键词包含匹配（如 "行政法规（国务院）" 匹配 "行政法规"）
      3. 无法识别时使用默认权重

    Args:
        evidence_items: 证据项列表

    Returns:
        平均权威性分数，取值范围 [0.0, 1.0]；无证据时返回 0.0
    """
    if not evidence_items:
        logger.debug("无证据项，权威性返回 0.0")
        return 0.0

    total_weight = 0.0
    for ev in evidence_items:
        weight = _get_authority_weight(ev.normative_level)
        total_weight += weight

    authority = total_weight / len(evidence_items)
    logger.debug(
        "来源权威性: %.4f（基于 %d 条证据）",
        authority,
        len(evidence_items),
    )
    return authority


def calculate_version_validity(evidence_items: List[EvidenceItem]) -> float:
    """
    计算版本有效性

    active 版本权重 1.0，superseded 版本权重 0.3，其他版本状态权重 0.5。
    返回所有证据项版本权重的加权平均。

    Args:
        evidence_items: 证据项列表

    Returns:
        版本有效性分数，取值范围 [0.0, 1.0]；无证据时返回 0.0
    """
    if not evidence_items:
        logger.debug("无证据项，版本有效性返回 0.0")
        return 0.0

    total_weight = 0.0
    for ev in evidence_items:
        weight = VERSION_WEIGHTS.get(ev.version_status, DEFAULT_VERSION_WEIGHT)
        total_weight += weight

    validity = total_weight / len(evidence_items)
    logger.debug(
        "版本有效性: %.4f（基于 %d 条证据）",
        validity,
        len(evidence_items),
    )
    return validity


def calculate_condition_completeness(bundle: EvidenceBundle) -> float:
    """
    计算条件完整性

    检查 missing_conditions 数量，缺失条件越少分数越高:
      - 0 个缺失 = 1.0
      - 每个缺失条件扣 0.15
      - 最低为 0.0

    Args:
        bundle: 证据包

    Returns:
        条件完整性分数，取值范围 [0.0, 1.0]
    """
    missing_count = len(bundle.missing_conditions)
    completeness = max(0.0, 1.0 - 0.15 * missing_count)
    logger.debug(
        "条件完整性: %.4f（缺失 %d 个条件）",
        completeness,
        missing_count,
    )
    return completeness


def calculate_channel_consistency(bundle: EvidenceBundle) -> float:
    """
    计算多通道一致性

    检查证据来源通道的多样性（通过 metadata.channel 字段）:
      - 2 个及以上不同通道 = 1.0（多通道一致）
      - 仅 1 个通道 = 0.7（单通道）
      - 无证据 = 0.0

    无 channel 字段的证据归入 "unknown" 通道。

    Args:
        bundle: 证据包

    Returns:
        多通道一致性分数，取值范围 [0.0, 1.0]
    """
    if not bundle.evidence_items:
        logger.debug("无证据项，多通道一致性返回 0.0")
        return 0.0

    channels: set = set()
    for ev in bundle.evidence_items:
        channel = ""
        if ev.metadata:
            channel = ev.metadata.get("channel", "") or ev.metadata.get("通道", "")
        channels.add(channel if channel else "unknown")

    consistency = 1.0 if len(channels) >= 2 else 0.7
    logger.debug(
        "多通道一致性: %.4f（通道数: %d, 通道: %s）",
        consistency,
        len(channels),
        sorted(channels),
    )
    return consistency


# ============================================================
# 内部方法
# ============================================================

def _get_authority_weight(normative_level: str) -> float:
    """
    获取 normative_level 对应的权威性权重

    匹配策略:
      1. 精确匹配
      2. 关键词包含匹配（normative_level 包含映射表中的某个键）
      3. 无法识别时返回默认权重

    Args:
        normative_level: 规范性层级字符串

    Returns:
        权威性权重
    """
    if not normative_level:
        return DEFAULT_AUTHORITY_WEIGHT

    # 精确匹配
    if normative_level in AUTHORITY_WEIGHTS:
        return AUTHORITY_WEIGHTS[normative_level]

    # 关键词包含匹配
    for key, weight in AUTHORITY_WEIGHTS.items():
        if key in normative_level:
            return weight

    logger.debug(
        "无法识别的 normative_level: %s，使用默认权重 %.2f",
        normative_level,
        DEFAULT_AUTHORITY_WEIGHT,
    )
    return DEFAULT_AUTHORITY_WEIGHT
