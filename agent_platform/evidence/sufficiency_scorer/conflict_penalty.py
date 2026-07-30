"""
冲突惩罚计算 — 证据冲突与缺失条件的惩罚分计算

职责:
  1. 冲突惩罚计算: 根据冲突类型对充分性评分施加惩罚
  2. 缺失条件惩罚计算: 根据缺失条件数量施加惩罚

设计要点:
  - 冲突惩罚按类型分级: 效力冲突(0.15) > 数值不一致(0.12) > 版本冲突(0.10) > 其他(0.08)
  - 缺失条件惩罚有上限（0.3），避免过度惩罚
  - 兼容多种冲突数据格式（Conflict 对象、dict）
  - 所有日志使用 logging.getLogger(__name__)
"""

import logging
from typing import Any, Dict, List, Optional

from agent_platform.evidence.conflict_detector.conflict_types import ConflictType

logger = logging.getLogger(__name__)


# ============================================================
# 冲突惩罚映射表 — 冲突类型到惩罚分的映射
# ============================================================
CONFLICT_PENALTIES: Dict[ConflictType, float] = {
    ConflictType.AUTHORITY_CONFLICT: 0.15,
    ConflictType.VERSION_CONFLICT: 0.10,
    ConflictType.NUMERIC_MISMATCH: 0.12,
}

# 默认冲突惩罚（未在映射表中登记的冲突类型）
DEFAULT_CONFLICT_PENALTY: float = 0.08


# ============================================================
# 缺失条件惩罚参数
# ============================================================
MISSING_PENALTY_PER_ITEM: float = 0.05  # 每个缺失条件的惩罚
MAX_MISSING_PENALTY: float = 0.3        # 缺失惩罚上限


# ============================================================
# 冲突类型字符串映射（兼容多种输入格式）
# 自动从枚举生成: 枚举名、小写名、中文值 三种形式均可匹配
# ============================================================
_TYPE_STRING_MAP: Dict[str, ConflictType] = {}
for _ct in ConflictType:
    _TYPE_STRING_MAP[_ct.name] = _ct            # 枚举名: "AUTHORITY_CONFLICT"
    _TYPE_STRING_MAP[_ct.name.lower()] = _ct    # 小写: "authority_conflict"
    _TYPE_STRING_MAP[_ct.value] = _ct           # 中文值: "效力冲突"


# ============================================================
# 公共函数
# ============================================================

def calculate_conflict_penalty(conflicts: List[Dict[str, Any]]) -> float:
    """
    计算冲突惩罚

    根据冲突类型对充分性评分施加惩罚:
      - AUTHORITY_CONFLICT（效力冲突）: 0.15
      - VERSION_CONFLICT（版本冲突）: 0.10
      - NUMERIC_MISMATCH（数值不一致）: 0.12
      - 其他: 0.08

    兼容多种冲突数据格式:
      - Conflict 对象（含 conflict_type 属性）
      - dict 含 "type" 键（builder.py 风格，如 "version_conflict"）
      - dict 含 "conflict_type" 键（resolver 风格，枚举或中文字符串）

    Args:
        conflicts: 冲突列表（dict 或 Conflict 对象）

    Returns:
        冲突惩罚总分
    """
    if not conflicts:
        return 0.0

    total_penalty = 0.0
    for conflict in conflicts:
        conflict_type = _normalize_conflict_type(conflict)
        penalty = CONFLICT_PENALTIES.get(conflict_type, DEFAULT_CONFLICT_PENALTY)
        total_penalty += penalty
        logger.debug(
            "冲突惩罚: 类型=%s, 惩罚=%.2f",
            conflict_type.value if conflict_type else "未知",
            penalty,
        )

    logger.debug(
        "冲突惩罚总计: %.4f（共 %d 条冲突）",
        total_penalty,
        len(conflicts),
    )
    return total_penalty


def calculate_missing_penalty(missing_conditions: List[str]) -> float:
    """
    计算缺失条件惩罚

    每个缺失条件惩罚 0.05，总惩罚不超过 0.3。

    Args:
        missing_conditions: 缺失条件列表

    Returns:
        缺失条件惩罚总分（上限 0.3）
    """
    if not missing_conditions:
        return 0.0

    penalty = min(
        len(missing_conditions) * MISSING_PENALTY_PER_ITEM,
        MAX_MISSING_PENALTY,
    )
    logger.debug(
        "缺失条件惩罚: %.4f（缺失 %d 个条件）",
        penalty,
        len(missing_conditions),
    )
    return penalty


# ============================================================
# 内部方法
# ============================================================

def _normalize_conflict_type(conflict: Any) -> Optional[ConflictType]:
    """
    从冲突对象或字典中提取并规范化冲突类型

    兼容以下格式:
      1. Conflict 对象（含 conflict_type 属性，值为 ConflictType 枚举）
      2. dict 含 "type" 键（builder.py 风格，如 "version_conflict"）
      3. dict 含 "conflict_type" 键:
         - ConflictType 枚举
         - 枚举名字符串（如 "AUTHORITY_CONFLICT"）
         - 小写字符串（如 "authority_conflict"）
         - 中文值字符串（如 "效力冲突"）

    Args:
        conflict: 冲突对象或字典

    Returns:
        规范化后的 ConflictType 枚举，无法识别时返回 None
    """
    # 1. Conflict 对象（含 conflict_type 属性）
    if hasattr(conflict, "conflict_type"):
        ct = conflict.conflict_type
        if isinstance(ct, ConflictType):
            return ct
        if isinstance(ct, str):
            return _TYPE_STRING_MAP.get(ct)

    # 2. dict 格式
    if isinstance(conflict, dict):
        # 优先尝试 "type" 键（builder.py 风格）
        type_str = conflict.get("type", "")
        if type_str:
            ct = _TYPE_STRING_MAP.get(str(type_str))
            if ct:
                return ct

        # 再尝试 "conflict_type" 键（resolver 风格）
        ct_val = conflict.get("conflict_type", "")
        if ct_val:
            if isinstance(ct_val, ConflictType):
                return ct_val
            ct = _TYPE_STRING_MAP.get(str(ct_val))
            if ct:
                return ct

    logger.debug("无法识别的冲突类型: %r", conflict)
    return None
