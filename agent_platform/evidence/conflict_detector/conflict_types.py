"""
冲突类型定义 — 证据冲突检测的类型与数据结构

职责:
  1. 定义冲突类型枚举（数值不一致、版本冲突、适用范围重叠、效力冲突、时效冲突）
  2. 定义冲突优先级顺序（法律效力 > 发文机关 > 版本状态 > 生效日期 > 适用范围）
  3. 定义冲突数据结构 Conflict

设计要点:
  - 冲突类型枚举值为中文标签，便于直接展示
  - 冲突优先级为整数，1 表示最高优先级
  - Conflict 使用 dataclass，提供 to_dict 方法便于序列化
  - 本模块不依赖 EvidenceItem，可独立复用
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ConflictType(Enum):
    """
    冲突类型枚举

    每种冲突类型对应一类证据间的不一致情况:
      - NUMERIC_MISMATCH: 同一指标在不同证据中出现不同数值
      - VERSION_CONFLICT: 同一文档存在 active 与 superseded 等不同版本状态
      - SCOPE_OVERLAP: 两个不同规定的适用范围相互重叠
      - AUTHORITY_CONFLICT: 不同效力层级的文件对同一问题有不同规定
      - TEMPORAL_CONFLICT: 同一规定存在多个生效日期版本（新旧过渡期）
    """

    NUMERIC_MISMATCH = "数值不一致"
    VERSION_CONFLICT = "版本冲突"
    SCOPE_OVERLAP = "适用范围重叠"
    AUTHORITY_CONFLICT = "效力冲突"
    TEMPORAL_CONFLICT = "时效冲突"


# 冲突优先级（不自动解决，按以下顺序展示）
# 排序依据: 法律效力 > 发文机关 > 版本状态 > 生效日期 > 适用范围
# 数字越小优先级越高，1 为最高
CONFLICT_PRIORITY: Dict[ConflictType, int] = {
    ConflictType.AUTHORITY_CONFLICT: 1,   # 效力冲突（法律效力）
    ConflictType.VERSION_CONFLICT: 2,     # 版本冲突（版本状态）
    ConflictType.TEMPORAL_CONFLICT: 3,    # 时效冲突（生效日期）
    ConflictType.SCOPE_OVERLAP: 4,        # 适用范围重叠
    ConflictType.NUMERIC_MISMATCH: 5,     # 数值不一致
}

# 默认优先级（用于未在 CONFLICT_PRIORITY 中显式登记的类型）
DEFAULT_PRIORITY: int = 9


@dataclass
class Conflict:
    """
    证据冲突描述

    一条 Conflict 描述一组证据间存在的某类冲突。

    Attributes:
        conflict_id: 冲突唯一标识
        conflict_type: 冲突类型（ConflictType 枚举）
        description: 冲突描述（人类可读）
        evidence_ids: 涉及的证据 ID 列表
        details: 详细信息（如冲突数值、版本状态、生效日期等）
        priority: 优先级（1 最高，数字越小越优先处理）
    """

    conflict_id: str
    conflict_type: ConflictType
    description: str
    evidence_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    priority: int = DEFAULT_PRIORITY

    def to_dict(self) -> dict:
        """序列化为字典，便于日志输出与前端展示"""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "description": self.description,
            "evidence_ids": list(self.evidence_ids),
            "details": dict(self.details),
            "priority": self.priority,
        }
