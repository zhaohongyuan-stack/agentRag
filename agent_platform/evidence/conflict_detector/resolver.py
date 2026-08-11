"""
冲突解决器 — 冲突优先级排序与展示格式化

职责:
  1. 按优先级排序冲突（法律效力 > 版本状态 > 生效日期 > 适用范围 > 数值）
  2. 格式化冲突为展示用字典列表
  3. 给出解决建议（不自动解决，仅提供提示文案）

设计要点:
  - 不自动解决冲突，仅提供排序和人工参考建议
  - 优先级由 conflict_types.CONFLICT_PRIORITY 决定
  - 解决建议根据冲突类型给出对应提示文案
  - 排序为稳定排序，优先级相同时保持原始相对顺序
"""

import logging
from typing import Any, Dict, List

from .conflict_types import (
    CONFLICT_PRIORITY,
    DEFAULT_PRIORITY,
    Conflict,
    ConflictType,
)

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    冲突优先级判断与展示格式化

    提供冲突排序、展示格式化和解决建议，但不自动决定冲突的解决结果。
    实际解决需人工审核或上层业务规则介入。

    优先级顺序（数字越小越优先）:
      1. 效力冲突（法律效力）
      2. 版本冲突（版本状态）
      3. 时效冲突（生效日期）
      4. 适用范围重叠
      5. 数值不一致
    """

    # 各冲突类型的解决建议文案
    _RESOLUTION_HINTS: Dict[ConflictType, str] = {
        ConflictType.AUTHORITY_CONFLICT: "建议以效力更高的文件为准",
        ConflictType.VERSION_CONFLICT: "建议以最新版本为准",
        ConflictType.TEMPORAL_CONFLICT: "注意过渡期条款适用",
        ConflictType.SCOPE_OVERLAP: "建议核实适用范围，按特别规定优先原则处理",
        ConflictType.NUMERIC_MISMATCH: "建议核实数据来源，以权威文件最新数值为准",
    }

    # ============================================================
    # 公共方法
    # ============================================================

    def sort_by_priority(self, conflicts: List[Conflict]) -> List[Conflict]:
        """
        按优先级排序冲突

        优先级数字越小越靠前（1 最高）。优先级相同时保持原始相对顺序（稳定排序）。

        Args:
            conflicts: 冲突列表

        Returns:
            按优先级升序排列的新列表（不修改原列表）
        """
        # 使用 enumerate 保持稳定排序
        indexed = list(enumerate(conflicts))
        indexed.sort(key=lambda pair: (pair[1].priority, pair[0]))
        return [c for _, c in indexed]

    def format_for_display(self, conflicts: List[Conflict]) -> List[Dict[str, Any]]:
        """
        格式化冲突为展示用字典列表

        输出字段:
          - conflict_id: 冲突 ID
          - conflict_type: 冲突类型中文标签
          - priority: 优先级
          - description: 冲突描述
          - evidence_ids: 涉及证据 ID 列表
          - resolution_hint: 解决建议
          - details: 详细信息

        排序后的列表可直接用于前端展示。

        Args:
            conflicts: 冲突列表

        Returns:
            展示用字典列表（已按优先级排序）
        """
        sorted_conflicts = self.sort_by_priority(conflicts)
        display_list: List[Dict[str, Any]] = []

        for conflict in sorted_conflicts:
            display_list.append({
                "conflict_id": conflict.conflict_id,
                "conflict_type": conflict.conflict_type.value,
                "priority": conflict.priority,
                "description": conflict.description,
                "evidence_ids": list(conflict.evidence_ids),
                "resolution_hint": self.get_resolution_hint(conflict),
                "details": dict(conflict.details),
            })

        return display_list

    def get_resolution_hint(self, conflict: Conflict) -> str:
        """
        给出冲突解决建议

        根据冲突类型返回对应的提示文案。本方法不自动解决冲突，
        仅提供人工参考建议。

        Args:
            conflict: 冲突对象

        Returns:
            解决建议文案
        """
        hint = self._RESOLUTION_HINTS.get(conflict.conflict_type)
        if hint:
            return hint

        # 未登记的冲突类型给出通用建议
        logger.warning(
            "未登记的冲突类型 %s，返回通用建议",
            conflict.conflict_type,
        )
        return "建议人工核实冲突涉及的证据后决定"

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _get_priority(conflict_type: ConflictType) -> int:
        """
        获取冲突类型的优先级

        Args:
            conflict_type: 冲突类型

        Returns:
            优先级数字（1 最高），未登记时返回默认值
        """
        return CONFLICT_PRIORITY.get(conflict_type, DEFAULT_PRIORITY)

    def __repr__(self) -> str:
        return "ConflictResolver()"
