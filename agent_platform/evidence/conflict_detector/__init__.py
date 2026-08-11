"""冲突检测器模块 — 证据冲突检测与优先级判断

职责:
  1. 检测证据项之间的五类冲突（数值、版本、范围、效力、时效）
  2. 按优先级排序冲突并提供解决建议
  3. 格式化冲突为展示用结构

主要接口:
  - ConflictDetector: 冲突检测器，detect() 返回 Conflict 列表
  - ConflictResolver: 冲突排序与展示格式化
  - ConflictType: 冲突类型枚举
  - Conflict: 冲突数据结构
"""

from .conflict_types import (
    CONFLICT_PRIORITY,
    DEFAULT_PRIORITY,
    Conflict,
    ConflictType,
)
from .detector import ConflictDetector
from .resolver import ConflictResolver

__all__ = [
    "ConflictDetector",
    "ConflictResolver",
    "ConflictType",
    "Conflict",
    "CONFLICT_PRIORITY",
    "DEFAULT_PRIORITY",
]
