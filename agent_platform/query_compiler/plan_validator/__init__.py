"""计划合法性校验模块 — M3.4

校验物理计划与 QueryIR 的一致性及执行约束。

主要组件:
  - PlanValidator: 计划校验器
  - ValidationResult: 校验结果
"""

from .validator import PlanValidator, ValidationResult

__all__ = [
    "PlanValidator",
    "ValidationResult",
]
