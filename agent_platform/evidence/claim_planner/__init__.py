"""声明槽位规划模块 — M3.1

根据问题意图规划声明槽位，并支持证据填充。

主要组件:
  - ClaimPlanner: 根据意图生成声明槽位
  - SlotFiller: 用证据填充声明槽位
  - CLAIM_TEMPLATES: 预定义声明槽位模板
"""

from .planner import ClaimPlanner, INTENT_TO_TEMPLATE
from .slot_filler import SlotFiller
from .templates import CLAIM_TEMPLATES, GENERIC_TEMPLATE

__all__ = [
    "ClaimPlanner",
    "SlotFiller",
    "CLAIM_TEMPLATES",
    "GENERIC_TEMPLATE",
    "INTENT_TO_TEMPLATE",
]
