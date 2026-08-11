"""查询 IR 构建模块 — M3.4

将查询规格与声明槽位编译为查询中间表示（Query IR）。

主要组件:
  - IRBuilder: 查询 IR 构建器
  - QueryIR: 查询中间表示
  - Operator / Dependency / AnswerShape / StopCondition: IR 数据结构
"""

from .ir_builder import (
    AnswerShape,
    Dependency,
    IRBuilder,
    Operator,
    QueryIR,
    StopCondition,
)

__all__ = [
    "IRBuilder",
    "QueryIR",
    "Operator",
    "Dependency",
    "AnswerShape",
    "StopCondition",
]
