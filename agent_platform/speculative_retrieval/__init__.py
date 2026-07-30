"""
推测式检索模块（M4.3）

在物理计划执行阶段，按分层策略并行启动多个检索分支（通道），
并根据证据增量动态执行早停与分支取消，以在保证证据充分性的
同时降低检索延迟与资源消耗。

分层启动策略:
  - T0: 低成本通道立即启动（exact, lexical, metadata）
  - T1: 延迟启动 Dense（100ms 后）
  - T2: 条件启动 table/relation（证据不足时）

核心导出:
    SpeculativeLauncher — 推测式检索启动器（分层启动检索分支）
    RetrievalBranch     — 检索分支数据结构
    BranchStatus        — 分支状态枚举
    ResultStream        — 检索结果流收集器
    EarlyStopEvaluator  — 早停评估器
    EarlyStopResult     — 早停评估结果
    BranchCanceller     — 动态分支取消器
"""

from .branch_cancellation.canceller import BranchCanceller
from .branch_launcher.launcher import (
    BranchStatus,
    RetrievalBranch,
    SpeculativeLauncher,
)
from .early_stop.evaluator import EarlyStopEvaluator, EarlyStopResult
from .result_stream.stream import ResultStream

__all__ = [
    # 启动器
    "SpeculativeLauncher",
    "RetrievalBranch",
    "BranchStatus",
    # 结果流
    "ResultStream",
    # 早停
    "EarlyStopEvaluator",
    "EarlyStopResult",
    # 分支取消
    "BranchCanceller",
]
