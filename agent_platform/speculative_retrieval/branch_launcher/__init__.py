"""
分支启动器子模块 — 分层启动检索分支

提供 SpeculativeLauncher（推测式检索启动器）与 RetrievalBranch
（检索分支数据结构），按 T0/T1/T2 分层策略启动各检索通道。
"""

from .launcher import BranchStatus, RetrievalBranch, SpeculativeLauncher

__all__ = ["SpeculativeLauncher", "RetrievalBranch", "BranchStatus"]
