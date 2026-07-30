"""
动态分支取消器子模块 — 取消不再需要的检索分支

提供 BranchCanceller（动态分支取消器），根据当前检索结果与证据
充分性动态取消冗余分支，节省检索开销。
"""

from .canceller import BranchCanceller

__all__ = ["BranchCanceller"]
