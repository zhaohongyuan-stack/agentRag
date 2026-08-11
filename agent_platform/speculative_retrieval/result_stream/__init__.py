"""
结果流收集器子模块 — 增量收集与去重合并检索结果

提供 ResultStream（检索结果流收集器），支持按分支增量收集结果、
按得分排序、按通道过滤与基于 chunk_id 的去重合并。
"""

from .stream import ResultStream

__all__ = ["ResultStream"]
