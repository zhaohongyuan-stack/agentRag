"""
查询分解器模块

将复合型查询（如多选题）拆分为独立子问题，
每个子问题可独立检索、独立验证证据充分性。
"""

from .decomposer import QueryDecomposer, SubQuery

__all__ = ["QueryDecomposer", "SubQuery"]
