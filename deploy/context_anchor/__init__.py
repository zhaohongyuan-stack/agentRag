"""
语境锚点提取器模块

从用户问题中提取"关联强语境"锚点，用于歧义场景下的优先检索。
当触发歧义警告时，不直接返回澄清请求，而是先用锚点做一轮检索。
"""

from .extractor import ContextAnchorExtractor, ContextAnchor

__all__ = ["ContextAnchorExtractor", "ContextAnchor"]
