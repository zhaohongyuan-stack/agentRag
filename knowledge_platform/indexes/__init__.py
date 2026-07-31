"""
indexes — 各类型索引

子模块：
  vector/   — FAISS 向量索引（VectorDB）
  lexical/  — 词汇索引（BM25 等）
  metadata/ — 元数据索引
  relation/ — 关系索引
  table/    — 表格索引
"""

from .vector import VectorDB

__all__ = ["VectorDB"]
