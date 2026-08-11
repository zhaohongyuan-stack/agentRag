"""
vector — FAISS 向量索引

提供基于 FAISS 的向量数据库能力：
  - 构建索引（Flat / IVF / HNSW）
  - 持久化（单文件 .index + 元数据）
  - ANN 近似最近邻搜索
  - 与 DenseRetriever 无缝集成
"""

from .vector_db import VectorDB

__all__ = ["VectorDB"]
