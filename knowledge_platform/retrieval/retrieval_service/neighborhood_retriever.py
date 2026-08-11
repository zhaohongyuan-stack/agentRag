"""
Neighborhood 检索器 — Chunk 邻域关系查询

职责：
  父子关系：    get_parent() / get_children()
  前后关系：    get_prev() / get_next() / get_surrounding()
  同级关系：    get_siblings()
  完整上下文：  get_context()（父 + 子 + 同级 + 前后 + 所属文档）

依赖：
  retrieval_db.RetrievalDB（SQLite 存储层，使用 parent_chunk_id / prev_chunk_id / next_chunk_id 链）

使用方式：
    from retrieval_db import RetrievalDB
    from neighborhood_retriever import NeighborhoodRetriever

    db = RetrievalDB("retrieval.db").open()
    retriever = NeighborhoodRetriever(db)

    parent   = retriever.get_parent("chunk_0042")
    children = retriever.get_children("chunk_0010")
    context  = retriever.get_context("chunk_0042")
"""

from typing import List, Optional, Dict, Any

from .retrieval_db import RetrievalDB


class NeighborhoodRetriever:
    """
    邻域检索器 — Chunk 之间的图和上下文查询。

    所有方法委托给 RetrievalDB，本类提供语义清晰的接口层。
    """

    def __init__(self, db: RetrievalDB):
        self._db = db

    # ============================================================
    # 父子
    # ============================================================
    def get_parent(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        获取父 chunk。
        适用于：查看某子条款的上级条款、某表格行的表头 chunk。
        """
        return self._db.get_parent(chunk_id)

    def get_children(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        获取所有子 chunks。
        适用于：展开某条款下的所有子条款、查看某表格的所有行。
        """
        return self._db.get_children(chunk_id)

    # ============================================================
    # 前后
    # ============================================================
    def get_prev(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取前一条 chunk（沿 prev_chunk_id 链）"""
        return self._db.get_prev(chunk_id)

    def get_next(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取后一条 chunk（沿 next_chunk_id 链）"""
        return self._db.get_next(chunk_id)

    def get_surrounding(self, chunk_id: str, window: int = 2) -> List[Dict[str, Any]]:
        """
        获取前后各 window 条邻居。
        适用于：给 LLM 提供当前 chunk 的上下文窗口。

        返回顺序：...prev_2, prev_1, current, next_1, next_2...
        """
        return self._db.get_surrounding(chunk_id, window)

    # ============================================================
    # 同级
    # ============================================================
    def get_siblings(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        获取同级 chunks（同一 parent_chunk_id 下的所有 chunk，含自身）。

        适用于：
        - 查看同一条款下的所有子条款
        - 对比同一表格块下的多个 section
        """
        return self._db.get_siblings(chunk_id)

    # ============================================================
    # 完整上下文
    # ============================================================
    def get_context(self, chunk_id: str) -> Dict[str, Any]:
        """
        获取完整关系上下文，一次性返回六个维度。

        返回结构：
          {
            "chunk":    {...},       # 自身
            "parent":   {...}|None,  # 父 chunk
            "children": [...],       # 子 chunks
            "siblings": [...],       # 同级 chunks
            "prev":     {...}|None,  # 前一个邻居
            "next":     {...}|None,  # 后一个邻居
            "doc":      {...}|None,  # 所属文档
          }

        适用于：用户选中一段文本后，查看其完整上下文（给 LLM 提供精确引用范围）。
        """
        return self._db.get_context(chunk_id)
