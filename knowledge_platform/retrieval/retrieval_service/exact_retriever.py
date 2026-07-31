"""
Exact 检索器 — 精确匹配 / 子串包含 / 正则表达式检索（DB 驱动）

职责：对文档文本做精确/模糊匹配，不依赖模型也不依赖内存全文缓存。

⚠️ 第三阶段改造（2026-07-31）：
  - 完全 DB 驱动：通过 SQLite FTS5 / LIKE / 正则查询，不在内存中保存 documents 全文
  - search() 返回 [{"chunk_id": str, "score": float, "match_pos": int}, ...]
  - 原文统一由 ChunkStore 管理
  - 内存占用：O(1)（仅持有 DB 引用，无全文缓存）

匹配模式：
  - "exact":    完整字符串相等（TRIM 后比较）
  - "contains": 子串包含（FTS5 MATCH 优先，降级为 LIKE）
  - "regex":    正则表达式搜索（LIKE 粗筛 + Python re 精排）
  - "prefix":   前缀匹配（查条款编号等）

使用方式：
    retriever = ExactRetriever(db)
    results = retriever.search("第十二条", top_k=5, mode="contains")
    # → [{"chunk_id": "...", "score": 1.0, "match_pos": 12}, ...]
"""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any


class ExactRetriever:
    """精确匹配检索器（DB 驱动，不保存 documents 全文）"""

    def __init__(self, db=None):
        """
        参数：
          db: RetrievalDB 实例（必须，用于 FTS5/LIKE/正则查询）
        """
        self._db = db
        self._chunk_ids: List[str] = []  # 保留用于兼容性（下标映射）
        self._indexed = False

    # ============================================================
    # 索引（兼容旧接口，实际数据在 DB 中）
    # ============================================================
    def index(self, chunk_ids: List[str], documents: List[str],
              metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        兼容旧接口的索引方法。

        DB 驱动模式下，数据已在 insert_chunks 时写入 SQLite，
        此处仅保留 chunk_ids 用于兼容性，不保存 documents 全文。

        参数：
          chunk_ids:  chunk_id 列表
          documents:  文档文本列表（不持久化，仅用于兼容旧接口）
          metadatas:  可选的元数据列表（不持久化）
        """
        self._chunk_ids = chunk_ids
        self._indexed = True
        # 不保存 self.documents 和 self._metadatas

    # ============================================================
    # 检索
    # ============================================================
    def search(self, query: str, top_k: int = 10,
               mode: str = "contains",
               case_sensitive: bool = False,
               allowed_indices: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        精确/模糊匹配检索（DB 驱动）。

        参数：
          query:           查询文本
          top_k:           返回条数
          mode:            匹配模式 "exact" | "contains" | "regex" | "prefix"
          case_sensitive:  是否区分大小写（contains 模式）
          allowed_indices: 允许匹配的文档索引集合（基于 _chunk_ids 下标）

        返回：
          [{"chunk_id": str, "score": float, "match_pos": int}, ...]
        """
        if not self._db:
            return []

        # 调用 DB 对应方法
        if mode == "exact":
            results = self._db.exact_match(query, top_k=top_k)
        elif mode == "contains":
            results = self._db.fts5_search(query, top_k=top_k)
        elif mode == "regex":
            results = self._db.regex_search(query, top_k=top_k)
        elif mode == "prefix":
            results = self._db.prefix_search(query, top_k=top_k)
        else:
            raise ValueError(f"不支持的模式: {mode}，可选: exact/contains/regex/prefix")

        # allowed_indices 过滤（基于下标映射）
        if allowed_indices is not None and self._chunk_ids:
            allowed_ids = set()
            for idx in allowed_indices:
                if 0 <= idx < len(self._chunk_ids):
                    allowed_ids.add(self._chunk_ids[idx])
            results = [r for r in results if r["chunk_id"] in allowed_ids]

        # 格式化返回（保持与旧接口兼容的字段名）
        return [
            {
                "index": self._get_index(r["chunk_id"]),
                "chunk_id": r["chunk_id"],
                "score": r["score"],
                "match_pos": r.get("match_pos", 0),
            }
            for r in results
        ]

    def _get_index(self, chunk_id: str) -> int:
        """chunk_id → 下标（兼容旧接口，性能不敏感）"""
        try:
            return self._chunk_ids.index(chunk_id)
        except ValueError:
            return -1

    def _get_chunk_id(self, idx: int) -> str:
        """下标 → chunk_id"""
        if 0 <= idx < len(self._chunk_ids):
            return self._chunk_ids[idx]
        return ""

    @property
    def doc_count(self) -> int:
        return len(self._chunk_ids)

    # ============================================================
    # 持久化（DB 驱动模式下，pickle 仅保存 chunk_ids）
    # ============================================================
    def save_index(self, path: str) -> None:
        """持久化到 pickle 文件（仅 chunk_ids，不含 documents）"""
        data = {
            "_chunk_ids": self._chunk_ids,
            "_indexed": self._indexed,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, path: str) -> bool:
        """从 pickle 文件加载，成功返回 True"""
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)

        # 兼容旧格式（含 documents 的旧缓存自动迁移：丢弃 documents）
        self._chunk_ids = data.get("_chunk_ids", [])
        self._indexed = data.get("_indexed", True)

        # 旧格式兼容
        if not self._chunk_ids:
            # 如果旧缓存有 documents，用其长度生成占位 ID
            old_docs = data.get("documents", [])
            if old_docs:
                self._chunk_ids = [str(i) for i in range(len(old_docs))]

        return True
