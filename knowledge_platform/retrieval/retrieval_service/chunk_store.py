"""
ChunkStore — 顶层统一 chunk 数据获取接口

设计目标：
  - 检索器不再保存 content 全文，只存索引必需数据 + chunk_id
  - 原文统一由 ChunkStore 管理，分层缓存：内存元信息 + LRU content + SQLite 回源
  - 检索器输出 chunk_id，由 ChunkStore 组装完整数据

分层缓存策略：
  1. _meta_map (常驻内存): chunk_id → ChunkMeta (不含 content，只存小字段)
  2. _lru_cache (LRU 500): chunk_id → content (命中率高的自动缓存)
  3. _db (SQLite 回源): 兜底获取 content

使用方式：
    store = ChunkStore(db)
    store.load_from_chunks(chunks)       # 从 Chunk 列表加载元信息
    meta = store.get_meta(chunk_id)      # O(1) 纯内存
    content = store.get_content(chunk_id)  # LRU → DB 透明回源
    full = store.get_full(chunk_id)      # meta + content 合并
"""

import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


# ============================================================
# ChunkMeta — 轻量元信息（不含 content 全文）
# ============================================================
@dataclass
class ChunkMeta:
    """chunk 元信息（常驻内存，不含 content 全文）"""
    chunk_id: str = ""
    chunk_type: str = "clause"
    doc_id: str = ""
    doc_name: str = ""
    doc_title: str = ""
    hierarchy_path: str = ""
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_title": self.doc_title,
            "hierarchy_path": self.hierarchy_path,
            "source_file": self.source_file,
            "metadata": self.metadata,
        }


# ============================================================
# ChunkStore — 统一数据获取接口
# ============================================================
class ChunkStore:
    """统一 chunk 数据管理 — 分层缓存 + DB 回源"""

    def __init__(self, db=None):
        """
        参数：
          db: RetrievalDB 实例（用于 content 回源）
        """
        self._db = db
        self._meta_map: Dict[str, ChunkMeta] = {}       # chunk_id → 元信息（常驻）
        self._chunk_id_list: List[str] = []              # 有序 chunk_id 列表（下标映射）
        self._lru: OrderedDict[str, str] = OrderedDict()  # chunk_id → content（LRU）
        self._lru_max: int = 500                          # LRU 最大缓存条数

    # ============================================================
    # 加载
    # ============================================================
    def load_from_chunks(self, chunks: List[Any]) -> None:
        """
        从 Chunk 列表加载元信息（不含 content 全文）。

        参数：
          chunks: Chunk 对象列表（仅读取元信息，不保存引用）

        后处理：跨 chunk 回填 doc_name / doc_title / source_file
          —— Excel/Word 解析器常只在首个 chunk 的 metadata 中携带 doc_name，
             后续同文档 chunk 缺失该字段。此处构建 doc_id → doc_name 映射后回填。
        """
        print("  [ChunkStore] 加载 chunk 元信息 ...")
        self._meta_map.clear()
        self._chunk_id_list.clear()

        # ── 第一遍：加载所有元信息 ──
        # 同时收集 doc_id → doc_name / doc_title / source_file 映射
        doc_name_map: Dict[str, str] = {}
        doc_title_map: Dict[str, str] = {}
        source_file_map: Dict[str, str] = {}

        for chunk in chunks:
            # 兼容 Chunk 对象和 dict
            if isinstance(chunk, dict):
                doc_id = str(chunk.get("doc_id", ""))
                doc_name = chunk.get("doc_name", "")
                doc_title = chunk.get("doc_title", "")
                source_file = chunk.get("source_file", chunk.get("source", ""))
                meta = ChunkMeta(
                    chunk_id=chunk.get("chunk_id", ""),
                    chunk_type=chunk.get("chunk_type", "clause"),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_title=doc_title,
                    hierarchy_path=chunk.get("hierarchy_path", ""),
                    source_file=source_file,
                    metadata=chunk.get("metadata", {}),
                )
            else:
                doc_id = str(getattr(chunk, "doc_id", ""))
                doc_name = getattr(chunk, "doc_name", "")
                doc_title = getattr(chunk, "doc_title", "")
                source_file = getattr(chunk, "source_file", "")
                meta = ChunkMeta(
                    chunk_id=getattr(chunk, "chunk_id", ""),
                    chunk_type=getattr(chunk, "chunk_type", "clause"),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_title=doc_title,
                    hierarchy_path=getattr(chunk, "hierarchy_path", ""),
                    source_file=source_file,
                    metadata=getattr(chunk, "metadata", {}),
                )

            self._meta_map[meta.chunk_id] = meta
            self._chunk_id_list.append(meta.chunk_id)

            # 收集非空的 doc_name / doc_title / source_file（首个非空值优先）
            if doc_id and doc_name and doc_id not in doc_name_map:
                doc_name_map[doc_id] = doc_name
            if doc_id and doc_title and doc_id not in doc_title_map:
                doc_title_map[doc_id] = doc_title
            if doc_id and source_file and doc_id not in source_file_map:
                source_file_map[doc_id] = source_file

        # ── 第二遍：回填空缺的 doc_name / doc_title / source_file ──
        filled = 0
        for meta in self._meta_map.values():
            changed = False
            if not meta.doc_name and meta.doc_id in doc_name_map:
                meta.doc_name = doc_name_map[meta.doc_id]
                changed = True
            if not meta.doc_title and meta.doc_id in doc_title_map:
                meta.doc_title = doc_title_map[meta.doc_id]
                changed = True
            if not meta.source_file and meta.doc_id in source_file_map:
                meta.source_file = source_file_map[meta.doc_id]
                changed = True
            if changed:
                filled += 1

        if filled:
            print(f"  [ChunkStore] 已加载 {len(self._meta_map)} 条 chunk 元信息"
                  f"（回填 {filled} 条 doc_name/doc_title/source_file）")
        else:
            print(f"  [ChunkStore] 已加载 {len(self._meta_map)} 条 chunk 元信息")

    def load_from_meta_list(self, meta_list: List[ChunkMeta]) -> None:
        """从 ChunkMeta 列表加载（用于反序列化）"""
        self._meta_map.clear()
        self._chunk_id_list.clear()
        for meta in meta_list:
            self._meta_map[meta.chunk_id] = meta
            self._chunk_id_list.append(meta.chunk_id)
        print(f"  [ChunkStore] 已加载 {len(self._meta_map)} 条 chunk 元信息（反序列化）")

    # ============================================================
    # 查询 — 元信息（纯内存 O(1)）
    # ============================================================
    def get_meta(self, chunk_id: str) -> Optional[ChunkMeta]:
        """获取 chunk 元信息（纯内存，O(1)）"""
        return self._meta_map.get(chunk_id)

    def get_metas_batch(self, chunk_ids: List[str]) -> List[Optional[ChunkMeta]]:
        """批量获取元信息"""
        return [self._meta_map.get(cid) for cid in chunk_ids]

    def get_chunk_id(self, index: int) -> str:
        """下标 → chunk_id 映射（用于检索器 doc_idx → chunk_id 转换）"""
        if 0 <= index < len(self._chunk_id_list):
            return self._chunk_id_list[index]
        return ""

    def get_index(self, chunk_id: str) -> int:
        """chunk_id → 下标映射（用于兼容旧接口）"""
        try:
            return self._chunk_id_list.index(chunk_id)
        except ValueError:
            return -1

    @property
    def chunk_count(self) -> int:
        return len(self._meta_map)

    @property
    def chunk_ids(self) -> List[str]:
        return self._chunk_id_list[:]

    # ============================================================
    # 查询 — content（LRU → DB 透明回源）
    # ============================================================
    def get_content(self, chunk_id: str) -> str:
        """获取 content 全文（LRU 缓存 → DB 回源）"""
        # LRU 命中
        if chunk_id in self._lru:
            self._lru.move_to_end(chunk_id)
            return self._lru[chunk_id]

        # DB 回源
        content = self._fetch_content_from_db(chunk_id)
        self._put_lru(chunk_id, content)
        return content

    def get_content_batch(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        批量获取 content（先查 LRU，未命中的批量查 DB）。
        减少 DB 往返次数。
        """
        result: Dict[str, str] = {}
        missing: List[str] = []

        for cid in chunk_ids:
            if cid in self._lru:
                self._lru.move_to_end(cid)
                result[cid] = self._lru[cid]
            else:
                missing.append(cid)

        # 批量 DB 查询
        if missing and self._db:
            for cid in missing:
                content = self._fetch_content_from_db(cid)
                result[cid] = content
                self._put_lru(cid, content)
        elif missing:
            for cid in missing:
                result[cid] = ""

        return result

    def _fetch_content_from_db(self, chunk_id: str) -> str:
        """从 SQLite 数据库获取 content"""
        if not self._db:
            return ""
        row = self._db.get_chunk(chunk_id)
        if row:
            return row.get("content", "")
        return ""

    def _put_lru(self, chunk_id: str, content: str) -> None:
        """写入 LRU 缓存"""
        self._lru[chunk_id] = content
        self._lru.move_to_end(chunk_id)
        if len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)

    # ============================================================
    # 查询 — 完整数据（meta + content）
    # ============================================================
    def get_full(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取完整 chunk 数据（meta + content 合并）"""
        meta = self._meta_map.get(chunk_id)
        if not meta:
            return None
        content = self.get_content(chunk_id)
        return {**meta.to_dict(), "content": content}

    # ============================================================
    # 持久化（轻量元信息，不含 content）
    # ============================================================
    def save_meta(self, path: str) -> None:
        """持久化元信息到 pickle 文件（不含 content，体积小）"""
        meta_list = list(self._meta_map.values())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "meta_list": meta_list,
                "chunk_id_list": self._chunk_id_list,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [ChunkStore] 元信息已持久化: {Path(path).name} ({len(meta_list)} 条)")

    def load_meta(self, path: str) -> bool:
        """从 pickle 文件加载元信息，成功返回 True"""
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        meta_list = data["meta_list"]
        self._chunk_id_list = data["chunk_id_list"]
        self._meta_map = {m.chunk_id: m for m in meta_list}
        print(f"  [ChunkStore] 元信息已加载: {p.name} ({len(self._meta_map)} 条)")
        return True

    # ============================================================
    # LRU 缓存管理
    # ============================================================
    @property
    def lru_size(self) -> int:
        return len(self._lru)

    @property
    def lru_hit_rate(self) -> float:
        """LRU 命中率（需要外部统计命中/未命中次数）"""
        return 0.0  # TODO: 可加统计

    def clear_lru(self) -> None:
        """清空 LRU 缓存"""
        self._lru.clear()

    def preload_contents(self, chunk_ids: List[str]) -> None:
        """预加载 content 到 LRU（批量预热缓存）"""
        self.get_content_batch(chunk_ids)
