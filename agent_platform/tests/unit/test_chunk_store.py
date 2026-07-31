"""
ChunkStore 专项测试

验证检索层架构升级第一阶段核心组件 ChunkStore 的功能完整性：
  1. ChunkMeta 数据结构
  2. 元信息加载（Chunk 对象 + dict 兼容）
  3. doc_name / doc_title / source_file 跨 chunk 回填
  4. O(1) 元信息查询
  5. 下标 ↔ chunk_id 双向映射
  6. LRU content 缓存（命中 / 未命中 / 淘汰）
  7. DB 回源（content 从 SQLite 获取）
  8. 批量获取（get_content_batch / get_metas_batch）
  9. get_full（meta + content 合并）
  10. 持久化（save_meta / load_meta 往返一致性）
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from dataclasses import is_dataclass

import pytest

from knowledge_platform.retrieval.retrieval_service.chunk_store import (
    ChunkStore,
    ChunkMeta,
)
from knowledge_platform.retrieval.retrieval_service.retrieval_db import RetrievalDB


# ============================================================
# 测试数据工厂
# ============================================================

def _make_chunk_dicts():
    """构造测试 chunk dict 列表（模拟 Excel 解析结果，首个 chunk 有 doc_name）"""
    return [
        {
            "chunk_id": "test_001",
            "chunk_type": "sheet_summary",
            "doc_id": "001",
            "doc_name": "测试文档A",
            "doc_title": "测试文档A标题",
            "source_file": "test_a.xlsx",
            "hierarchy_path": "工作表/汇总",
            "content": "这是第一个 chunk 的内容，包含银行业总资产数据。",
            "metadata": {"sheet_name": "汇总", "keywords": ["总资产"]},
        },
        {
            "chunk_id": "test_002",
            "chunk_type": "table",
            "doc_id": "001",
            "doc_name": "",  # 空 doc_name，应被回填
            "doc_title": "",
            "source_file": "",
            "hierarchy_path": "工作表/表格",
            "content": "银行业总负债数据表格。",
            "metadata": {"sheet_name": "汇总", "table_name": "负债表"},
        },
        {
            "chunk_id": "test_003",
            "chunk_type": "cell_fact",
            "doc_id": "002",
            "doc_name": "测试文档B",
            "doc_title": "测试文档B标题",
            "source_file": "test_b.xlsx",
            "hierarchy_path": "工作表/单元格",
            "content": "资本充足率为 8%。",
            "metadata": {"cell": "B7", "value": "0.08"},
        },
        {
            "chunk_id": "test_004",
            "chunk_type": "cell_fact",
            "doc_id": "002",
            "doc_name": "",  # 空，应回填为 "测试文档B"
            "doc_title": "",
            "source_file": "",
            "hierarchy_path": "工作表/单元格2",
            "content": "核心一级资本充足率。",
            "metadata": {"cell": "C8"},
        },
    ]


def _make_db_with_chunks(chunk_dicts):
    """创建内存 SQLite 并写入 chunks，返回 RetrievalDB 实例"""
    db = RetrievalDB(":memory:")
    db.open()
    # 写入文档
    for cd in chunk_dicts:
        doc = {
            "doc_id": cd["doc_id"],
            "doc_name": cd.get("doc_name") or f"doc_{cd['doc_id']}",
            "doc_title": cd.get("doc_title", ""),
            "parser_type": "test",
            "source_file": cd.get("source_file", ""),
        }
        db.upsert_document(doc)
    # 写入 chunks
    db.insert_chunks(chunk_dicts)
    return db


# ============================================================
# ChunkMeta 数据结构测试
# ============================================================

class TestChunkMeta:
    """ChunkMeta 数据类测试"""

    def test_is_dataclass(self):
        """ChunkMeta 应为 dataclass"""
        assert is_dataclass(ChunkMeta)

    def test_default_values(self):
        """默认值正确"""
        meta = ChunkMeta()
        assert meta.chunk_id == ""
        assert meta.chunk_type == "clause"
        assert meta.doc_id == ""
        assert meta.doc_name == ""
        assert meta.metadata == {}

    def test_to_dict(self):
        """to_dict 返回完整字段"""
        meta = ChunkMeta(
            chunk_id="c1",
            chunk_type="clause",
            doc_id="d1",
            doc_name="文档",
            doc_title="标题",
            hierarchy_path="路径",
            source_file="file.xlsx",
            metadata={"key": "val"},
        )
        d = meta.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["chunk_type"] == "clause"
        assert d["doc_id"] == "d1"
        assert d["doc_name"] == "文档"
        assert d["metadata"] == {"key": "val"}
        # 不应包含 content
        assert "content" not in d


# ============================================================
# ChunkStore 加载与元信息查询
# ============================================================

class TestChunkStoreLoad:
    """ChunkStore 加载与元信息查询"""

    def test_load_from_chunk_dicts(self):
        """从 dict 列表加载元信息"""
        store = ChunkStore()
        chunks = _make_chunk_dicts()
        store.load_from_chunks(chunks)

        assert store.chunk_count == 4
        assert len(store.chunk_ids) == 4

    def test_doc_name_backfill(self):
        """跨 chunk 回填 doc_name / doc_title / source_file"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        # test_002 的 doc_name 应被回填为 "测试文档A"
        meta_002 = store.get_meta("test_002")
        assert meta_002 is not None
        assert meta_002.doc_name == "测试文档A"
        assert meta_002.doc_title == "测试文档A标题"
        assert meta_002.source_file == "test_a.xlsx"

        # test_004 的 doc_name 应被回填为 "测试文档B"
        meta_004 = store.get_meta("test_004")
        assert meta_004 is not None
        assert meta_004.doc_name == "测试文档B"

    def test_get_meta_o1(self):
        """get_meta 返回正确元信息（纯内存 O(1)）"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        meta = store.get_meta("test_001")
        assert meta is not None
        assert meta.chunk_type == "sheet_summary"
        assert meta.doc_id == "001"

    def test_get_meta_not_found(self):
        """查询不存在的 chunk_id 返回 None"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())
        assert store.get_meta("nonexistent") is None

    def test_get_metas_batch(self):
        """批量获取元信息"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        metas = store.get_metas_batch(["test_001", "test_003", "nonexistent"])
        assert len(metas) == 3
        assert metas[0] is not None
        assert metas[0].chunk_id == "test_001"
        assert metas[1] is not None
        assert metas[1].chunk_id == "test_003"
        assert metas[2] is None  # 不存在

    def test_chunk_id_order_preserved(self):
        """chunk_id 列表顺序与加载顺序一致"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        ids = store.chunk_ids
        assert ids == ["test_001", "test_002", "test_003", "test_004"]


# ============================================================
# 下标映射测试
# ============================================================

class TestChunkStoreIndexMapping:
    """下标 ↔ chunk_id 双向映射"""

    def test_get_chunk_id_by_index(self):
        """下标 → chunk_id"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        assert store.get_chunk_id(0) == "test_001"
        assert store.get_chunk_id(1) == "test_002"
        assert store.get_chunk_id(3) == "test_004"

    def test_get_chunk_id_out_of_range(self):
        """越界下标返回空字符串"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        assert store.get_chunk_id(-1) == ""
        assert store.get_chunk_id(100) == ""

    def test_get_index_by_chunk_id(self):
        """chunk_id → 下标"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        assert store.get_index("test_001") == 0
        assert store.get_index("test_004") == 3

    def test_get_index_not_found(self):
        """不存在的 chunk_id 返回 -1"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())
        assert store.get_index("nonexistent") == -1

    def test_round_trip_mapping(self):
        """下标 ↔ chunk_id 往返一致性"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        for i in range(store.chunk_count):
            cid = store.get_chunk_id(i)
            assert store.get_index(cid) == i


# ============================================================
# LRU content 缓存测试
# ============================================================

class TestChunkStoreLRU:
    """LRU content 缓存行为"""

    def test_get_content_from_db(self):
        """content 从 DB 回源获取"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store.load_from_chunks(chunks)

        content = store.get_content("test_001")
        assert "银行业总资产" in content

    def test_get_content_no_db(self):
        """无 DB 时返回空字符串"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())

        content = store.get_content("test_001")
        assert content == ""

    def test_lru_cache_hit(self):
        """第二次访问命中 LRU 缓存"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store.load_from_chunks(chunks)

        # 第一次访问 → DB 回源
        content1 = store.get_content("test_001")
        assert store.lru_size == 1

        # 第二次访问 → LRU 命中
        content2 = store.get_content("test_001")
        assert content1 == content2
        assert store.lru_size == 1  # 不会增加

    def test_lru_eviction(self):
        """LRU 超过上限时淘汰最久未访问的条目"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store._lru_max = 2  # 设小上限便于测试
        store.load_from_chunks(chunks)

        # 访问 3 个不同的 content（只有 4 个 chunk）
        store.get_content("test_001")
        store.get_content("test_002")
        store.get_content("test_003")

        # LRU 应只保留最近 2 个
        assert store.lru_size == 2
        assert "test_001" not in store._lru  # 被淘汰
        assert "test_002" in store._lru
        assert "test_003" in store._lru

    def test_lru_move_to_end_on_access(self):
        """访问 LRU 条目时移动到末尾（最近使用）"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store._lru_max = 2
        store.load_from_chunks(chunks)

        store.get_content("test_001")
        store.get_content("test_002")
        # 再次访问 test_001，使其成为最近使用
        store.get_content("test_001")
        # 访问 test_003，应淘汰 test_002（最久未使用）
        store.get_content("test_003")

        assert "test_001" in store._lru  # 仍在缓存
        assert "test_002" not in store._lru  # 被淘汰

    def test_clear_lru(self):
        """清空 LRU 缓存"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store.load_from_chunks(chunks)

        store.get_content("test_001")
        assert store.lru_size > 0

        store.clear_lru()
        assert store.lru_size == 0

    def test_get_content_batch(self):
        """批量获取 content"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store.load_from_chunks(chunks)

        result = store.get_content_batch(["test_001", "test_002", "nonexistent"])
        assert len(result) == 3
        assert "银行业总资产" in result["test_001"]
        assert "银行业总负债" in result["test_002"]
        assert result["nonexistent"] == ""

    def test_preload_contents(self):
        """预加载 content 到 LRU"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store.load_from_chunks(chunks)

        store.preload_contents(["test_001", "test_002"])
        assert store.lru_size == 2


# ============================================================
# get_full 完整数据测试
# ============================================================

class TestChunkStoreGetFull:
    """get_full — meta + content 合并"""

    def test_get_full(self):
        """获取完整 chunk 数据"""
        chunks = _make_chunk_dicts()
        db = _make_db_with_chunks(chunks)
        store = ChunkStore(db)
        store.load_from_chunks(chunks)

        full = store.get_full("test_001")
        assert full is not None
        assert full["chunk_id"] == "test_001"
        assert full["chunk_type"] == "sheet_summary"
        assert full["doc_id"] == "001"
        assert "银行业总资产" in full["content"]

    def test_get_full_not_found(self):
        """不存在的 chunk_id 返回 None"""
        store = ChunkStore()
        store.load_from_chunks(_make_chunk_dicts())
        assert store.get_full("nonexistent") is None


# ============================================================
# 持久化测试
# ============================================================

class TestChunkStorePersistence:
    """save_meta / load_meta 持久化"""

    def test_save_load_round_trip(self, tmp_path):
        """持久化往返一致性"""
        chunks = _make_chunk_dicts()
        store = ChunkStore()
        store.load_from_chunks(chunks)

        meta_path = str(tmp_path / "chunk_store_meta.pkl")
        store.save_meta(meta_path)

        # 新建 store 加载
        store2 = ChunkStore()
        assert store2.load_meta(meta_path) is True

        # 验证元信息一致
        assert store2.chunk_count == store.chunk_count
        assert store2.chunk_ids == store.chunk_ids

        meta_orig = store.get_meta("test_001")
        meta_loaded = store2.get_meta("test_001")
        assert meta_loaded.chunk_id == meta_orig.chunk_id
        assert meta_loaded.doc_id == meta_orig.doc_id
        assert meta_loaded.doc_name == meta_orig.doc_name

    def test_load_meta_file_not_found(self):
        """加载不存在的文件返回 False"""
        store = ChunkStore()
        assert store.load_meta("/nonexistent/path/file.pkl") is False

    def test_meta_pickle_no_content(self):
        """持久化文件不包含 content 全文"""
        chunks = _make_chunk_dicts()
        store = ChunkStore()
        store.load_from_chunks(chunks)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            meta_path = f.name

        try:
            store.save_meta(meta_path)
            with open(meta_path, "rb") as f:
                data = pickle.load(f)

            # 验证结构
            assert "meta_list" in data
            assert "chunk_id_list" in data

            # 每个 ChunkMeta 不应包含 content 字段
            for meta in data["meta_list"]:
                assert not hasattr(meta, "content")
        finally:
            os.unlink(meta_path)


# ============================================================
# Chunk 对象兼容性测试
# ============================================================

class TestChunkStoreObjectCompat:
    """兼容 Chunk 对象（非 dict）输入"""

    def test_load_from_objects(self):
        """从模拟 Chunk 对象列表加载"""

        class MockChunk:
            """模拟 Chunk 对象"""
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        chunks = [
            MockChunk(
                chunk_id="obj_001",
                chunk_type="clause",
                doc_id="100",
                doc_name="对象文档",
                doc_title="对象标题",
                hierarchy_path="路径",
                source_file="obj.xlsx",
                metadata={"key": "val"},
            ),
            MockChunk(
                chunk_id="obj_002",
                chunk_type="clause",
                doc_id="100",
                doc_name="",  # 空，应回填
                doc_title="",
                hierarchy_path="路径2",
                source_file="",
                metadata={},
            ),
        ]

        store = ChunkStore()
        store.load_from_chunks(chunks)

        assert store.chunk_count == 2
        meta_002 = store.get_meta("obj_002")
        assert meta_002.doc_name == "对象文档"  # 回填成功
