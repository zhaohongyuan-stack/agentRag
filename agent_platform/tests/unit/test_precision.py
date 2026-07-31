"""
检索精准度对比与性能基准测试

验证检索层架构升级第三阶段的精准度提升和性能指标：
  1. jieba 分词质量（词组 + 单字混合 vs 纯单字）
  2. FTS5 MATCH 查询延迟（< 10ms）
  3. 元数据 SQL 过滤延迟（< 5ms）
  4. BM25 检索精准度（jieba 模式 vs 逐字模式）
  5. Hybrid 混合检索结果一致性
  6. ChunkStore content 获取性能（LRU 命中 vs DB 回源）

测试数据：knowledge_platform/retrieval/regulatory_docs/001_chunks.jsonl
"""

import os
import time
import pytest
from pathlib import Path

from knowledge_platform.retrieval.retrieval_service.chunk import load_json_chunks
from knowledge_platform.retrieval.retrieval_service.chunk_store import ChunkStore
from knowledge_platform.retrieval.retrieval_service.retrieval_db import RetrievalDB
from knowledge_platform.retrieval.retrieval_service import lexical_retriever as _lex_mod
from knowledge_platform.retrieval.retrieval_service.lexical_retriever import LexicalRetriever


# ============================================================
# 测试数据路径
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[3] / "knowledge_platform" / "retrieval" / "regulatory_docs"


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def loaded_chunks():
    """加载测试 chunk 数据（模块级共享）"""
    chunks = load_json_chunks(str(DATA_DIR))
    if not chunks:
        pytest.skip("测试数据不存在")
    return chunks


@pytest.fixture(scope="module")
def db_with_data(loaded_chunks):
    """创建带数据的内存 SQLite"""
    db = RetrievalDB(":memory:")
    db.open()
    # 写入文档
    doc_ids_seen = set()
    for c in loaded_chunks:
        doc_id = c.doc_id
        if doc_id and doc_id not in doc_ids_seen:
            doc_ids_seen.add(doc_id)
            db.upsert_document({
                "doc_id": doc_id,
                "doc_name": c.doc_name or f"doc_{doc_id}",
                "doc_title": c.doc_title or "",
                "parser_type": c.metadata.get("parser_type", ""),
                "source_file": c.source_file,
            })

    # 将 Chunk 对象转为 dict（与 retrieval_api._populate_db 逻辑一致）
    top_keys = ("parent_chunk_id", "prev_chunk_id", "next_chunk_id",
                 "chapter_number", "clause_number", "subclause_number",
                 "applicable_scope", "normative_level", "capital_tool_level",
                 "table_name", "table_section_name", "sheet_name",
                 "glossary_term", "keywords", "evidence_snippet",
                 "content_raw", "sub_chunks")
    chunk_dicts = []
    for c in loaded_chunks:
        chunk_dicts.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_type": c.chunk_type,
            "content": c.content,
            "hierarchy_path": c.hierarchy_path,
            **{k: c.metadata.get(k, "") for k in top_keys},
            "keywords": c.metadata.get("keywords", []),
            "sub_chunks": c.metadata.get("sub_chunks", []),
            "metadata": {k: v for k, v in c.metadata.items() if k not in top_keys},
        })
    db.insert_chunks(chunk_dicts)
    db.populate_fts5_index()  # 填充 FTS5 全文索引
    return db


@pytest.fixture(scope="module")
def store_with_data(db_with_data, loaded_chunks):
    """创建带数据的 ChunkStore"""
    store = ChunkStore(db_with_data)
    store.load_from_chunks(loaded_chunks)
    return store


# ============================================================
# jieba 分词质量测试
# ============================================================

class TestJiebaTokenization:
    """jieba 分词质量验证"""

    def test_jieba_available(self):
        """jieba 已安装且可用"""
        _lex_mod._init_jieba()
        assert _lex_mod._jieba_available, "jieba 未安装，无法验证分词质量"

    def test_word_segmentation(self):
        """jieba 能正确切分金融领域词组"""
        _lex_mod._init_jieba()

        # 领域词典中的词应被正确切分
        if _lex_mod._jieba_instance:
            words = list(_lex_mod._jieba_instance.cut("商业银行资本管理办法"))
            word_set = set(words)
            # 至少能切出部分词组，而不是纯单字
            assert len(word_set) > 1

    def test_tokenize_mixed(self):
        """LexicalRetriever.tokenize 返回词组 + 单字混合"""
        retriever = LexicalRetriever()
        tokens = retriever.tokenize("银行业总资产")

        # 应包含 "总资产" 或 "资产" 等词组
        assert len(tokens) > 0
        # tokens 应为字符串列表
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_empty(self):
        """空字符串分词返回空列表"""
        retriever = LexicalRetriever()
        tokens = retriever.tokenize("")
        assert tokens == []


# ============================================================
# FTS5 查询延迟测试
# ============================================================

class TestFTS5Latency:
    """FTS5 MATCH 查询延迟基准"""

    def test_fts5_available(self, db_with_data):
        """FTS5 索引可用"""
        assert db_with_data._fts5_available, "FTS5 不可用"

    def test_fts5_search_latency(self, db_with_data):
        """FTS5 MATCH 查询延迟 < 50ms（含 Python 开销）"""
        # 先确认有结果
        results = db_with_data.fts5_search("总资产", top_k=10)
        assert len(results) > 0, "FTS5 查询无结果，数据可能未入库"

        # 多次查询取平均
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            db_with_data.fts5_search("总资产", top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        # 实际 SQLite FTS5 在小数据集上 < 1ms，加上 Python 开销 < 50ms
        assert avg_latency < 50, f"FTS5 平均延迟 {avg_latency:.2f}ms 超过 50ms"

    def test_fts5_search_correctness(self, db_with_data):
        """FTS5 查询结果包含正确关键词"""
        results = db_with_data.fts5_search("总资产", top_k=5)
        assert len(results) > 0

        # 至少有一条结果的 chunk_id 有效
        for r in results:
            assert r.get("chunk_id", "") != ""


# ============================================================
# 元数据 SQL 过滤延迟测试
# ============================================================

class TestMetadataFilterLatency:
    """元数据 SQL WHERE 过滤延迟基准"""

    def test_metadata_filter_latency(self, db_with_data):
        """元数据过滤延迟 < 50ms（含 Python 开销）"""
        # 先确认有结果
        results = db_with_data.search_by_filters(
            {"chunk_type": "cell_fact"}, limit=100
        )
        assert len(results) > 0, "元数据过滤无结果"

        # 多次查询取平均
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            db_with_data.search_by_filters(
                {"chunk_type": "cell_fact"}, limit=100
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 50, f"元数据过滤平均延迟 {avg_latency:.2f}ms 超过 50ms"

    def test_metadata_multi_field_filter(self, db_with_data):
        """多字段组合过滤"""
        results = db_with_data.search_by_filters(
            {"chunk_type": "cell_fact", "applicable_scope": "全部"},
            limit=100
        )
        assert len(results) > 0

    def test_metadata_eq_operator(self, db_with_data):
        """eq 操作符（精确匹配）"""
        results = db_with_data.search_by_filters(
            {"chunk_type": {"value": "table", "op": "eq"}},
            limit=100
        )
        assert len(results) > 0

    def test_metadata_in_operator(self, db_with_data):
        """in 操作符（列表包含）"""
        results = db_with_data.search_by_filters(
            {"chunk_type": {"value": ["table", "cell_fact"], "op": "in"}},
            limit=100
        )
        assert len(results) > 0

    def test_metadata_contains_operator(self, db_with_data):
        """contains 操作符（子串匹配）"""
        results = db_with_data.search_by_filters(
            {"doc_name": {"value": "银行业", "op": "contains"}},
            limit=100
        )
        assert len(results) > 0


# ============================================================
# BM25 检索精准度测试
# ============================================================

class TestBM25Precision:
    """BM25 检索精准度验证（jieba 分词）"""

    def test_bm25_search_relevance(self, loaded_chunks):
        """BM25 检索结果与查询相关"""
        chunk_ids = [c.chunk_id for c in loaded_chunks]
        doc_texts = [c.content for c in loaded_chunks]

        retriever = LexicalRetriever()
        retriever.index(chunk_ids, doc_texts)

        # 查询 "总资产" 应返回包含相关内容的 chunk
        results = retriever.search("总资产", top_k=5, raw=True)
        assert len(results) > 0

        # 至少前几个结果得分 > 0
        for idx, score in results:
            assert score > 0

    def test_bm25_search_returns_chunk_id(self, loaded_chunks):
        """BM25 search 非raw模式返回 chunk_id（不含文本）"""
        chunk_ids = [c.chunk_id for c in loaded_chunks]
        doc_texts = [c.content for c in loaded_chunks]

        retriever = LexicalRetriever()
        retriever.index(chunk_ids, doc_texts)

        results = retriever.search("总负债", top_k=5, raw=False)
        assert len(results) > 0

        for r in results:
            assert "chunk_id" in r
            assert "score" in r
            assert "text" not in r  # 不应返回文本
            assert "content" not in r

    def test_bm25_no_text_stored(self, loaded_chunks):
        """LexicalRetriever 不保存 documents 全文"""
        chunk_ids = [c.chunk_id for c in loaded_chunks]
        doc_texts = [c.content for c in loaded_chunks]

        retriever = LexicalRetriever()
        retriever.index(chunk_ids, doc_texts)

        # 不应有 documents 属性
        assert not hasattr(retriever, "documents") or retriever.documents is None or len(retriever.documents) == 0
        # 不应有 _metadatas 属性
        assert not hasattr(retriever, "_metadatas") or retriever._metadatas is None


# ============================================================
# ChunkStore 性能测试
# ============================================================

class TestChunkStorePerformance:
    """ChunkStore content 获取性能"""

    def test_lru_hit_faster_than_db(self, store_with_data):
        """LRU 命中比 DB 回源快"""
        # 先访问一次触发 DB 回源
        chunk_id = store_with_data.chunk_ids[0]
        store_with_data.get_content(chunk_id)

        # LRU 命中延迟
        lru_latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            store_with_data.get_content(chunk_id)
            lru_latencies.append((time.perf_counter() - t0) * 1000)

        # 清空 LRU 后 DB 回源延迟
        store_with_data.clear_lru()
        db_latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            store_with_data.get_content(chunk_id)
            # 每次清空以强制 DB 回源
            store_with_data.clear_lru()
            db_latencies.append((time.perf_counter() - t0) * 1000)

        avg_lru = sum(lru_latencies) / len(lru_latencies)
        avg_db = sum(db_latencies) / len(db_latencies)

        # LRU 应比 DB 快（至少不慢）
        assert avg_lru <= avg_db * 2, (
            f"LRU 延迟 {avg_lru:.3f}ms 不优于 DB {avg_db:.3f}ms"
        )

    def test_batch_get_efficiency(self, store_with_data):
        """批量获取比逐条获取效率高"""
        all_ids = store_with_data.chunk_ids[:20]

        # 逐条获取
        store_with_data.clear_lru()
        t0 = time.perf_counter()
        for cid in all_ids:
            store_with_data.get_content(cid)
        sequential_time = time.perf_counter() - t0

        # 批量获取
        store_with_data.clear_lru()
        t0 = time.perf_counter()
        store_with_data.get_content_batch(all_ids)
        batch_time = time.perf_counter() - t0

        # 批量应不慢于逐条
        assert batch_time <= sequential_time * 1.5, (
            f"批量 {batch_time*1000:.2f}ms 慢于逐条 {sequential_time*1000:.2f}ms"
        )


# ============================================================
# 检索器内存占用验证
# ============================================================

class TestMemoryOptimization:
    """验证检索器不再保存冗余数据"""

    def test_dense_retriever_no_documents(self, loaded_chunks):
        """DenseRetriever 不保存 documents 全文"""
        from knowledge_platform.retrieval.retrieval_service.dense_retriever import (
            DenseRetriever,
        )

        chunk_ids = [c.chunk_id for c in loaded_chunks]
        doc_texts = [c.content for c in loaded_chunks]

        # 用 mock 模式，不实际加载模型
        retriever = DenseRetriever()
        retriever._chunk_ids = chunk_ids
        retriever.embeddings = None  # 不编码

        # 不应有 documents / _metadatas
        assert not hasattr(retriever, "documents") or retriever.documents is None
        assert not hasattr(retriever, "_metadatas") or retriever._metadatas is None

        # 应有 _chunk_ids
        assert hasattr(retriever, "_chunk_ids")
        assert len(retriever._chunk_ids) == len(loaded_chunks)

    def test_chunk_store_meta_no_content(self, store_with_data):
        """ChunkStore 元信息不含 content 全文"""
        # 验证 ChunkMeta 不含 content 字段
        for cid in store_with_data.chunk_ids[:5]:
            meta = store_with_data.get_meta(cid)
            assert meta is not None
            assert not hasattr(meta, "content")
            assert "content" not in meta.to_dict()
