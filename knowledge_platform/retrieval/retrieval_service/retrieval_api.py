"""
Retrieval API v1 — 统一检索接口，组合 7 大独立检索器

检索器拆分：
  lexical_retriever     → BM25 关键词检索
  dense_retriever       → Dense 向量语义检索
  exact_retriever       → 精确/子串/正则匹配
  metadata_retriever    → 元数据字段过滤
  relation_retriever    → 文档 & Chunk 关系查询
  neighborhood_retriever → 邻域图查询（父子、前后、同级、上下文）
  table_retriever       → 表格行列检索

集成：
  - RRF 融合（lexical + dense → 混合排序）
  - Cross-Encoder 精排（可选）
  - 关系数据库（SQLite 存储）
  - LLM 上下文格式化
  - 向量文件导出
"""

import hashlib
import heapq
import json
import numpy as np
from collections import Counter  # 保留：可能被外部代码引用
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from .chunk import Chunk, load_json_chunks, CHUNK_TYPE_ICONS
from .chunk_store import ChunkStore, ChunkMeta
from .retrieval_db import RetrievalDB
from .utils import modelscope_download

from .lexical_retriever import LexicalRetriever
from .dense_retriever import DenseRetriever
from .exact_retriever import ExactRetriever
from .metadata_retriever import MetadataRetriever
from .relation_retriever import RelationRetriever
from .neighborhood_retriever import NeighborhoodRetriever
from .table_retriever import TableRetriever
from .retrieval_request import (
    RetrievalRequest, RetrievalHit, RetrievalStrategy, RerankMode,
)
from .siliconflow_client import SiliconFlowReranker, SiliconFlowEmbedding

# ── 全局缓存版本号：索引结构或分词算法变更时递增，自动失效旧缓存 ──
# v2_jieba: jieba 分词 + 本地嵌入模型
# v3_api_embed: 切换到硅基流动 API 嵌入（向量维度 512 → 1024）
# v4_fts5: ExactRetriever 改为 DB/FTS5 驱动
# v5_metadata_sql: MetadataRetriever 改为 SQL WHERE 驱动，消除 _records 内存存储
CACHE_VERSION = "v5_metadata_sql"


class RetrievalAPI:
    """Retrieval API v1 — 7 大检索器 + RRF 混合融合"""

    def __init__(self,
                 embed_model: str = "BAAI/bge-small-zh-v1.5",
                 use_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 reranker_api_key: Optional[str] = None,
                 use_embed_api: bool = False,
                 embed_api_key: Optional[str] = None,
                 embed_api_model: Optional[str] = None,
                 db_path: Optional[str] = None):
        """
        参数：
          embed_model:       嵌入模型名（本地 ModelScope 模型），use_embed_api=False 时使用
          use_reranker:      是否启用重排序
          reranker_model:    重排序模型名（默认硅基流动 BAAI/bge-reranker-v2-m3）
          reranker_api_key:  硅基流动重排序 API Key（None 则从环境变量读取）
          use_embed_api:     是否使用硅基流动 API 嵌入（True 时不再加载本地模型）
          embed_api_key:     硅基流动嵌入 API Key（None 则从环境变量 SILICONFLOW_EMBED_API_KEY 读取）
          embed_api_model:   API 嵌入模型名（None 则从环境变量读取，默认 BAAI/bge-large-zh-v1.5）
          db_path:           SQLite 数据库路径
        """
        self._embed_model = embed_model
        self._use_reranker = use_reranker
        self._reranker_model = reranker_model
        self._reranker_api_key = reranker_api_key
        self._reranker = None

        # ── 嵌入 API 配置 ──
        self._use_embed_api = use_embed_api
        self._embed_api_key = embed_api_key
        self._embed_api_model = embed_api_model

        # 7 大独立检索器（load 后填充，也可外部注入用于测试）
        self.lexical: Optional[LexicalRetriever] = None
        self.dense: Optional[DenseRetriever] = None
        self.exact: Optional[ExactRetriever] = None
        self.metadata: Optional[MetadataRetriever] = None
        self.relation: Optional[RelationRetriever] = None
        self.neighborhood: Optional[NeighborhoodRetriever] = None
        self.table: Optional[TableRetriever] = None

        # ── 顶层统一数据管理（分层缓存：内存元信息 + LRU content + DB 回源）──
        self._store = ChunkStore()
        self._db = RetrievalDB(db_path or ":memory:")
        self._loaded = False

    # ============================================================
    # 依赖注入 — 测试时替换任意检索器
    # ============================================================
    def inject(self,
               lexical: Optional[Any] = None,
               dense: Optional[Any] = None,
               exact: Optional[Any] = None,
               metadata: Optional[Any] = None,
               relation: Optional[Any] = None,
               neighborhood: Optional[Any] = None,
               table: Optional[Any] = None) -> "RetrievalAPI":
        """
        注入检索器实例（用于测试/自定义）。

        传入 None 表示保留现有实例，不替换。

        用法：
            from retrieval_service.retrieval_interface import MockTextRetriever

            api = RetrievalAPI()
            api.inject(
                lexical=MockTextRetriever([(0, 12.34), (1, 8.90)]),
                dense=MockTextRetriever([(1, 0.95), (0, 0.87)]),
            )
            hits = api.search_request(RetrievalRequest(query="test"))
        """
        if lexical is not None:
            self.lexical = lexical
        if dense is not None:
            self.dense = dense
        if exact is not None:
            self.exact = exact
        if metadata is not None:
            self.metadata = metadata
        if relation is not None:
            self.relation = relation
        if neighborhood is not None:
            self.neighborhood = neighborhood
        if table is not None:
            self.table = table
        return self

    # ============================================================
    # 加载（构建全部 7 个检索器 + RRF 融合所需索引）
    # ============================================================
    def load(self, source: Union[str, Path],
             populate_db: bool = True,
             db_path: Optional[str] = None) -> "RetrievalAPI":
        source = str(source)
        source_path = Path(source)
        cache_dir = (source_path / ".cache") if source_path.is_dir() else (source_path.parent / ".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── 默认使用持久化 SQLite 文件（放在 .cache 目录下）──
        if db_path:
            self._db = RetrievalDB(db_path)
        else:
            self._db = RetrievalDB(str(cache_dir / "retrieval.db"))

        # ChunkStore 绑定 DB（用于 content 回源）
        self._store = ChunkStore(self._db)

        print("=" * 60)
        print("  Retrieval API v1 — 加载全部检索器 ...")
        print("=" * 60)

        # ── 计算数据指纹（用于判断缓存是否有效）──
        # 先快速扫描文件列表计算文件级 hash（不读内容，速度快）
        jsonl_files = sorted(
            list(source_path.glob("*.jsonl")) if source_path.is_dir()
            else [source_path]
        )
        file_sig = "|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in jsonl_files)
        data_hash = hashlib.sha256(file_sig.encode()).hexdigest()[:16]

        manifest_path = cache_dir / f"manifest_{data_hash}.json"

        # ════════════════════════════════════════════════════════
        # 快速路径：缓存命中，全部从磁盘加载
        # ════════════════════════════════════════════════════════
        if manifest_path.exists():
            print(f"\n  [缓存命中] data_hash={data_hash}")
            if self._load_from_cache(cache_dir, data_hash, populate_db):
                self._loaded = True
                doc_count = len(set(m.doc_id for m in self._store._meta_map.values()))
                print(f"\n  Retrieval API v1 就绪（缓存启动）— "
                      f"{doc_count} 文档, "
                      f"{self._store.chunk_count} Chunks, "
                      f"{len(self.table.list_tables())} 表格")
                print("=" * 60 + "\n")
                return self
            else:
                print("  [缓存加载失败，回退到全量构建]")

        # ════════════════════════════════════════════════════════
        # 慢速路径：从 JSONL 全量构建 + 持久化
        # ════════════════════════════════════════════════════════
        print(f"\n  [全量构建] data_hash={data_hash}")

        # ── 1. 加载 JSON chunks（临时变量，构建完毕后释放）──
        print("\n[1/8] 加载 JSON chunk 文件 ...")
        chunks = load_json_chunks(source)
        if not chunks:
            print("[警告] 未找到任何 chunk")
            return self
        chunk_ids = [c.chunk_id for c in chunks]
        doc_texts = [c.content for c in chunks]
        print(f"  共 {len(chunks)} 个 chunks")

        # ── 1b. 初始化 ChunkStore（仅元信息常驻内存，不含 content 全文）──
        print("\n[1b/8] 初始化 ChunkStore ...")
        self._store.load_from_chunks(chunks)

        # ── 2. Lexical（BM25）—— 新接口：chunk_ids + documents ──
        print("\n[2/8] 构建 Lexical (BM25) 索引 ...")
        self.lexical = LexicalRetriever()
        self.lexical.index(chunk_ids, doc_texts)

        # ── 3. Dense + 缓存 ──
        print("\n[3/8] 构建 Dense (向量) 索引 ...")
        self.dense = DenseRetriever(
            self._embed_model,
            use_api=self._use_embed_api,
            api_key=self._embed_api_key,
            api_model=self._embed_api_model,
        )
        text_hash = hashlib.sha256("\n".join(doc_texts).encode()).hexdigest()[:16]
        # 向量缓存文件名包含模型标识，避免不同模型维度冲突
        emb_model_tag = "api" if self._use_embed_api else "local"
        emb_cache = cache_dir / f"embeddings_{emb_model_tag}_{text_hash}.npy"

        if emb_cache.exists():
            print(f"  [向量缓存命中] {emb_cache.name}")
            self.dense._chunk_ids = chunk_ids
            self.dense.embeddings = np.load(str(emb_cache))
        else:
            for old in cache_dir.glob("embeddings_*.npy"):
                old.unlink()
            self.dense.index(chunk_ids, doc_texts)
            np.save(str(emb_cache), self.dense.embeddings)
            print(f"  [向量缓存已保存] {emb_cache.name}")

        # ── 4. Exact（DB 驱动，需先打开 DB）──
        print("\n[4/8] 构建 Exact 索引 (DB 驱动) ...")
        if populate_db:
            self._db.open()
            self._populate_db(chunks)
            # 填充 FTS5 全文索引
            self._db.populate_fts5_index()
        self.exact = ExactRetriever(self._db)
        self.exact.index(chunk_ids, doc_texts)

        # ── 5. Metadata（DB 驱动，SQL WHERE 过滤）──
        print("\n[5/8] 构建 Metadata 索引 (DB 驱动) ...")
        self.metadata = MetadataRetriever(self._db)
        self.metadata.index(chunks)

        # ── 6. Relation / Neighborhood（DB 已在 step 4 打开）──
        print(f"\n[6/8] 初始化关系检索器 ...")
        self.relation = RelationRetriever(self._db)
        self.neighborhood = NeighborhoodRetriever(self._db)
        print(f"  文档: {len(self.relation.list_documents())}, Chunks: {self.relation.count_chunks()}")

        # ── 7. Table ──
        print(f"\n[7/8] 构建 Table 索引 ...")
        self.table = TableRetriever()
        self.table.index(chunks)

        # ── 8. Cross-Encoder (optional，通过硅基流动 API) ──
        if self._use_reranker:
            print(f"\n[8/8] 初始化重排序 API: {self._reranker_model} ...")
            self._reranker = SiliconFlowReranker(
                api_key=self._reranker_api_key,
                model=self._reranker_model,
            )
        else:
            print(f"\n[8/8] 跳过重排序（未启用）")

        # ── 持久化全部索引 ──
        print(f"\n  持久化全部索引到缓存 ...")
        self._save_all_indexes(cache_dir, data_hash, text_hash)

        # ── 释放临时 chunks 引用（原文已入库，后续由 ChunkStore 管理）──
        del chunks
        del doc_texts

        self._loaded = True
        doc_count = len(set(m.doc_id for m in self._store._meta_map.values()))
        print(f"\n  Retrieval API v1 就绪（全量构建）— "
              f"{doc_count} 文档, "
              f"{self._store.chunk_count} Chunks, "
              f"{len(self.table.list_tables())} 表格")
        print("=" * 60 + "\n")
        return self

    def _save_all_indexes(self, cache_dir: Path, data_hash: str, text_hash: str) -> None:
        """将全部索引持久化到缓存目录"""
        # 清理旧 manifest
        for old_manifest in cache_dir.glob("manifest_*.json"):
            old_manifest.unlink()

        # 各检索器索引
        self.lexical.save_index(str(cache_dir / f"bm25_{data_hash}.pkl"))
        self.exact.save_index(str(cache_dir / f"exact_{data_hash}.pkl"))
        self.metadata.save_index(str(cache_dir / f"metadata_{data_hash}.pkl"))
        self.table.save_index(str(cache_dir / f"table_{data_hash}.pkl"))
        self.dense.save_meta(str(cache_dir / f"dense_meta_{text_hash}.pkl"))

        # ChunkStore 元信息（轻量，不含 content 全文）
        self._store.save_meta(str(cache_dir / f"chunk_store_meta_{data_hash}.pkl"))

        # manifest
        emb_model_tag = "api" if self._use_embed_api else "local"
        manifest = {
            "cache_version": CACHE_VERSION,
            "data_hash": data_hash,
            "text_hash": text_hash,
            "chunk_count": self._store.chunk_count,
            "files": {
                "chunk_store_meta": f"chunk_store_meta_{data_hash}.pkl",
                "bm25": f"bm25_{data_hash}.pkl",
                "exact": f"exact_{data_hash}.pkl",
                "metadata": f"metadata_{data_hash}.pkl",
                "table": f"table_{data_hash}.pkl",
                "dense_meta": f"dense_meta_{text_hash}.pkl",
                "dense_emb": f"embeddings_{emb_model_tag}_{text_hash}.npy",
            },
        }
        with open(cache_dir / f"manifest_{data_hash}.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  [持久化完成] manifest_{data_hash}.json")

    def _load_from_cache(self, cache_dir: Path, data_hash: str, populate_db: bool) -> bool:
        """从缓存加载全部索引，成功返回 True"""
        manifest_path = cache_dir / f"manifest_{data_hash}.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        # ── 全局缓存版本校验 ──
        cached_version = manifest.get("cache_version", "v1")
        if cached_version != CACHE_VERSION:
            print(f"  [缓存版本不匹配] {cached_version} → {CACHE_VERSION}，需全量重建")
            # 清理旧缓存文件（含 .pkl 索引、.npy 向量、manifest）
            for old in cache_dir.glob("*.pkl"):
                old.unlink()
            for old in cache_dir.glob("*.npy"):
                old.unlink()
            for old in cache_dir.glob("manifest_*.json"):
                old.unlink()
            return False

        files = manifest["files"]
        text_hash = manifest["text_hash"]

        try:
            # 1. ChunkStore 元信息（替代旧的 chunks pickle）
            print("  [1/6] 加载 ChunkStore 元信息 ...")
            if not self._store.load_meta(str(cache_dir / files["chunk_store_meta"])):
                print("  [ChunkStore 元信息加载失败]")
                return False

            # 2. BM25
            print("  [2/6] 加载 BM25 索引 ...")
            self.lexical = LexicalRetriever()
            if not self.lexical.load_index(str(cache_dir / files["bm25"])):
                return False

            # 3. Dense（向量 + chunk_ids）
            print("  [3/6] 加载 Dense 向量索引 ...")
            self.dense = DenseRetriever(
                self._embed_model,
                use_api=self._use_embed_api,
                api_key=self._embed_api_key,
                api_model=self._embed_api_model,
            )
            if not self.dense.load_meta(
                str(cache_dir / files["dense_meta"]),
                str(cache_dir / files["dense_emb"]),
            ):
                return False

            # 4. Exact（DB 驱动 — 需先打开 DB）
            print("  [4/6] 加载 Exact 索引 ...")
            if populate_db:
                self._db.open()
                # SQLite 已持久化，检查是否有数据
                chunk_count = self._db.count_chunks()
                if chunk_count == 0:
                    print("  [DB 为空，缓存不可用，需全量重建]")
                    return False
            self.exact = ExactRetriever(self._db)
            if not self.exact.load_index(str(cache_dir / files["exact"])):
                return False

            # 5. Metadata（DB 驱动）
            print("  [5/6] 加载 Metadata 索引 ...")
            self.metadata = MetadataRetriever(self._db)
            if not self.metadata.load_index(str(cache_dir / files["metadata"])):
                return False

            # 6. Relation + Neighborhood + Table（DB 已在 step 4 打开）
            print("  [6/6] 加载 Relation + Table ...")
            self.relation = RelationRetriever(self._db)
            self.neighborhood = NeighborhoodRetriever(self._db)

            self.table = TableRetriever()
            if not self.table.load_index(str(cache_dir / files["table"])):
                return False

            # Cross-Encoder（可选，通过硅基流动 API，按需初始化）
            if self._use_reranker:
                self._reranker = SiliconFlowReranker(
                    api_key=self._reranker_api_key,
                    model=self._reranker_model,
                )

            return True

        except Exception as e:
            print(f"  [缓存加载异常: {e}]")
            return False

    # ============================================================
    # search — 便捷入口（薄包装，委托给 search_request）
    # ============================================================
    def search(self, query: str, top_k: int = 5,
               bm25_k: int = 20, vector_k: int = 20,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        BM25 + Dense → RRF 融合（便捷入口，委托给 search_request）。

        直接返回 dict 列表以保持向后兼容。
        新代码推荐直接使用 search_request() 获取强类型的 RetrievalHit。
        """
        req = RetrievalRequest(
            query=query, top_k=top_k,
            bm25_k=bm25_k, vector_k=vector_k,
            filters=filters or {},
        )
        hits = self.search_request(req)
        return [h.to_dict() for h in hits]

    # ============================================================
    # 结构化检索入口 — RetrievalRequest → List[RetrievalHit]
    # ============================================================
    def search_request(self, req: RetrievalRequest) -> List[RetrievalHit]:
        """
        接受 RetrievalRequest，返回 List[RetrievalHit]。

        与 search() 不同之处：
          - 输入/输出均为强类型数据结构，可序列化、可复现
          - 每条命中自带 trace（检索链路）和 scores_detail（分项得分）
          - 支持 expand_context 自动扩展邻域

        这是推荐的外部调用入口 — 稳定、可解释、可追溯。
        """
        if not self._store.chunk_count:
            return []

        # ── Phase 0: 元数据过滤 ──
        # metadata.search() 返回 chunk_id 列表，转换为 doc_idx 集合供检索器过滤
        allowed: Optional[set] = None
        filters_applied: Dict[str, Any] = {}
        if req.filters and self.metadata:
            allowed = self.metadata.get_allowed_indices(
                req.filters, self._store.chunk_ids
            )
            filters_applied = dict(req.filters)
            if not allowed:
                return []

        # ── Phase 1: 按策略检索 ──
        strategy = req.strategy

        if strategy == RetrievalStrategy.BM25:
            raw = self.lexical.search(req.query, top_k=req.top_k, raw=True) if self.lexical else []
            hits = self._build_hits(req, raw, matched_by="bm25", filters_applied=filters_applied)

        elif strategy == RetrievalStrategy.DENSE:
            raw = self.dense.search(req.query, top_k=req.top_k, raw=True) if self.dense else []
            hits = self._build_hits(req, raw, matched_by="dense", filters_applied=filters_applied)

        elif strategy == RetrievalStrategy.EXACT:
            raw = self.exact.search(req.query, top_k=req.top_k, mode=req.exact_mode) if self.exact else []
            hits = self._build_hits_exact(req, raw, filters_applied=filters_applied)

        elif strategy == RetrievalStrategy.METADATA:
            chunk_ids_result = self.metadata.search(req.filters, limit=req.top_k) if self.metadata else []
            hits = self._build_hits_metadata(req, chunk_ids_result, filters_applied=filters_applied)

        elif strategy == RetrievalStrategy.HYBRID:
            # ── BM25 + Dense → RRF 融合（内联，同时捕获每路原始得分用于 trace）──
            bm25_raw = self.lexical.search(req.query, top_k=req.bm25_k, raw=True) if self.lexical else []
            dense_raw = self.dense.search(req.query, top_k=req.vector_k, raw=True) if self.dense else []

            # 构建 per-doc_idx 的分数映射（用于后续 trace 填充）
            bm25_map: Dict[int, Tuple[int, float]] = {}  # doc_idx → (rank, score)
            dense_map: Dict[int, Tuple[int, float]] = {}
            for rank, (doc_idx, score) in enumerate(bm25_raw, 1):
                if doc_idx not in bm25_map:
                    bm25_map[doc_idx] = (rank, score)
            for rank, (doc_idx, score) in enumerate(dense_raw, 1):
                if doc_idx not in dense_map:
                    dense_map[doc_idx] = (rank, score)

            # RRF 融合
            rrf_scores: Dict[int, float] = {}
            rrf_k = req.rrf_k
            for rank, (doc_idx, _) in enumerate(bm25_raw):
                if allowed is None or doc_idx in allowed:
                    rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank + 1)
            for rank, (doc_idx, _) in enumerate(dense_raw):
                if allowed is None or doc_idx in allowed:
                    rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (rrf_k + rank + 1)

            sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            # Phase 2: 重排序精排（可选，通过硅基流动 API）
            if req.rerank == RerankMode.CROSS_ENC and self._reranker and len(sorted_candidates) > req.top_k:
                rerank_candidates = sorted_candidates[:req.rerank_k]
                candidate_indices = [idx for idx, _ in rerank_candidates]
                # ── 通过 ChunkStore 获取 content（LRU → DB 透明回源）──
                candidate_chunk_ids = [self._store.get_chunk_id(idx) for idx in candidate_indices]
                contents = self._store.get_content_batch(candidate_chunk_ids)
                doc_texts = [contents[cid] for cid in candidate_chunk_ids]

                # ── 调用硅基流动 rerank API ──
                # 返回 [{"index": int, "score": float}, ...] 按 score 降序
                rerank_results = self._reranker.rerank(req.query, doc_texts, top_n=req.top_k)
                scored = [(candidate_indices[r["index"]], r["score"]) for r in rerank_results]

                hits = []
                for rank, (doc_idx, rerank_score) in enumerate(scored[:req.top_k], 1):
                    rrf_score = rrf_scores.get(doc_idx, 0)
                    bm25_entry = bm25_map.get(doc_idx)
                    dense_entry = dense_map.get(doc_idx)
                    trace = self._build_trace(req, {
                        "bm25_rank": bm25_entry[0] if bm25_entry else None,
                        "bm25_score": round(bm25_entry[1], 4) if bm25_entry else None,
                        "dense_rank": dense_entry[0] if dense_entry else None,
                        "dense_score": round(dense_entry[1], 4) if dense_entry else None,
                        "rrf_score": round(rrf_score, 6),
                        "rerank_input_position": rank,
                        "rerank_score": round(float(rerank_score), 4),
                        "filters_applied": filters_applied,
                    })
                    scores_detail = {
                        "bm25": round(bm25_entry[1], 4) if bm25_entry else 0,
                        "dense": round(dense_entry[1], 4) if dense_entry else 0,
                        "rrf": round(rrf_score, 6),
                        "rerank": round(float(rerank_score), 4),
                    }
                    matched_by = [ch for ch in ["bm25", "dense"]
                                  if (ch == "bm25" and bm25_entry) or (ch == "dense" and dense_entry)]
                    hit = self._build_single_hit(doc_idx, req, rank, scores_detail, matched_by, trace)
                    hits.append(hit)
            else:
                # 无精排，直接取 RRF top_k
                hits = []
                for rank, (doc_idx, rrf_score) in enumerate(sorted_candidates[:req.top_k], 1):
                    bm25_entry = bm25_map.get(doc_idx)
                    dense_entry = dense_map.get(doc_idx)
                    trace = self._build_trace(req, {
                        "bm25_rank": bm25_entry[0] if bm25_entry else None,
                        "bm25_score": round(bm25_entry[1], 4) if bm25_entry else None,
                        "dense_rank": dense_entry[0] if dense_entry else None,
                        "dense_score": round(dense_entry[1], 4) if dense_entry else None,
                        "rrf_score": round(rrf_score, 6),
                        "filters_applied": filters_applied,
                    })
                    scores_detail = {
                        "bm25": round(bm25_entry[1], 4) if bm25_entry else 0,
                        "dense": round(dense_entry[1], 4) if dense_entry else 0,
                        "rrf": round(rrf_score, 6),
                    }
                    matched_by = [ch for ch in ["bm25", "dense"]
                                  if (ch == "bm25" and bm25_entry) or (ch == "dense" and dense_entry)]
                    hit = self._build_single_hit(doc_idx, req, rank, scores_detail, matched_by, trace)
                    hits.append(hit)

        else:
            # strategy == RELATION / TABLE — 委托给对应检索器
            hits = self._search_structured(req)

        # ── Phase 3: 可选邻域扩展 ──
        if req.expand_context and self.neighborhood:
            hits = self._expand_context_for_hits(req, hits)

        return hits

    # ============================================================
    # search_request() 内部构建方法
    # ============================================================
    def _build_hits(self, req: RetrievalRequest,
                    raw: List[Tuple[int, float]],
                    matched_by: str,
                    filters_applied: Dict[str, Any]) -> List[RetrievalHit]:
        """从 (doc_idx, score) 原始列表构建 RetrievalHit 列表"""
        hits = []
        for rank, (doc_idx, score) in enumerate(raw, 1):
            trace = self._build_trace(req, {
                f"{matched_by}_rank": rank,
                f"{matched_by}_score": round(score, 4),
                "filters_applied": filters_applied,
            })
            hit = self._build_single_hit(
                doc_idx, req, rank,
                {matched_by: round(score, 4)},
                [matched_by], trace,
            )
            hits.append(hit)
        return hits[:req.top_k]

    def _build_hits_exact(self, req: RetrievalRequest,
                          raw: List[Dict[str, Any]],
                          filters_applied: Dict[str, Any]) -> List[RetrievalHit]:
        """从 exact 检索结果构建 RetrievalHit 列表（通过 ChunkStore 补全字段）"""
        hits = []
        for rank, r in enumerate(raw[:req.top_k], 1):
            trace = self._build_trace(req, {
                "exact_mode": req.exact_mode,
                "match_position": r.get("match_pos", -1),
                "filters_applied": filters_applied,
            })
            # exact 检索结果只有 index/chunk_id/text/score/match_pos
            # 通过 ChunkStore 补全 doc_id, doc_name, chunk_type 等字段
            chunk_id = r.get("chunk_id", "")
            store_meta = self._store.get_meta(chunk_id) if chunk_id else None

            if store_meta:
                content = self._store.get_content(chunk_id)
                hit = RetrievalHit(
                    chunk_id=chunk_id,
                    chunk_type=store_meta.chunk_type,
                    doc_id=store_meta.doc_id,
                    doc_name=store_meta.doc_name,
                    doc_title=store_meta.doc_title,
                    hierarchy_path=store_meta.hierarchy_path,
                    source_file=store_meta.source_file,
                    content=content,
                    content_raw=store_meta.metadata.get("content_raw", "") if req.include_content_raw else "",
                    evidence_snippet=store_meta.metadata.get("evidence_snippet", "") if req.include_evidence else "",
                    score=float(r.get("score", 0)),
                    scores_detail={"exact": float(r.get("score", 0))},
                    rank=rank,
                    matched_by=["exact"],
                    trace=trace,
                    metadata=store_meta.metadata,
                )
            else:
                # ChunkStore 中找不到（理论不应发生），降级用 exact 原始字段
                hit = RetrievalHit(
                    chunk_id=chunk_id,
                    chunk_type=r.get("chunk_type", "clause"),
                    doc_id=r.get("doc_id", ""),
                    doc_name=r.get("doc_name", ""),
                    doc_title=r.get("doc_title", ""),
                    hierarchy_path=r.get("hierarchy_path", ""),
                    source_file=r.get("source", ""),
                    content=r.get("text", r.get("content", "")),
                    content_raw=r.get("content_raw", "") if req.include_content_raw else "",
                    evidence_snippet=r.get("evidence_snippet", "") if req.include_evidence else "",
                    score=float(r.get("score", 0)),
                    scores_detail={"exact": float(r.get("score", 0))},
                    rank=rank,
                    matched_by=["exact"],
                    trace=trace,
                    metadata=r.get("metadata", {}),
                )
            hits.append(hit)
        return hits

    def _build_hits_metadata(self, req: RetrievalRequest,
                             chunk_ids: List[str],
                             filters_applied: Dict[str, Any]) -> List[RetrievalHit]:
        """从 metadata 过滤结果（chunk_id 列表）构建 RetrievalHit 列表"""
        hits = []
        for rank, chunk_id in enumerate(chunk_ids[:req.top_k], 1):
            trace = self._build_trace(req, {"filters_applied": filters_applied})
            result = self._build_result_from_store(chunk_id, 0.0)
            hit = RetrievalHit.from_chunk_result(
                result,
                rank=rank,
                matched_by=["metadata"],
                trace=trace,
            )
            hits.append(hit)
        return hits

    def _build_single_hit(self, doc_idx: int, req: RetrievalRequest,
                          rank: int, scores_detail: Dict[str, float],
                          matched_by: List[str],
                          trace: Dict[str, Any]) -> RetrievalHit:
        """从 doc_idx 构建 RetrievalHit（通过 ChunkStore 获取数据）"""
        chunk_id = self._store.get_chunk_id(doc_idx)
        primary_score = scores_detail.get("rrf", scores_detail.get(list(scores_detail.keys())[0], 0))
        result = self._build_result_from_store(chunk_id, primary_score)

        # 截断
        content = result.get("content", "")
        if req.max_chars_per_hit > 0 and len(content) > req.max_chars_per_hit:
            content = content[:req.max_chars_per_hit]

        return RetrievalHit(
            chunk_id=chunk_id,
            chunk_type=result.get("chunk_type", "clause"),
            doc_id=result.get("doc_id", ""),
            doc_name=result.get("doc_name", ""),
            doc_title=result.get("doc_title", ""),
            hierarchy_path=result.get("hierarchy_path", ""),
            source_file=result.get("source", ""),
            content=content,
            content_raw=result.get("content_raw", "") if req.include_content_raw else "",
            evidence_snippet=result.get("evidence_snippet", "") if req.include_evidence else "",
            score=result.get("score", 0),
            scores_detail=scores_detail,
            rank=rank,
            matched_by=matched_by,
            trace=trace,
            metadata=result.get("metadata", {}),
        )

    def _build_trace(self, req: RetrievalRequest,
                     details: Dict[str, Any]) -> Dict[str, Any]:
        """构建标准 trace dict"""
        return {
            "request_id": req.request_id,
            "strategy": req.strategy.value,
            "rerank": req.rerank.value,
            "timestamp": req.timestamp,
            **details,
        }

    def _search_structured(self, req: RetrievalRequest) -> List[RetrievalHit]:
        """结构化检索：relation / table"""
        if req.strategy == RetrievalStrategy.RELATION:
            results = self.search_chunks(**req.filters, limit=req.top_k) if self.relation else []
        elif req.strategy == RetrievalStrategy.TABLE:
            table_name = req.filters.get("table_name", "")
            pattern = req.query or req.filters.get("pattern", "")
            col_name = req.filters.get("col_name")
            results = self.table.find_rows(table_name, pattern, col_name) if self.table else []
        else:
            return []

        hits = []
        for rank, r in enumerate(results[:req.top_k], 1):
            # DB/Table 返回 dict，直接使用；否则尝试通过 ChunkStore 补全
            if isinstance(r, dict):
                result_dict = r
            elif hasattr(r, "chunk_id"):
                result_dict = self._build_result_from_store(r.chunk_id, 0)
            else:
                continue
            hit = RetrievalHit.from_chunk_result(
                result_dict,
                rank=rank,
                matched_by=[req.strategy.value],
                trace=self._build_trace(req, {"filters_applied": dict(req.filters)}),
            )
            hits.append(hit)
        return hits

    def _expand_context_for_hits(self, req: RetrievalRequest,
                                 hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """为每条命中附加邻域上下文"""
        for hit in hits:
            if hit.chunk_id and self.neighborhood:
                hit.context = self.neighborhood.get_context(hit.chunk_id)
        return hits

    # ============================================================
    # 结果格式化
    # ============================================================
    def _build_result_from_store(self, chunk_id: str, score: float) -> Dict[str, Any]:
        """
        通过 ChunkStore 构建统一返回格式（meta + content 分层获取）。
        对齐 chunk_json约定.md v1.2，与 _chunk_to_result 输出结构一致。
        """
        meta = self._store.get_meta(chunk_id)
        if not meta:
            return {
                "chunk_id": chunk_id, "chunk_type": "clause",
                "content": "", "score": score, "metadata": {},
            }

        m = meta.metadata  # metadata dict（包含所有详细字段）
        content = self._store.get_content(chunk_id)

        return {
            # ╔══════════════════════════════════════════════════════════════════╗
            # ║                     原 JSON 字段                             ║
            # ║              从解析器产出的 JSONL 中直接读取，不经检索加工          ║
            # ╚══════════════════════════════════════════════════════════════════╝
            "chunk_id":   meta.chunk_id,
            "chunk_type": meta.chunk_type,
            "content":          content,
            "content_raw":      m.get("content_raw", ""),
            "content_markdown": m.get("content_markdown", ""),
            "content_json":     m.get("content_json", {}),
            "hierarchy_path":   meta.hierarchy_path,
            "evidence_snippet": m.get("evidence_snippet", ""),
            "source":    meta.source_file,
            "doc_id":    meta.doc_id,
            "doc_name":  meta.doc_name,
            "doc_title": meta.doc_title,
            "parent_chunk_id": m.get("parent_chunk_id", ""),
            "sub_chunks":      m.get("sub_chunks", []),

            # ── metadata ──
            "metadata": {
                # 文档级
                "parser_type":      m.get("parser_type", ""),
                "parser_version":   m.get("parser_version", ""),
                "parse_timestamp":  m.get("parse_timestamp", ""),
                "source_url":       m.get("source_url", ""),
                "sha256":           m.get("sha256", ""),
                "column":           m.get("column", ""),
                # 结构级
                "attachment_no":        m.get("attachment_no", ""),
                "applicable_scope":     m.get("applicable_scope", ""),
                "parent_section":       m.get("parent_section", ""),
                "chapter_number":       m.get("chapter_number", ""),
                "clause_number":        m.get("clause_number", ""),
                "subclause_number":     m.get("subclause_number", ""),
                "capital_tool_level":   m.get("capital_tool_level", ""),
                "context_chunk_id":     m.get("context_chunk_id", ""),
                "glossary_term":        m.get("glossary_term", ""),
                "glossary_definition":  m.get("glossary_definition", ""),
                "glossary_term_number": m.get("glossary_term_number", ""),
                # 语义级
                "normative_level":        m.get("normative_level", ""),
                "numeric_conditions":     m.get("numeric_conditions", []),
                "keywords":               m.get("keywords", []),
                "cross_attachment_refs":  m.get("cross_attachment_refs", []),
                "cross_table_refs":       m.get("cross_table_refs", []),
                # 表格专属
                "table_name":         m.get("table_name", ""),
                "table_full_name":    m.get("table_full_name", ""),
                "table_section_name": m.get("table_section_name", ""),
                "sheet_name":         m.get("sheet_name", ""),
                "row_count":          m.get("row_count", 0),
                "col_count":          m.get("col_count", 0),
                "merge_info":         m.get("merge_info", []),
                "cross_refs":         m.get("cross_refs", []),
                # 格式专属
                "_extra": m.get("_extra", {}),
            },

            # ╔══════════════════════════════════════════════════════════════════╗
            # ║                   检索过程定义的字段                              ║
            # ║           由检索层计算/注入，不来自原始 JSONL                        ║
            # ╚══════════════════════════════════════════════════════════════════╝
            "score": score,
        }

    def _chunk_to_result(self, chunk: Chunk, score: float) -> Dict[str, Any]:
        """将 Chunk 转为统一返回格式，对齐 chunk_json约定.md v1.2"""
        meta = chunk.metadata
        return {
            # ╔══════════════════════════════════════════════════════════════════╗
            # ║                     原 JSON 字段                             ║
            # ║              从解析器产出的 JSONL 中直接读取，不经检索加工          ║
            # ╚══════════════════════════════════════════════════════════════════╝
            "chunk_id":   chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "content":          chunk.content,
            "content_raw":      meta.get("content_raw", ""),
            "content_markdown": meta.get("content_markdown", ""),
            "content_json":     meta.get("content_json", {}),
            "hierarchy_path":   chunk.hierarchy_path,
            "evidence_snippet": meta.get("evidence_snippet", ""),
            "source":    chunk.source_file,
            "doc_id":    chunk.doc_id,
            "doc_name":  chunk.doc_name,
            "doc_title": chunk.doc_title,
            "parent_chunk_id": meta.get("parent_chunk_id", ""),
            "sub_chunks":      meta.get("sub_chunks", []),

            # ── metadata ──
            "metadata": {
                # 文档级
                "parser_type":      meta.get("parser_type", ""),
                "parser_version":   meta.get("parser_version", ""),
                "parse_timestamp":  meta.get("parse_timestamp", ""),
                "source_url":       meta.get("source_url", ""),
                "sha256":           meta.get("sha256", ""),
                "column":           meta.get("column", ""),
                # 结构级
                "attachment_no":        meta.get("attachment_no", ""),
                "applicable_scope":     meta.get("applicable_scope", ""),
                "parent_section":       meta.get("parent_section", ""),
                "chapter_number":       meta.get("chapter_number", ""),
                "clause_number":        meta.get("clause_number", ""),
                "subclause_number":     meta.get("subclause_number", ""),
                "capital_tool_level":   meta.get("capital_tool_level", ""),
                "context_chunk_id":     meta.get("context_chunk_id", ""),
                "glossary_term":        meta.get("glossary_term", ""),
                "glossary_definition":  meta.get("glossary_definition", ""),
                "glossary_term_number": meta.get("glossary_term_number", ""),
                # 语义级
                "normative_level":        meta.get("normative_level", ""),
                "numeric_conditions":     meta.get("numeric_conditions", []),
                "keywords":               meta.get("keywords", []),
                "cross_attachment_refs":  meta.get("cross_attachment_refs", []),
                "cross_table_refs":       meta.get("cross_table_refs", []),
                # 表格专属
                "table_name":         meta.get("table_name", ""),
                "table_full_name":    meta.get("table_full_name", ""),
                "table_section_name": meta.get("table_section_name", ""),
                "sheet_name":         meta.get("sheet_name", ""),
                "row_count":          meta.get("row_count", 0),
                "col_count":          meta.get("col_count", 0),
                "merge_info":         meta.get("merge_info", []),
                "cross_refs":         meta.get("cross_refs", []),
                # 格式专属
                "_extra": meta.get("_extra", {}),
            },

            # ╔══════════════════════════════════════════════════════════════════╗
            # ║                   检索过程定义的字段                              ║
            # ║           由检索层计算/注入，不来自原始 JSONL                        ║
            # ╚══════════════════════════════════════════════════════════════════╝

            # ── 分数 ──
            "score": score,
            #   检索得分。用于 top_k 截断、证据充分性判断。
            #   ⚠️ 待拆为 scores.{lexical,dense,fusion,rerank} + matched_by
        }

    # ============================================================
    # LLM 上下文格式化
    # ============================================================
    def format_for_llm(self, results: Union[List[RetrievalHit], List[Dict[str, Any]]],
                       max_chars: int = 3000) -> str:
        """
        检索结果 → LLM prompt 参考文本。

        接受 RetrievalHit 列表（推荐）或兼容旧 dict 列表。
        每条附编号 [1][2]...，LLM 回答时可引用。
        """
        if not results:
            return ""

        parts = ["【参考资料 — 以下内容来自监管法规文件，请严格基于此回答，并注明出处】\n"]

        total_chars = 0
        for i, r in enumerate(results, 1):
            # 统一提取：RetrievalHit 用属性，dict 用 .get()
            if isinstance(r, RetrievalHit):
                ctype = r.chunk_type
                doc_name = r.doc_name or r.source_file
                evidence = r.evidence_snippet or r.hierarchy_path
                content = r.content
                attachment = r.metadata.get("attachment_no", "")
                applicable_scope = r.metadata.get("applicable_scope", "")
            else:
                ctype = r.get("chunk_type", "clause")
                doc_name = r.get("doc_name", r.get("source", ""))
                meta = r.get("metadata", {})
                evidence = r.get("evidence_snippet", "") or r.get("hierarchy_path", "")
                content = r.get("content", r.get("text", ""))
                attachment = meta.get("attachment_no", "")
                applicable_scope = meta.get("applicable_scope", "")

            icon = CHUNK_TYPE_ICONS.get(ctype, "📄")
            scope = (f" [适用: {applicable_scope}]"
                     if applicable_scope and applicable_scope not in ("全部", "未指定")
                     else "")
            att_str = f" {attachment}" if attachment else ""
            header = f"[{i}] {icon}{att_str} {doc_name}{scope} — {evidence}"
            snippet = f"{header}\n{content[:1000]}\n"
            total_chars += len(snippet)
            if total_chars > max_chars:
                break
            parts.append(snippet)

        parts.append(f"\n【以上共 {len(parts) - 1} 条参考资料，回答时请引用编号 [1][2]...】")
        return "\n".join(parts)

    # ============================================================
    # 向量导出
    # ============================================================
    def export_vectorized(self, output_path: Union[str, Path],
                          fmt: str = "jsonl"):
        """导出向量化数据"""
        if self.dense is None or self.dense.embeddings is None:
            print("[导出] 错误：尚未构建向量索引")
            return
        if not self._store.chunk_count:
            print("[导出] 错误：chunk 列表为空")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self.dense.embeddings

        # 通过 ChunkStore 获取元信息 + content
        chunk_ids = self._store.chunk_ids
        metas = self._store.get_metas_batch(chunk_ids)
        contents = self._store.get_content_batch(chunk_ids)

        meta_list = [{
            "index": i,
            "chunk_id": chunk_ids[i],
            "chunk_type": metas[i].chunk_type if metas[i] else "",
            "content": contents.get(chunk_ids[i], ""),
            "hierarchy_path": metas[i].hierarchy_path if metas[i] else "",
            "source_file": metas[i].source_file if metas[i] else "",
            "doc_id": metas[i].doc_id if metas[i] else "",
            "doc_name": metas[i].doc_name if metas[i] else "",
            "doc_title": metas[i].doc_title if metas[i] else "",
        } for i in range(len(chunk_ids))]

        if fmt in ("jsonl", "txt"):
            ext = ".jsonl" if fmt == "jsonl" else ".txt"
            out_file = output_path.with_suffix(ext)
            with open(out_file, "w", encoding="utf-8") as f:
                for i, meta in enumerate(meta_list):
                    f.write(json.dumps({**meta, "embedding": embeddings[i].tolist()},
                                       ensure_ascii=False) + "\n")
            print(f"[导出] {out_file}")
        elif fmt == "parquet":
            out_file = output_path.with_suffix(".parquet")
            import pandas as pd
            df = pd.DataFrame(meta_list)
            emb_df = pd.DataFrame(embeddings, columns=[f"emb_{j}" for j in range(embeddings.shape[1])])
            pd.concat([df, emb_df], axis=1).to_parquet(str(out_file), index=False)
            print(f"[导出] {out_file}")
        elif fmt == "split":
            meta_file = output_path.with_suffix("").with_name(output_path.name + "_meta.jsonl")
            emb_file = output_path.with_suffix("").with_name(output_path.name + "_emb.npy")
            with open(meta_file, "w", encoding="utf-8") as f:
                for meta in meta_list:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            np.save(str(emb_file), embeddings)
            print(f"[导出] {meta_file} + {emb_file}")
        else:
            print(f"[导出] 不支持的格式 '{fmt}'，可选: jsonl / parquet / split")

    # ============================================================
    # 关系/邻域/表格 — 快捷委托（直接透传到底层检索器）
    # ============================================================
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.relation.get_chunk(chunk_id) if self.relation else None

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.relation.get_document(doc_id) if self.relation else None

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.relation.list_documents() if self.relation else []

    def get_document_versions(self, doc_name: str) -> List[Dict[str, Any]]:
        return self.relation.get_document_versions(doc_name) if self.relation else []

    def get_latest_version(self, doc_name: str) -> Optional[Dict[str, Any]]:
        return self.relation.get_latest_version(doc_name) if self.relation else None

    def search_chunks(self, **filters) -> List[Dict[str, Any]]:
        return self.relation.search_chunks(**filters) if self.relation else []

    def get_parent(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.neighborhood.get_parent(chunk_id) if self.neighborhood else None

    def get_children(self, chunk_id: str) -> List[Dict[str, Any]]:
        return self.neighborhood.get_children(chunk_id) if self.neighborhood else []

    def get_prev(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.neighborhood.get_prev(chunk_id) if self.neighborhood else None

    def get_next(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.neighborhood.get_next(chunk_id) if self.neighborhood else None

    def get_siblings(self, chunk_id: str) -> List[Dict[str, Any]]:
        return self.neighborhood.get_siblings(chunk_id) if self.neighborhood else []

    def get_surrounding(self, chunk_id: str, window: int = 2) -> List[Dict[str, Any]]:
        return self.neighborhood.get_surrounding(chunk_id, window) if self.neighborhood else []

    def get_context(self, chunk_id: str) -> Dict[str, Any]:
        return self.neighborhood.get_context(chunk_id) if self.neighborhood else {}

    def list_tables(self) -> List[str]:
        return self.table.list_tables() if self.table else []

    def get_table_data(self, table_name: str) -> List[Dict[str, str]]:
        return self.table.as_dict_list(table_name) if self.table else []

    def find_table_rows(self, table_name: str, pattern: str,
                        col_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.table.find_rows(table_name, pattern, col_name) if self.table else []

    def get_table_cell(self, table_name: str, row_index: int,
                       col_spec: Any) -> Optional[str]:
        return self.table.get_cell(table_name, row_index, col_spec) if self.table else None

    # ============================================================
    # 内部
    # ============================================================
    @staticmethod
    def _meta(chunk: Chunk) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "doc_id": chunk.doc_id,
            "doc_name": chunk.doc_name,
            "doc_title": chunk.doc_title,
            "hierarchy_path": chunk.hierarchy_path,
            "source_file": chunk.source_file,
        }

    def _populate_db(self, chunks: List[Chunk]):
        """将 chunks 写入 SQLite（增量模式：跳过已存在的记录）

        参数：
          chunks: Chunk 对象列表（临时传入，不持久化到 self）
        """
        # ── 获取已存在的 ID 集合 ──
        existing_doc_ids = self._db.get_existing_doc_ids()
        existing_chunk_ids = self._db.get_existing_chunk_ids()
        print(f"  [DB 增量写入] 已有 {len(existing_doc_ids)} 文档, {len(existing_chunk_ids)} chunks")

        # ── 增量写入文档 ──
        docs_seen = set()
        new_doc_count = 0
        for chunk in chunks:
            doc_id = chunk.doc_id
            if doc_id in docs_seen:
                continue
            docs_seen.add(doc_id)
            if doc_id in existing_doc_ids:
                continue  # 跳过已存在的文档
            self._db.upsert_document({
                "doc_id": doc_id,
                "doc_name": chunk.doc_name,
                "doc_title": chunk.doc_title,
                "parser_type": chunk.metadata.get("parser_type", ""),
                "source_file": chunk.source_file,
                "parse_timestamp": chunk.metadata.get("parse_timestamp", ""),
                "attachment_no": chunk.metadata.get("attachment_no", ""),
                "applicable_scope": chunk.metadata.get("applicable_scope", "全部"),
                "metadata": {
                    "sha256": chunk.metadata.get("sha256", ""),
                    "source_url": chunk.metadata.get("source_url", ""),
                    "parser_version": chunk.metadata.get("parser_version", ""),
                },
            })
            new_doc_count += 1
        if new_doc_count:
            print(f"  [DB] 新增 {new_doc_count} 个文档（跳过 {len(existing_doc_ids & docs_seen)} 个已有）")

        # ── 增量写入 chunks ──
        chunk_dicts = []
        top_keys = ("parent_chunk_id", "prev_chunk_id", "next_chunk_id",
                     "chapter_number", "clause_number", "subclause_number",
                     "applicable_scope", "normative_level", "capital_tool_level",
                     "table_name", "table_section_name", "sheet_name",
                     "glossary_term", "keywords", "evidence_snippet",
                     "content_raw", "sub_chunks")
        skipped = 0
        for chunk in chunks:
            if chunk.chunk_id in existing_chunk_ids:
                skipped += 1
                continue
            chunk_dicts.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "hierarchy_path": chunk.hierarchy_path,
                **{k: chunk.metadata.get(k, "") for k in top_keys},
                "keywords": chunk.metadata.get("keywords", []),
                "sub_chunks": chunk.metadata.get("sub_chunks", []),
                "metadata": {k: v for k, v in chunk.metadata.items() if k not in top_keys},
            })

        if chunk_dicts:
            self._db.insert_chunks(chunk_dicts)
            self._db.auto_link_chunks()
            print(f"  [DB] 新增 {len(chunk_dicts)} 条 chunks（跳过 {skipped} 条已有）")
        else:
            print(f"  [DB] 全部 {skipped} 条 chunks 已存在，跳过写入")

    # ============================================================
    # 属性
    # ============================================================
    @property
    def store(self) -> ChunkStore:
        """ChunkStore 实例（顶层统一数据管理）"""
        return self._store

    @property
    def chunk_count(self) -> int:
        return self._store.chunk_count

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def db(self) -> RetrievalDB:
        return self._db
