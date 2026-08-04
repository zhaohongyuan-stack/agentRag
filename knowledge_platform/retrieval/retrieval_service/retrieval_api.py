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

import json
import hashlib
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from .chunk import Chunk, load_json_chunks, CHUNK_TYPE_ICONS
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


class RetrievalAPI:
    """Retrieval API v1 — 7 大检索器 + RRF 混合融合"""

    def __init__(self,
                 embed_model: str = "BAAI/bge-small-zh-v1.5",
                 use_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-large",
                 db_path: Optional[str] = None):
        self._embed_model = embed_model
        self._use_reranker = use_reranker
        self._reranker_model = reranker_model
        self._reranker = None

        # 7 大独立检索器（load 后填充，也可外部注入用于测试）
        self.lexical: Optional[LexicalRetriever] = None
        self.dense: Optional[DenseRetriever] = None
        self.exact: Optional[ExactRetriever] = None
        self.metadata: Optional[MetadataRetriever] = None
        self.relation: Optional[RelationRetriever] = None
        self.neighborhood: Optional[NeighborhoodRetriever] = None
        self.table: Optional[TableRetriever] = None

        self._chunks: List[Chunk] = []
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

        print("=" * 60)
        print("  Retrieval API v1 — 加载全部检索器 ...")
        print("=" * 60)

        # ── 尝试从缓存恢复（索引持久化：不依赖源 JSON 文件）──
        if cache_dir.exists():
            if self._try_load_from_cache(cache_dir):
                return self
            print("\n  [提示] 缓存不完整，将从 JSON 源文件构建 ...")

        # ── 1. 加载 JSON chunks ──
        print("\n[1/8] 加载 JSON chunk 文件 ...")
        self._chunks = load_json_chunks(source)
        if not self._chunks:
            print("[警告] 未找到任何 chunk")
            return self
        doc_texts = [c.content for c in self._chunks]
        print(f"  共 {len(self._chunks)} 个 chunks")

        # ── 缓存目录 & 内容哈希（用于索引持久化）──
        cache_dir.mkdir(parents=True, exist_ok=True)
        content_hash = hashlib.sha256("\n".join(doc_texts).encode()).hexdigest()[:16]

        # ── 2. Lexical (BM25) ──
        bm25_cache = cache_dir / f"bm25_{content_hash}.pkl"
        if bm25_cache.exists():
            print("\n[2/8] 加载 Lexical (BM25) 缓存 ...")
            self.lexical = LexicalRetriever.load(str(bm25_cache))
        else:
            print("\n[2/8] 构建 Lexical (BM25) 索引 ...")
            self.lexical = LexicalRetriever()
            self.lexical.index(doc_texts, metadatas=[self._meta(c) for c in self._chunks])
            self.lexical.save(str(bm25_cache))
            print(f"  已缓存: {bm25_cache.name}")

        # ── 3. Dense + FAISS 向量库 ──
        print("\n[3/8] 构建 Dense (向量) 索引 + FAISS 向量库 ...")
        self.dense = DenseRetriever(self._embed_model, use_faiss=True)
        self.dense.index(doc_texts, metadatas=[self._meta(c) for c in self._chunks], cache_dir=str(cache_dir))

        # ── 4. Exact ──
        print("\n[4/8] 构建 Exact 索引 ...")
        self.exact = ExactRetriever()
        self.exact.index(doc_texts, metadatas=[self._meta(c) for c in self._chunks])

        # ── 5. Metadata ──
        meta_cache = cache_dir / f"metadata_{content_hash}.pkl"
        if meta_cache.exists():
            print("\n[5/8] 加载 Metadata 缓存 ...")
            self.metadata = MetadataRetriever.load(str(meta_cache))
        else:
            print("\n[5/8] 构建 Metadata 索引 ...")
            self.metadata = MetadataRetriever()
            self.metadata.index(self._chunks)
            self.metadata.save(str(meta_cache))
            print(f"  已缓存: {meta_cache.name}")

        # ── 6. DB + Relation / Neighborhood ──
        db_cache = cache_dir / f"retrieval_{content_hash}.db"
        self._db = RetrievalDB(db_path or str(db_cache))
        self._db.open()
        if populate_db and self._db.count_chunks() == 0:
            print(f"\n[6/8] 写入关系数据库 ...")
            self._populate_db()
        else:
            print(f"\n[6/8] 加载关系数据库缓存 ...")
        self.relation = RelationRetriever(self._db)
        self.neighborhood = NeighborhoodRetriever(self._db)
        print(f"  文档: {len(self.relation.list_documents())}, Chunks: {self.relation.count_chunks()}")

        # ── 7. Table ──
        table_cache = cache_dir / f"table_{content_hash}.pkl"
        if table_cache.exists():
            print(f"\n[7/8] 加载 Table 缓存 ...")
            self.table = TableRetriever.load(str(table_cache))
        else:
            print(f"\n[7/8] 构建 Table 索引 ...")
            self.table = TableRetriever()
            self.table.index(self._chunks)
            self.table.save(str(table_cache))
            print(f"  已缓存: {table_cache.name}")

        # ── 8. Cross-Encoder (optional) ──
        if self._use_reranker:
            print(f"\n[8/8] 加载 Cross-Encoder: {self._reranker_model} ...")
            from sentence_transformers import CrossEncoder
            local_path = modelscope_download(self._reranker_model)
            self._reranker = CrossEncoder(local_path)
        else:
            print(f"\n[8/8] 跳过 Cross-Encoder（未启用）")

        self._loaded = True
        self._print_summary()
        return self

    # ============================================================
    # 索引持久化 — 从缓存恢复，不依赖源 JSON 文件
    # ============================================================
    def _try_load_from_cache(self, cache_dir: Path) -> bool:
        """
        尝试从缓存目录完整恢复所有检索器。
        成功返回 True（后续启动无需 JSON 源文件）。
        失败返回 False（需回退到 JSON 加载）。
        """
        # ── 1. 扫描缓存目录，找到内容哈希 ──
        bm25_files = sorted(cache_dir.glob("bm25_*.pkl"))
        if not bm25_files:
            return False
        content_hash = bm25_files[0].stem.replace("bm25_", "")

        # ── 2. 检查所有必需缓存文件是否就绪 ──
        required = [
            cache_dir / f"bm25_{content_hash}.pkl",
            cache_dir / f"metadata_{content_hash}.pkl",
            cache_dir / f"table_{content_hash}.pkl",
            cache_dir / f"retrieval_{content_hash}.db",
        ]
        faiss_index = list(cache_dir.glob(f"faiss_*_{content_hash}.index"))
        if not all(p.exists() for p in required) or not faiss_index:
            return False

        print(f"\n  [缓存] 检测到完整索引 (hash={content_hash})，跳过 JSON 加载")

        # ── 3. 从 DB 恢复 chunks ──
        db_path = str(cache_dir / f"retrieval_{content_hash}.db")
        self._db = RetrievalDB(db_path)
        self._db.open()
        chunk_count = self._db.count_chunks()
        if chunk_count == 0:
            return False

        chunk_dicts = self._db.load_all_chunks()
        self._chunks = [
            Chunk(
                chunk_id=d["chunk_id"],
                chunk_type=d["chunk_type"],
                content=d["content"],
                hierarchy_path=d.get("hierarchy_path", ""),
                source_file=d.get("source_file", ""),
                doc_id=d["doc_id"],
                doc_name=d.get("doc_name", ""),
                doc_title=d.get("doc_title", ""),
                metadata=d.get("_meta", {}),
            )
            for d in chunk_dicts
        ]
        doc_texts = [c.content for c in self._chunks]
        print(f"  [缓存] 从 DB 恢复 {len(self._chunks)} 个 chunks")

        # ── 4. 加载各检索器索引 ──
        print(f"  [缓存] 加载 BM25 索引 ...")
        self.lexical = LexicalRetriever.load(str(cache_dir / f"bm25_{content_hash}.pkl"))

        print(f"  [缓存] 加载 Dense 向量索引 ...")
        self.dense = DenseRetriever(self._embed_model, use_faiss=True)
        self.dense.index(doc_texts, metadatas=[self._meta(c) for c in self._chunks], cache_dir=str(cache_dir))

        print(f"  [缓存] 加载 Exact 索引 ...")
        self.exact = ExactRetriever()
        self.exact.index(doc_texts, metadatas=[self._meta(c) for c in self._chunks])

        print(f"  [缓存] 加载 Metadata 索引 ...")
        self.metadata = MetadataRetriever.load(str(cache_dir / f"metadata_{content_hash}.pkl"))

        print(f"  [缓存] 加载 Table 索引 ...")
        self.table = TableRetriever.load(str(cache_dir / f"table_{content_hash}.pkl"))

        print(f"  [缓存] 加载关系数据库 ...")
        self.relation = RelationRetriever(self._db)
        self.neighborhood = NeighborhoodRetriever(self._db)

        self._loaded = True
        self._print_summary()
        return True

    def _print_summary(self):
        """打印加载完成的摘要信息"""
        print(f"\n  Retrieval API v1 就绪 — "
              f"{len(Counter(c.doc_id for c in self._chunks))} 文档, "
              f"{len(self._chunks)} Chunks, "
              f"{len(self.table.list_tables())} 表格")
        print("=" * 60 + "\n")

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
        if not self._chunks:
            return []

        # ── Phase 0: 元数据过滤 ──
        allowed: Optional[set] = None
        filters_applied: Dict[str, Any] = {}
        if req.filters and self.metadata:
            allowed = {
                self._chunks.index(r) for r in self.metadata.search(req.filters, limit=999999)
                if r in self._chunks
            }
            filters_applied = dict(req.filters)
            if not allowed:
                # 渐进式回退：逐步去掉过滤条件直到命中（避免元数据过滤过严导致 0 命中空拒答）
                # 第1步：去掉 doc_name 重试
                _fallback_to_full = False
                if "doc_name" in req.filters:
                    relaxed_filters = {k: v for k, v in req.filters.items() if k != "doc_name"}
                    if relaxed_filters:
                        print(f"  [Phase 0] doc_name 过滤无匹配，回退去掉doc_name重试 filters={relaxed_filters}")
                        allowed = {
                            self._chunks.index(r) for r in self.metadata.search(relaxed_filters, limit=999999)
                            if r in self._chunks
                        }
                        filters_applied = dict(relaxed_filters)
                    else:
                        _fallback_to_full = True
                        filters_applied = {}
                        print(f"  [Phase 0] doc_name 过滤无匹配，回退全库检索 ({len(self._chunks)} chunks)")
                # 第2步：去掉 doc_name 后仍无匹配，去掉所有过滤条件 → 全库检索
                if not _fallback_to_full and not allowed:
                    print(f"  [Phase 0] 所有过滤条件均无匹配，回退全库检索 ({len(self._chunks)} chunks)")
                    _fallback_to_full = True
                    filters_applied = {}
                if _fallback_to_full:
                    allowed = None  # 全库检索（下游 allowed is None 即不过滤）

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
            results = self.metadata.search(req.filters, limit=req.top_k) if self.metadata else []
            hits = self._build_hits_metadata(req, results, filters_applied=filters_applied)

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

            # Phase 2: Cross-Encoder 精排（可选）
            if req.rerank == RerankMode.CROSS_ENC and self._reranker and len(sorted_candidates) > req.top_k:
                rerank_candidates = sorted_candidates[:req.rerank_k]
                candidate_indices = [idx for idx, _ in rerank_candidates]
                pairs = [[req.query, self._chunks[idx].content] for idx in candidate_indices]
                rerank_scores = self._reranker.predict(pairs)
                scored = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)

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
        """从 exact 检索结果构建 RetrievalHit 列表"""
        hits = []
        for rank, r in enumerate(raw[:req.top_k], 1):
            trace = self._build_trace(req, {
                "exact_mode": req.exact_mode,
                "match_position": r.get("match_pos", -1),
                "filters_applied": filters_applied,
            })
            hit = RetrievalHit(
                chunk_id=r.get("chunk_id", ""),
                chunk_type=r.get("chunk_type", "clause"),
                doc_id=r.get("doc_id", ""),
                doc_name=r.get("doc_name", ""),
                doc_title=r.get("doc_title", ""),
                hierarchy_path=r.get("hierarchy_path", ""),
                source_file=r.get("source", ""),
                content=r.get("content", ""),
                content_raw=r.get("content_raw", "") if req.include_content_raw else "",
                evidence_snippet=r.get("evidence_snippet", ""),
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
                             chunks: List[Any],
                             filters_applied: Dict[str, Any]) -> List[RetrievalHit]:
        """从 metadata 过滤结果构建 RetrievalHit 列表"""
        hits = []
        for rank, chunk in enumerate(chunks[:req.top_k], 1):
            trace = self._build_trace(req, {"filters_applied": filters_applied})
            hit = RetrievalHit.from_chunk_result(
                self._chunk_to_result(chunk, 0.0),
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
        """从 Chunk 索引构建单条 RetrievalHit"""
        chunk = self._chunks[doc_idx]
        result = self._chunk_to_result(chunk, scores_detail.get("rrf", scores_detail.get(list(scores_detail.keys())[0], 0)))

        # 截断
        content = result.get("content", "")
        if req.max_chars_per_hit > 0 and len(content) > req.max_chars_per_hit:
            content = content[:req.max_chars_per_hit]

        return RetrievalHit(
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
            doc_id=chunk.doc_id,
            doc_name=chunk.doc_name,
            doc_title=chunk.doc_title,
            hierarchy_path=chunk.hierarchy_path,
            source_file=chunk.source_file,
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
            hit = RetrievalHit.from_chunk_result(
                r if isinstance(r, dict) else self._chunk_to_result(r, 0),
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
        if not self._chunks:
            print("[导出] 错误：chunk 列表为空")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self.dense.embeddings

        meta_list = [{
            "index": i,
            "chunk_id": c.chunk_id,
            "chunk_type": c.chunk_type,
            "content": c.content,
            "hierarchy_path": c.hierarchy_path,
            "source_file": c.source_file,
            "doc_id": c.doc_id,
            "doc_name": c.doc_name,
            "doc_title": c.doc_title,
        } for i, c in enumerate(self._chunks)]

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

    def _populate_db(self):
        docs_seen = set()
        for chunk in self._chunks:
            doc_id = chunk.doc_id
            if doc_id in docs_seen:
                continue
            docs_seen.add(doc_id)
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

        chunk_dicts = []
        top_keys = ("parent_chunk_id", "prev_chunk_id", "next_chunk_id",
                     "chapter_number", "clause_number", "subclause_number",
                     "applicable_scope", "normative_level", "capital_tool_level",
                     "table_name", "table_section_name", "sheet_name",
                     "glossary_term", "keywords", "evidence_snippet",
                     "content_raw", "sub_chunks")
        for chunk in self._chunks:
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
        self._db.insert_chunks(chunk_dicts)
        self._db.auto_link_chunks()

    # ============================================================
    # 属性
    # ============================================================
    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def db(self) -> RetrievalDB:
        return self._db
