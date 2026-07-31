"""
Dense 检索器 — 基于 Sentence-Transformers 的稠密向量语义检索

职责：将文档编码为归一化向量，通过余弦相似度（内积）做语义检索。

后端：
  - 默认：暴力矩阵乘法（numpy），适合 <10 万条
  - 可选：FAISS 向量索引（VectorDB），适合大规模数据，支持 ANN 加速

使用方式：
    # === 暴力搜索（默认） ===
    retriever = DenseRetriever("BAAI/bge-small-zh-v1.5")
    retriever.index(documents)
    results = retriever.search("核心一级资本合格标准", top_k=10)

    # === FAISS 加速 ===
    retriever = DenseRetriever("BAAI/bge-small-zh-v1.5", use_faiss=True)
    retriever.index(documents)
    retriever.vector_db.save("knowledge")        # 保存向量库到磁盘
    # results 格式相同，但内部走 ANN 搜索
    results = retriever.search("核心一级资本合格标准", top_k=10)
    # → [{"index": 3, "text": "...", "score": 0.98}, ...]
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import hashlib


class DenseRetriever:
    """Dense 向量语义检索器（支持暴力搜索 / FAISS 加速）"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 use_faiss: bool = False,
                 faiss_index_type: str = "flat",
                 faiss_nlist: int = 100,
                 faiss_nprobe: int = 10):
        self.model_name = model_name
        self._model = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

        # FAISS 后端
        self._use_faiss = use_faiss
        self._faiss_index_type = faiss_index_type
        self._faiss_nlist = faiss_nlist
        self._faiss_nprobe = faiss_nprobe
        self.vector_db: Optional[Any] = None  # VectorDB 实例，index() 后填充

    # ============================================================
    # 模型加载
    # ============================================================
    def _load_model(self):
        """延迟加载模型（首次 index 或 search 时触发）"""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        from .utils import modelscope_download
        local_path = modelscope_download(self.model_name)
        print(f"  [DenseRetriever] 加载模型: {self.model_name} → {local_path}")
        self._model = SentenceTransformer(local_path)

    # ============================================================
    # 索引
    # ============================================================
    def index(self, documents: List[str],
              metadatas: Optional[List[Dict[str, Any]]] = None,
              cache_dir: Optional[str] = None):
        """
        将文档编码为归一化向量，可选构建 FAISS 向量库。

        参数：
          documents: 文档文本列表
          metadatas: 可选的元数据列表
          cache_dir: 向量缓存目录（为 None 则不缓存；目录下存放 .npy 和 .index 文件）
        """
        self._load_model()
        self.documents = documents
        self._metadatas = metadatas or [{} for _ in documents]

        content_hash = hashlib.sha256("\n".join(documents).encode()).hexdigest()[:16]
        emb_cache = Path(cache_dir) / f"embeddings_{content_hash}.npy" if cache_dir else None
        faiss_base = Path(cache_dir) / f"faiss_{self._faiss_index_type}_{content_hash}" if cache_dir else None

        # ── 尝试从缓存加载 ──
        if emb_cache and emb_cache.exists():
            print(f"  [DenseRetriever] 缓存命中: {emb_cache.name}")
            self.embeddings = np.load(str(emb_cache))

            # FAISS 索引缓存：有则加载，无则从 embeddings 补建
            if self._use_faiss:
                if faiss_base and faiss_base.with_suffix(".index").exists():
                    print(f"  [DenseRetriever] FAISS 缓存命中: {faiss_base.name}.index")
                    from ...indexes.vector.vector_db import VectorDB
                    self.vector_db = VectorDB.load(str(faiss_base))
                else:
                    print(f"  [DenseRetriever] 补建 FAISS 索引 ...")
                    self._build_faiss_from_embeddings(faiss_base)
            return

        # ── 编码 ──
        print(f"  [DenseRetriever] 编码 {len(documents)} 条文档 ...")
        self.embeddings = self._model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )

        # ── 保存 embedding 缓存 ──
        if emb_cache:
            emb_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(emb_cache), self.embeddings)
            print(f"  [DenseRetriever] 缓存已保存: {emb_cache.name}")

        # ── 构建 FAISS 向量库 ──
        if self._use_faiss:
            self._build_faiss_from_embeddings(faiss_base)

    def _build_faiss_from_embeddings(self, save_path: Optional[Path] = None):
        """从 self.embeddings 构建 FAISS 向量库"""
        from ...indexes.vector.vector_db import VectorDB, IndexConfig
        config = IndexConfig(
            index_type=self._faiss_index_type,
            nlist=self._faiss_nlist,
            nprobe=self._faiss_nprobe,
        )
        self.vector_db = VectorDB(config=config)
        self.vector_db._model = self._model
        self.vector_db._model_name = self.model_name
        self.vector_db.build_from_embeddings(
            self.embeddings,
            metadatas=self._metadatas,
        )
        if save_path:
            self.vector_db.save(str(save_path))
            print(f"  [DenseRetriever] FAISS 索引已保存: {save_path.name}.index")

    @staticmethod
    def _get_cache_path(documents: List[str], cache_dir: str) -> Optional[str]:
        """生成基于文本 hash 的缓存路径"""
        text_hash = hashlib.sha256("\n".join(documents).encode()).hexdigest()[:16]
        return str(Path(cache_dir) / f"dense_emb_{text_hash}.npy")

    # ============================================================
    # 检索
    # ============================================================
    def search(self, query: str, top_k: int = 10,
               allowed_indices: Optional[set] = None,
               raw: bool = False) -> Any:
        """
        Dense 向量检索。

        参数：
          query:           查询文本
          top_k:           返回条数
          allowed_indices: 允许参与检索的文档索引集合
          raw:             是否返回原始元组格式 (index, score)

        返回：
          raw=False: [{"index": int, "text": str, "score": float}, ...]
          raw=True:  [(int, float), ...]  — 向后兼容旧 EmbeddingRetriever API
        """
        self._load_model()
        if self.embeddings is None:
            return []

        q_emb = self._model.encode([query], normalize_embeddings=True)

        # ── FAISS 加速路径 ──
        if self.vector_db is not None and self.vector_db._index is not None:
            faiss_results = self.vector_db.search(q_emb[0], top_k=top_k)
            raw_results = []
            for r in faiss_results:
                idx = r["id"]
                if allowed_indices is not None and idx not in allowed_indices:
                    continue
                raw_results.append((idx, r["score"]))
        else:
            # ── 暴力搜索路径 ──
            scores = (self.embeddings @ q_emb.T).flatten()
            if allowed_indices is not None:
                indices = sorted(allowed_indices, key=lambda i: scores[i], reverse=True)
            else:
                indices = np.argsort(scores)[::-1]

            raw_results = []
            for i in indices:
                if len(raw_results) >= top_k:
                    break
                s = float(scores[i])
                if s > 0:
                    raw_results.append((int(i), s))

        if raw:
            return raw_results  # [(index, score), ...]

        return [
            {"index": idx, "text": self.documents[idx][:500], "score": round(score, 4)}
            for idx, score in raw_results
        ]

    @property
    def metadatas(self) -> List[Dict[str, Any]]:
        return self._metadatas

    @property
    def doc_count(self) -> int:
        return len(self.documents)
