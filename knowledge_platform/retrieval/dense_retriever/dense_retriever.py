"""
Dense 检索器 — 基于 Sentence-Transformers 的稠密向量语义检索

职责：将文档编码为归一化向量，通过余弦相似度（内积）做语义检索。

使用方式：
    retriever = DenseRetriever("BAAI/bge-small-zh-v1.5")
    retriever.index(documents)
    results = retriever.search("核心一级资本合格标准", top_k=10)
    # → [{"index": 3, "text": "...", "score": 0.98}, ...]
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import hashlib


class DenseRetriever:
    """Dense 向量语义检索器"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

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
        将文档编码为归一化向量。

        参数：
          documents: 文档文本列表
          metadatas: 可选的元数据列表
          cache_dir: 向量缓存目录（为 None 则不缓存）
        """
        self._load_model()
        self.documents = documents
        self._metadatas = metadatas or [{} for _ in documents]

        # 尝试缓存加载
        if cache_dir:
            cache_path = self._get_cache_path(documents, cache_dir)
            if cache_path and Path(cache_path).exists():
                print(f"  [DenseRetriever] 缓存命中: {Path(cache_path).name}")
                self.embeddings = np.load(cache_path)
                return

        print(f"  [DenseRetriever] 编码 {len(documents)} 条文档 ...")
        self.embeddings = self._model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )

        # 保存缓存
        if cache_dir:
            cache_path = self._get_cache_path(documents, cache_dir)
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.embeddings)
                print(f"  [DenseRetriever] 缓存已保存: {Path(cache_path).name}")

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
        scores = (self.embeddings @ q_emb.T).flatten()

        # 按得分降序取 top_k
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
