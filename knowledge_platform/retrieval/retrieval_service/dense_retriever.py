"""
Dense 检索器 — 稠密向量语义检索（支持本地模型 / 硅基流动 API）

职责：将文档编码为归一化向量，通过余弦相似度（内积）做语义检索。

两种嵌入模式：
  1. 本地模式（默认）：使用 Sentence-Transformers + ModelScope 本地模型
  2. API 模式（use_api=True）：使用硅基流动 SiliconFlow API 嵌入
     - 无需下载本地模型，降低资源占用
     - 兼容 Sentence-Transformers encode() 接口，切换透明

⚠️ 第一阶段改造（2026-07-31）：
  - 不再保存 documents 全文和 _metadatas 副本
  - 新增 _chunk_ids 列表用于下标 → chunk_id 映射
  - search() raw 模式仍返回 [(doc_idx, score)]，非 raw 模式返回 chunk_id（不含 text）
  - 原文统一由 ChunkStore 管理

使用方式：
    # 本地模式
    retriever = DenseRetriever("BAAI/bge-small-zh-v1.5")
    retriever.index(chunk_ids, documents)
    results = retriever.search("核心一级资本合格标准", top_k=10, raw=True)

    # API 模式（硅基流动）
    retriever = DenseRetriever(use_api=True, api_key="sk-xxx", api_model="BAAI/bge-large-zh-v1.5")
    retriever.index(chunk_ids, documents)
    results = retriever.search("核心一级资本合格标准", top_k=10, raw=True)
"""

import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import hashlib


class DenseRetriever:
    """Dense 向量语义检索器（支持本地模型 / 硅基流动 API 两种模式）"""

    def __init__(self,
                 model_name: str = "BAAI/bge-small-zh-v1.5",
                 use_api: bool = False,
                 api_key: Optional[str] = None,
                 api_model: Optional[str] = None):
        """
        参数：
          model_name:  本地模型名（ModelScope），use_api=False 时使用
          use_api:     是否使用硅基流动 API 嵌入（True 时不再加载本地模型）
          api_key:     硅基流动嵌入 API Key（None 则从环境变量 SILICONFLOW_EMBED_API_KEY 读取）
          api_model:   API 嵌入模型名（默认 BAAI/bge-large-zh-v1.5）
        """
        self.model_name = model_name
        self._use_api = use_api
        self._api_key = api_key
        self._api_model = api_model or "BAAI/bge-large-zh-v1.5"
        self._model = None
        # ── 索引必需数据（不含 content 全文）──
        self.embeddings: Optional[np.ndarray] = None  # 向量矩阵（检索必需）
        self._chunk_ids: List[str] = []                # 下标 → chunk_id 映射

    # ============================================================
    # 模型加载
    # ============================================================
    def _load_model(self):
        """延迟加载模型（首次 index 或 search 时触发）

        根据 _use_api 选择：
          - True:  创建 SiliconFlowEmbedding 客户端（API 嵌入，无需本地模型）
          - False: 加载本地 SentenceTransformer 模型（ModelScope 缓存）
        """
        if self._model is not None:
            return

        if self._use_api:
            # ── 硅基流动 API 嵌入模式 ──
            from .siliconflow_client import SiliconFlowEmbedding
            self._model = SiliconFlowEmbedding(
                api_key=self._api_key,
                model=self._api_model,
            )
            print(f"  [DenseRetriever] 使用 API 嵌入: {self._api_model}")
        else:
            # ── 本地模型模式 ──
            from sentence_transformers import SentenceTransformer
            from .utils import modelscope_download
            local_path = modelscope_download(self.model_name)
            print(f"  [DenseRetriever] 加载本地模型: {self.model_name} → {local_path}")
            self._model = SentenceTransformer(local_path)

    # ============================================================
    # 索引
    # ============================================================
    def index(self, chunk_ids: List[str], documents: List[str],
              cache_dir: Optional[str] = None):
        """
        将文档编码为归一化向量。

        参数：
          chunk_ids:  chunk_id 列表（与 documents 一一对应）
          documents:  文档文本列表（仅用于编码，不持久化保存）
          cache_dir:  向量缓存目录（为 None 则不缓存）
        """
        self._load_model()
        self._chunk_ids = chunk_ids

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
          raw=True:  [(int, float), ...]  — (doc_idx, score) 列表
          raw=False: [{"index": int, "chunk_id": str, "score": float}, ...]
                     （不含 text，原文由 ChunkStore 获取）
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
            return raw_results  # [(doc_idx, score), ...]

        return [
            {"index": idx, "chunk_id": self._chunk_ids[idx] if idx < len(self._chunk_ids) else "",
             "score": round(score, 4)}
            for idx, score in raw_results
        ]

    # ============================================================
    # 下标 → chunk_id 映射
    # ============================================================
    def get_chunk_id(self, doc_idx: int) -> str:
        """doc_idx → chunk_id 映射"""
        if 0 <= doc_idx < len(self._chunk_ids):
            return self._chunk_ids[doc_idx]
        return ""

    @property
    def chunk_ids(self) -> List[str]:
        return self._chunk_ids

    @property
    def doc_count(self) -> int:
        return len(self._chunk_ids)

    # ============================================================
    # 持久化（向量已有 .npy 缓存，这里补充 chunk_ids）
    # ============================================================
    def save_meta(self, path: str) -> None:
        """持久化 chunk_ids 和 API 配置到 pickle 文件（向量通过 .npy 单独保存）"""
        data = {
            "_chunk_ids": self._chunk_ids,
            "_use_api": self._use_api,
            "_api_model": self._api_model,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_meta(self, path: str, embeddings_path: str) -> bool:
        """从 pickle + .npy 加载完整状态，成功返回 True"""
        p = Path(path)
        e = Path(embeddings_path)
        if not p.exists() or not e.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)

        # 兼容旧格式（含 documents/_metadatas 的旧缓存自动迁移）
        self._chunk_ids = data.get("_chunk_ids", [])
        self._use_api = data.get("_use_api", False)
        self._api_model = data.get("_api_model", "BAAI/bge-large-zh-v1.5")
        self.embeddings = np.load(str(e))

        # 旧格式兼容：如果没有 _chunk_ids，用 embeddings 长度生成占位 ID
        if not self._chunk_ids and self.embeddings is not None:
            self._chunk_ids = [str(i) for i in range(len(self.embeddings))]

        return True
