"""
VectorDB — 基于 FAISS 的向量数据库

职责：将文档编码为向量，构建 FAISS 索引，支持 ANN 近似最近邻搜索和文件持久化。

与 DenseRetriever 的关系：
  - DenseRetriever 负责编码（embedding），VectorDB 负责索引与检索
  - VectorDB 可独立使用，也可作为 DenseRetriever 的加速后端

索引类型：
  - "flat"   : IndexFlatIP — 精确内积搜索（归一化向量 = 余弦相似度），适合 <10 万条
  - "ivf"    : IndexIVFFlat — 倒排索引 + 量化，适合 10 万 ~ 1000 万条
  - "hnsw"   : IndexHNSWFlat — 图索引，高召回、内存占用大，适合 <100 万条

使用方式：
    # === 方式一：从文本构建 ===
    db = VectorDB()
    db.build_from_texts(texts, model_name="BAAI/bge-small-zh-v1.5")
    db.save("knowledge.index")
    results = db.search("核心一级资本合格标准", top_k=10)

    # === 方式二：从已有 embeddings 构建 ===
    db = VectorDB()
    db.build_from_embeddings(embeddings, metadatas)
    db.save("knowledge.index")

    # === 方式三：加载已有索引 ===
    db = VectorDB.load("knowledge.index")
    results = db.search("查询文本", top_k=5)
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np


# ============================================================
# FAISS 可用性检测（可选依赖）
# ============================================================
_FAISS_AVAILABLE = False
_FAISS_ERROR = None
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError as e:
    _FAISS_ERROR = str(e)


# ============================================================
# 索引配置
# ============================================================
class IndexConfig:
    """FAISS 索引配置"""

    def __init__(self,
                 index_type: str = "flat",
                 nlist: int = 100,          # IVF: 聚类中心数
                 nprobe: int = 10,          # IVF: 搜索时探测的聚类数
                 M: int = 32,               # HNSW: 每个节点的连接数
                 efConstruction: int = 200,  # HNSW: 构建时的搜索宽度
                 efSearch: int = 64):       # HNSW: 检索时的搜索宽度
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "M": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IndexConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


# ============================================================
# VectorDB
# ============================================================
class VectorDB:
    """基于 FAISS 的向量数据库"""

    def __init__(self, config: Optional[IndexConfig] = None):
        if not _FAISS_AVAILABLE:
            raise ImportError(
                f"FAISS 未安装，请运行: pip install faiss-cpu\n"
                f"(如需 GPU 版: pip install faiss-gpu)\n"
                f"原始错误: {_FAISS_ERROR}"
            )
        self.config = config or IndexConfig()
        self._index: Optional[faiss.Index] = None
        self._model = None
        self._model_name: str = ""
        self._dim: int = 0
        self._id_to_meta: Dict[int, Dict[str, Any]] = {}

    # ============================================================
    # 构建索引
    # ============================================================

    def build_from_texts(self,
                         texts: List[str],
                         model_name: str = "BAAI/bge-small-zh-v1.5",
                         metadatas: Optional[List[Dict[str, Any]]] = None,
                         batch_size: int = 32,
                         show_progress: bool = True) -> "VectorDB":
        """
        从文本列表构建索引（先编码，再建索引）。

        参数：
          texts:         文档文本列表
          model_name:    Sentence-Transformers 模型名
          metadatas:     每条文本的元数据
          batch_size:    编码批大小
          show_progress: 是否显示进度条
        """
        # 1. 加载模型并编码
        self._load_model(model_name)
        print(f"  [VectorDB] 编码 {len(texts)} 条文本 ...")
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=batch_size,
        )
        return self.build_from_embeddings(embeddings, metadatas)

    def build_from_embeddings(self,
                              embeddings: np.ndarray,
                              metadatas: Optional[List[Dict[str, Any]]] = None,
                              ids: Optional[List[int]] = None) -> "VectorDB":
        """
        从已有的 embedding 数组构建索引。

        参数：
          embeddings:  shape (N, dim)，建议已归一化
          metadatas:   每条向量的元数据
          ids:         自定义 ID 列表（默认 0..N-1）
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings 必须是 2D 数组，当前 shape: {embeddings.shape}")

        n, dim = embeddings.shape
        self._dim = dim
        embeddings = embeddings.astype(np.float32)

        # 元数据
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        if ids is None:
            ids = list(range(n))
        self._id_to_meta = {i: {**meta, "vector_id": i} for i, meta in zip(ids, metadatas)}

        # 2. 构建 FAISS 索引
        self._index = self._build_faiss_index(embeddings)
        print(f"  [VectorDB] 索引构建完成 — {n} 条向量, dim={dim}, "
              f"类型={self.config.index_type}")

        return self

    def _build_faiss_index(self, embeddings: np.ndarray) -> "faiss.Index":
        """根据配置构建 FAISS 索引"""
        n, dim = embeddings.shape
        index_type = self.config.index_type

        if index_type == "flat":
            # IndexFlatIP: 内积搜索，归一化向量等同于余弦相似度
            index = faiss.IndexFlatIP(dim)

        elif index_type == "ivf":
            # IVF: 先用 KMeans 聚类，搜索时只探测最近的 nprobe 个聚类
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.config.nlist)
            # IVF 需要先训练
            print(f"  [VectorDB] 训练 IVF 索引 (nlist={self.config.nlist}) ...")
            index.train(embeddings)
            index.nprobe = self.config.nprobe

        elif index_type == "hnsw":
            # HNSW: 图索引，构建慢但检索快
            index = faiss.IndexHNSWFlat(dim, self.config.M)
            index.hnsw.efConstruction = self.config.efConstruction
            index.hnsw.efSearch = self.config.efSearch

        else:
            raise ValueError(f"不支持的索引类型: {index_type}，可选: flat / ivf / hnsw")

        # 添加向量
        index.add(embeddings)
        return index

    # ============================================================
    # 检索
    # ============================================================

    def search(self,
               query: Union[str, np.ndarray],
               top_k: int = 10,
               return_meta: bool = True) -> List[Dict[str, Any]]:
        """
        向量相似度搜索。

        参数：
          query:       查询文本（str）或查询向量（np.ndarray, shape (dim,)）
          top_k:       返回条数
          return_meta: 是否附带元数据

        返回：
          [{"id": int, "score": float, "metadata": {...}}, ...]
        """
        if self._index is None:
            raise RuntimeError("索引未构建，请先调用 build_from_texts() 或 build_from_embeddings()")

        # 文本查询 → 编码为向量
        if isinstance(query, str):
            self._load_model()
            q_emb = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
        else:
            q_emb = np.asarray(query, dtype=np.float32).reshape(1, -1)

        # FAISS 搜索
        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS 用 -1 表示无结果
                continue
            result = {
                "id": int(idx),
                "score": round(float(score), 4),
            }
            if return_meta and idx in self._id_to_meta:
                result["metadata"] = self._id_to_meta[idx]
            results.append(result)

        return results

    def search_batch(self,
                     queries: Union[List[str], np.ndarray],
                     top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        批量搜索。

        参数：
          queries: 查询文本列表 或 shape (M, dim) 的向量数组
          top_k:   每条查询返回数

        返回：
          [[{"id": int, "score": float, ...}, ...], ...]  — 每条查询一个结果列表
        """
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            if self._model is None:
                raise RuntimeError("文本查询需要先加载模型")
            q_embs = self._model.encode(queries, normalize_embeddings=True).astype(np.float32)
        else:
            q_embs = np.asarray(queries, dtype=np.float32)
            if q_embs.ndim == 1:
                q_embs = q_embs.reshape(1, -1)

        scores_all, indices_all = self._index.search(q_embs, top_k)

        all_results = []
        for scores, indices in zip(scores_all, indices_all):
            batch_results = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                result = {"id": int(idx), "score": round(float(score), 4)}
                if idx in self._id_to_meta:
                    result["metadata"] = self._id_to_meta[idx]
                batch_results.append(result)
            all_results.append(batch_results)

        return all_results

    # ============================================================
    # 持久化
    # ============================================================

    def save(self, path: Union[str, Path]):
        """
        保存向量数据库到磁盘。

        生成文件：
          {path}.index     — FAISS 索引文件
          {path}.meta.json — 元数据 + 配置
          {path}.model.txt — 模型名（如有）

        参数：
          path: 保存路径（不含扩展名），如 "knowledge" → knowledge.index + knowledge.meta.json
        """
        if self._index is None:
            raise RuntimeError("索引为空，无法保存")

        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        # 1. FAISS 索引
        index_path = base.with_suffix(".index")
        faiss.write_index(self._index, str(index_path))
        print(f"  [VectorDB] 索引已保存: {index_path}")

        # 2. 元数据
        meta_path = base.with_name(base.name + ".meta.json")
        meta = {
            "dim": self._dim,
            "model_name": self._model_name,
            "config": self.config.to_dict(),
            "id_to_meta": self._id_to_meta,
            "total_vectors": self._index.ntotal,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"  [VectorDB] 元数据已保存: {meta_path}")

        # 3. 模型名（轻量提示）
        if self._model_name:
            model_path = base.with_name(base.name + ".model.txt")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(self._model_name + "\n")

    @classmethod
    def load(cls, path: Union[str, Path],
             model_name: Optional[str] = None) -> "VectorDB":
        """
        从磁盘加载向量数据库。

        参数：
          path:       保存时的基础路径（不含扩展名）
          model_name: 覆盖保存时的模型名（可选）

        返回：
          VectorDB 实例（已加载索引，可立即搜索）
        """
        if not _FAISS_AVAILABLE:
            raise ImportError(f"FAISS 未安装: {_FAISS_ERROR}")

        base = Path(path)

        # 1. 加载元数据
        meta_path = base.with_name(base.name + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 2. 加载 FAISS 索引
        index_path = base.with_suffix(".index")
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        index = faiss.read_index(str(index_path))

        # 3. 恢复 VectorDB 实例
        config = IndexConfig.from_dict(meta.get("config", {}))
        db = cls(config=config)
        db._index = index
        db._dim = meta["dim"]
        db._model_name = model_name or meta.get("model_name", "")
        db._id_to_meta = {
            int(k): v for k, v in meta.get("id_to_meta", {}).items()
        }

        # 恢复 IVF nprobe / HNSW efSearch
        if config.index_type == "ivf" and hasattr(index, 'nprobe'):
            index.nprobe = config.nprobe
        elif config.index_type == "hnsw" and hasattr(index, 'hnsw'):
            if hasattr(index.hnsw, 'efSearch'):
                index.hnsw.efSearch = config.efSearch

        print(f"  [VectorDB] 索引已加载: {index_path} "
              f"({index.ntotal} 条, dim={meta['dim']}, type={config.index_type})")
        return db

    # ============================================================
    # 模型
    # ============================================================

    def _load_model(self, model_name: Optional[str] = None):
        """延迟加载 Sentence-Transformers 模型（优先 ModelScope 本地缓存）"""
        name = model_name or self._model_name
        if not name:
            raise RuntimeError("模型名未知，请使用 build_from_texts() 构建索引，"
                             "或 VectorDB.load(path, model_name='BAAI/bge-small-zh-v1.5') 指定模型")
        if self._model is not None and self._model_name == name:
            return
        from sentence_transformers import SentenceTransformer
        # 优先从 ModelScope 获取本地路径（与 DenseRetriever 一致）
        try:
            from modelscope import snapshot_download
            local_path = snapshot_download(name)
        except Exception:
            local_path = name
        print(f"  [VectorDB] 加载模型: {name}")
        self._model = SentenceTransformer(local_path)
        self._model_name = name

    def encode(self, texts: List[str],
               batch_size: int = 32,
               show_progress: bool = True) -> np.ndarray:
        """编码文本为归一化向量（便捷方法）"""
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 build_from_texts()")
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=batch_size,
        )

    # ============================================================
    # 属性
    # ============================================================

    @property
    def count(self) -> int:
        """向量总数"""
        return self._index.ntotal if self._index is not None else 0

    @property
    def dim(self) -> int:
        """向量维度"""
        return self._dim

    @property
    def index_type(self) -> str:
        """索引类型"""
        return self.config.index_type

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return (f"VectorDB({self.count} vectors, dim={self._dim}, "
                f"type={self.config.index_type}, model={self._model_name or 'N/A'})")


# ============================================================
# 便捷工具：将 VectorDB 接入 DenseRetriever
# ============================================================

def faiss_search_adapter(vector_db: VectorDB, query_emb: np.ndarray,
                         top_k: int) -> Tuple[List[int], List[float]]:
    """
    适配器：用 FAISS VectorDB 替代 DenseRetriever 的暴力搜索。

    用法：
        # 用 VectorDB 加速 DenseRetriever
        db = VectorDB.load("knowledge.index")
        retriever = DenseRetriever()
        retriever.embeddings = None  # 不再用暴力搜索
        retriever._vector_db = db   # 注入向量库

        # search 时会自动走 FAISS（需在 DenseRetriever.search 中适配）
    """
    results = vector_db.search_batch(query_emb, top_k)
    if not results or not results[0]:
        return [], []
    ids = [r["id"] for r in results[0]]
    scores = [r["score"] for r in results[0]]
    return ids, scores
