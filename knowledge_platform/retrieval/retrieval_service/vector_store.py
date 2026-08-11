"""
向量存储后端 — 支持 numpy（默认）和 FAISS 两种模式。

numpy 模式：纯内存矩阵 (N, dim)，内积检索，适合万级数据。
FAISS 模式：本地向量索引，支持 GPU 加速和近似搜索，可扩展至百万级。

使用方式：
    from vector_store import FAISSVectorStore

    store = FAISSVectorStore(dim=1024)
    store.add(chunk_ids, vectors)
    results = store.search(query_vec, top_k=10)
"""

import numpy as np
from typing import List, Tuple, Optional


class NumpyVectorStore:
    """默认 numpy 后端 — 暴力内积"""

    def __init__(self, dim: int = 512):
        self._dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []

    def add(self, chunk_ids: List[str], vectors: np.ndarray) -> None:
        self._chunk_ids = list(chunk_ids)
        self._vectors = vectors.astype(np.float32)

    def search(self, query_vec: np.ndarray, top_k: int = 10,
               allowed_indices: Optional[set] = None) -> List[Tuple[int, float]]:
        if self._vectors is None:
            return []

        scores = (self._vectors @ query_vec.reshape(-1, 1)).flatten()

        if allowed_indices is not None:
            indices = sorted(allowed_indices, key=lambda i: scores[i], reverse=True)
        else:
            indices = np.argsort(scores)[::-1]

        results = []
        for i in indices:
            if len(results) >= top_k:
                break
            s = float(scores[i])
            if s > 0:
                results.append((int(i), s))
        return results

    def save(self, path: str) -> None:
        np.save(path, self._vectors)

    def load(self, path: str) -> bool:
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return False
        self._vectors = np.load(str(p))
        return True

    @property
    def count(self) -> int:
        return len(self._chunk_ids)


class FAISSVectorStore:
    """FAISS 后端 — 本地索引，支持 GPU 加速"""

    def __init__(self, dim: int = 512, use_gpu: bool = False):
        try:
            import faiss
        except ImportError:
            raise ImportError("请先安装 faiss: pip install faiss-cpu")

        self._dim = dim
        self._chunk_ids: List[str] = []
        # IndexFlatIP = 内积 = 余弦相似度（向量已归一化时）
        self._index = faiss.IndexFlatIP(dim)

        if use_gpu:
            try:
                self._index = faiss.index_cpu_to_all_gpus(self._index)
            except Exception:
                pass  # GPU 不可用时保持 CPU

    def add(self, chunk_ids: List[str], vectors: np.ndarray) -> None:
        self._chunk_ids = list(chunk_ids)
        self._index.add(vectors.astype(np.float32))

    def search(self, query_vec: np.ndarray, top_k: int = 10,
               allowed_indices: Optional[set] = None) -> List[Tuple[int, float]]:
        if self._index.ntotal == 0:
            return []

        q = query_vec.astype(np.float32).reshape(1, -1)

        # 有过滤条件时扩大搜索量再后过滤
        fetch_k = max(top_k * 5, top_k) if allowed_indices else top_k
        scores, indices = self._index.search(q, fetch_k)

        results = []
        for i, s in zip(indices[0], scores[0]):
            if i < 0:
                continue
            if allowed_indices is not None and int(i) not in allowed_indices:
                continue
            results.append((int(i), float(s)))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str) -> None:
        import faiss
        faiss.write_index(self._index, path)

    def load(self, path: str) -> bool:
        import faiss
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return False
        self._index = faiss.read_index(str(p))
        return True

    @property
    def count(self) -> int:
        return self._index.ntotal
