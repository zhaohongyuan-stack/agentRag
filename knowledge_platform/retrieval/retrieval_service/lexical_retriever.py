"""
Lexical 检索器 — 基于 BM25 的关键词检索

职责：对文档做 BM25 得分计算，按关键词匹配度排序返回结果。

使用方式：
    retriever = LexicalRetriever()
    retriever.index(documents)
    results = retriever.search("核心一级资本合格标准", top_k=10)
    # → [{"index": 3, "text": "...", "score": 12.34}, ...]
"""

import math
import heapq
from typing import List, Tuple, Dict, Optional, Any


class LexicalRetriever:
    """
    BM25 关键词检索器。

    BM25 公式:
      score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b + b*|D|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self._tokenized: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._N = 0

    # ============================================================
    # 分词
    # ============================================================
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中文按字分词 + 英文/数字按空白和标点切分"""
        tokens: List[str] = []
        for ch in text:
            if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
                tokens.append(ch)
            elif ch.isalnum():
                tokens.append(ch.lower())
            elif ch.isspace():
                if tokens and tokens[-1] != ' ':
                    tokens.append(' ')
        raw = ''.join(tokens)
        return [t for t in raw.split() if t != ' ']

    # ============================================================
    # 索引
    # ============================================================
    def index(self, documents: List[str],
              metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        构建 BM25 索引。

        参数：
          documents: 文档文本列表
          metadatas: 可选的元数据列表（与 documents 一一对应，供外部过滤使用）
        """
        self.documents = documents
        self._metadatas = metadatas or [{} for _ in documents]
        self._N = len(documents)
        self._tokenized = [self.tokenize(doc) for doc in documents]
        self._doc_len = [len(tokens) for tokens in self._tokenized]
        self._avgdl = sum(self._doc_len) / max(self._N, 1)

        # IDF 计算
        df: Dict[str, int] = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._idf = {
            term: math.log((self._N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    # ============================================================
    # 检索
    # ============================================================
    def search(self, query: str, top_k: int = 10,
               allowed_indices: Optional[set] = None,
               raw: bool = False) -> Any:
        """
        BM25 检索。

        参数：
          query:           查询文本
          top_k:           返回条数
          allowed_indices: 允许参与检索的文档索引集合（外部过滤后传入）
          raw:             是否返回原始元组格式 (index, score)

        返回：
          raw=False: [{"index": int, "text": str, "score": float}, ...]
          raw=True:  [(int, float), ...]  — 向后兼容旧 BM25 API
        """
        query_tokens = self.tokenize(query)
        scores: List[Tuple[int, float]] = []

        for idx, doc_tokens in enumerate(self._tokenized):
            if allowed_indices is not None and idx not in allowed_indices:
                continue
            score = 0.0
            doc_len = self._doc_len[idx]
            for term in query_tokens:
                if term not in self._idf:
                    continue
                tf = doc_tokens.count(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                score += self._idf[term] * numerator / max(denominator, 1e-8)
            if score > 0:
                scores.append((idx, score))

        top = heapq.nlargest(top_k, scores, key=lambda x: x[1])

        if raw:
            return top  # [(index, score), ...]

        return [
            {"index": idx, "text": self.documents[idx][:500], "score": round(score, 4)}
            for idx, score in top
        ]

    # ============================================================
    # 持久化
    # ============================================================
    def save(self, path: str):
        """持久化 BM25 索引到文件"""
        import pickle
        data = {
            "documents": self.documents,
            "_tokenized": self._tokenized,
            "_doc_len": self._doc_len,
            "_avgdl": self._avgdl,
            "_idf": self._idf,
            "_N": self._N,
            "_metadatas": self._metadatas,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "LexicalRetriever":
        """从文件加载 BM25 索引"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        retriever = cls()
        retriever.documents = data["documents"]
        retriever._tokenized = data["_tokenized"]
        retriever._doc_len = data["_doc_len"]
        retriever._avgdl = data["_avgdl"]
        retriever._idf = data["_idf"]
        retriever._N = data["_N"]
        retriever._metadatas = data.get("_metadatas", [])
        return retriever

    @property
    def metadatas(self) -> List[Dict[str, Any]]:
        return self._metadatas

    @property
    def doc_count(self) -> int:
        return self._N
