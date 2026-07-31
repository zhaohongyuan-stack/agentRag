"""
Lexical 检索器 — 基于 BM25 的关键词检索

职责：对文档做 BM25 得分计算，按关键词匹配度排序返回结果。

⚠️ 第三阶段改造（2026-07-31）：
  - 集成 jieba 分词，替代逐字分词，大幅提升中文 BM25 召回率
  - 内置金融监管领域自定义词典（资本充足率、核心一级资本等）
  - 保留逐字分词作为 fallback（jieba 不可用时自动降级）

⚠️ 第一阶段改造（2026-07-31）：
  - 不再保存 documents 全文和 _metadatas 副本
  - 新增 _chunk_ids 列表用于下标 → chunk_id 映射
  - search() raw 模式仍返回 [(doc_idx, score)]，非 raw 模式返回 chunk_id（不含 text）
  - 原文统一由 ChunkStore 管理

使用方式：
    retriever = LexicalRetriever()
    retriever.index(chunk_ids, documents)  # documents 仅用于分词，不持久化
    results = retriever.search("核心一级资本合格标准", top_k=10, raw=True)
    # → [(3, 12.34), ...]  — raw 模式返回 (doc_idx, score)
    chunk_id = retriever.get_chunk_id(doc_idx)  # doc_idx → chunk_id
"""

import math
import heapq
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# ── 缓存版本号：分词算法变更时递增，自动失效旧缓存 ──
INDEX_VERSION = "v2_jieba"

# ── jieba 懒加载 ──
_jieba_instance = None
_jieba_available = False

# ── 金融监管领域自定义词典（避免被错误切分）──
_DOMAIN_TERMS = [
    "核心一级资本", "核心一级资本充足率", "一级资本", "二级资本",
    "资本充足率", "资本管理办法", "商业银行", "系统重要性银行",
    "附加资本", "杠杆率", "流动性覆盖率", "净稳定资金比例",
    "不良贷款率", "拨备覆盖率", "贷款拨备率", "总资产", "总负债",
    "银行业金融机构", "金融租赁公司", "消费金融公司", "汽车金融公司",
    "资产管理公司", "信托公司", "财务公司", "货币经纪公司",
    "资产负债月度", "比上年同期增长率", "表格区域", "工作表",
    "数据单元格", "银行业", "金融机构", "监管指标",
]


def _init_jieba():
    """懒加载 jieba 并注入领域词典"""
    global _jieba_instance, _jieba_available
    if _jieba_instance is not None:
        return

    try:
        import jieba
        _jieba_instance = jieba
        _jieba_available = True

        # 注入领域词典（提高专业术语分词准确度）
        for term in _DOMAIN_TERMS:
            jieba.add_word(term, freq=10000)

        # 设置静默模式，避免 jieba 日志干扰
        jieba.setLogLevel(60)

        print("  [LexicalRetriever] jieba 分词已就绪（含金融监管领域词典）")
    except ImportError:
        _jieba_available = False
        print("  [LexicalRetriever] jieba 未安装，降级为逐字分词")


class LexicalRetriever:
    """
    BM25 关键词检索器。

    BM25 公式:
      score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b + b*|D|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # ── 索引必需数据（不含 content 全文）──
        self._chunk_ids: List[str] = []         # 下标 → chunk_id 映射
        self._tokenized: List[List[str]] = []   # 预分词结果（BM25 运算必需）
        self._doc_len: List[int] = []            # 每篇文档长度
        self._avgdl: float = 0.0                 # 平均文档长度
        self._idf: Dict[str, float] = {}         # 逆文档频率
        self._N = 0                              # 文档总数

    # ============================================================
    # 分词
    # ============================================================
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        中文分词（jieba 优先，逐字 fallback）。

        jieba 模式下：
          - 先用 jieba 粗分（cut_mode=default）
          - 过滤单字符噪声（标点、空白、单字英文）
          - 保留中文单字（金融术语可能为单字，如"贷"）
        逐字 fallback 模式下：
          - 中文逐字 + 英文/数字按空白切分
        """
        if not text:
            return []

        # ── jieba 分词 ──
        if _jieba_available and _jieba_instance is not None:
            tokens = []
            for word in _jieba_instance.cut(text):
                word = word.strip().lower()
                if not word:
                    continue
                # 过滤纯标点和空白
                if re.match(r'^[\s\W]+$', word):
                    continue
                tokens.append(word)
            return tokens

        # ── 逐字 fallback ──
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
    def index(self, chunk_ids: List[str], documents: List[str]) -> None:
        """
        构建 BM25 索引。

        参数：
          chunk_ids:  chunk_id 列表（与 documents 一一对应）
          documents:  文档文本列表（仅用于分词，不持久化保存）
        """
        # 初始化 jieba（首次调用时加载）
        _init_jieba()

        self._chunk_ids = chunk_ids
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
          raw=True:  [(int, float), ...]  — (doc_idx, score) 列表
          raw=False: [{"index": int, "chunk_id": str, "score": float}, ...]
                     （不含 text，原文由 ChunkStore 获取）
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
            return top  # [(doc_idx, score), ...]

        return [
            {"index": idx, "chunk_id": self._chunk_ids[idx] if idx < len(self._chunk_ids) else "",
             "score": round(score, 4)}
            for idx, score in top
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
        return self._N

    # ============================================================
    # 持久化
    # ============================================================
    def save_index(self, path: str) -> None:
        """将 BM25 索引持久化到 pickle 文件（不含 documents 全文）"""
        data = {
            "version": INDEX_VERSION,
            "_chunk_ids": self._chunk_ids,
            "_tokenized": self._tokenized,
            "_doc_len": self._doc_len,
            "_avgdl": self._avgdl,
            "_idf": self._idf,
            "_N": self._N,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, path: str) -> bool:
        """从 pickle 文件加载 BM25 索引，成功返回 True"""
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)

        # ── 版本校验：旧版本缓存自动失效 ──
        cached_version = data.get("version", "v1")
        if cached_version != INDEX_VERSION:
            print(f"  [LexicalRetriever] 缓存版本不匹配 ({cached_version} → {INDEX_VERSION})，需重建")
            return False

        # 兼容旧格式（含 documents/_metadatas 的旧缓存自动迁移）
        self._chunk_ids = data.get("_chunk_ids", [])
        self._tokenized = data["_tokenized"]
        self._doc_len = data["_doc_len"]
        self._avgdl = data["_avgdl"]
        self._idf = data["_idf"]
        self._N = data["_N"]

        # 旧格式兼容：如果没有 _chunk_ids，用 documents 长度生成占位 ID
        if not self._chunk_ids and self._N > 0:
            self._chunk_ids = [str(i) for i in range(self._N)]

        # 确保 jieba 已初始化（查询时需要用到）
        _init_jieba()

        return True
