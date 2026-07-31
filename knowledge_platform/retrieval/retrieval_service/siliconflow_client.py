"""
硅基流动 SiliconFlow API 客户端

统一封装嵌入（Embedding）和重排序（Rerank）API 调用。

API 文档：
  - 嵌入: https://api-docs.siliconflow.cn/docs/api/embeddings-post
  - 重排序: https://api-docs.siliconflow.cn/docs/api/rerank-post

设计原则：
  - 延迟加载 requests（避免未安装时报错）
  - 自动重试（指数退避，最多 3 次）
  - 分批调用（避免超长输入）
  - 配置通过环境变量或构造参数传入

使用方式：

    # 嵌入
    client = SiliconFlowEmbedding(api_key="sk-xxx", model="BAAI/bge-large-zh-v1.5")
    vectors = client.embed(["文本1", "文本2"])
    query_vec = client.embed_query("查询文本")

    # 重排序
    reranker = SiliconFlowReranker(api_key="sk-yyy", model="BAAI/bge-reranker-v2-m3")
    scores = reranker.rerank("查询", ["文档1", "文档2", "文档3"])
    # → [{"index": 0, "score": 0.95}, {"index": 2, "score": 0.78}, ...]
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# ── 默认配置 ──
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_TIMEOUT = 30  # 秒
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 32  # 嵌入批量大小


# ============================================================
# 请求工具
# ============================================================
def _post_json(url: str, headers: dict, payload: dict,
               timeout: int = DEFAULT_TIMEOUT,
               max_retries: int = DEFAULT_MAX_RETRIES) -> dict:
    """
    带重试的 POST JSON 请求。

    指数退避：1s, 2s, 4s
    400 等客户端错误不重试，直接抛出（附带响应体便于诊断）
    """
    import requests

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)

            # 限流（429）或服务端错误（5xx）→ 重试
            if resp.status_code in (429, 500, 502, 503, 504):
                delay = 1.0 * (2 ** attempt)
                logger.warning(
                    f"  [SiliconFlow] HTTP {resp.status_code}，"
                    f"第 {attempt+1}/{max_retries+1} 次重试，等待 {delay:.1f}s"
                )
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(delay)
                continue

            # 400 等客户端错误 → 不重试，直接抛出（附带响应体）
            if not resp.ok:
                error_body = resp.text[:500]
                raise RuntimeError(
                    f"  [SiliconFlow] HTTP {resp.status_code}: {error_body}"
                )

            return resp.json()

        except requests.exceptions.Timeout:
            last_error = f"请求超时 ({timeout}s)"
            delay = 1.0 * (2 ** attempt)
            logger.warning(f"  [SiliconFlow] {last_error}，重试 {attempt+1}/{max_retries+1}")
            time.sleep(delay)

        except requests.exceptions.ConnectionError as e:
            last_error = f"连接错误: {e}"
            delay = 1.0 * (2 ** attempt)
            logger.warning(f"  [SiliconFlow] {last_error}，重试 {attempt+1}/{max_retries+1}")
            time.sleep(delay)

    raise RuntimeError(f"  [SiliconFlow] 请求失败（已达最大重试次数 {max_retries}）: {last_error}")


# ============================================================
# 嵌入客户端
# ============================================================
class SiliconFlowEmbedding:
    """
    硅基流动嵌入 API 客户端。

    兼容 Sentence-Transformers 的 encode() 接口，可直接替换 DenseRetriever 中的本地模型。
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_EMBED_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 timeout: int = DEFAULT_TIMEOUT,
                 max_chars: int = 512):
        """
        参数：
          api_key:    硅基流动 API Key
          model:      嵌入模型名
          base_url:   API 基础 URL
          batch_size: 每批最大文本数
          timeout:    请求超时（秒）
          max_chars:  单条文本最大字符数（超长自动截断，适配 512 token 限制）
        """
        self.api_key = api_key or os.environ.get("SILICONFLOW_EMBED_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "未找到嵌入 API Key。请通过 api_key 参数传入，"
                "或设置环境变量 SILICONFLOW_EMBED_API_KEY"
            )
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout = timeout
        self._max_chars = max_chars
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._dim: Optional[int] = None  # 向量维度（首次调用后缓存）

        print(f"  [SiliconFlow] 嵌入客户端就绪: model={model}, batch_size={batch_size}, max_chars={max_chars}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本。

        参数：
          texts: 文本列表

        返回：
          向量列表 List[List[float]]，与输入文本一一对应

        注意：
          BAAI/bge-large-zh-v1.5 最大 512 tokens。客户端预截断至 max_chars 字符
          （中文约 1 字 = 1-2 token，512 字符约 300-500 token，留余量）。
          空字符串会被替换为单个空格，避免 400 错误。
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # 分批调用
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # 空字符串替换为空格；超长文本预截断
            batch = [
                (t if t.strip() else " ")[:self._max_chars]
                for t in batch
            ]
            payload = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            url = f"{self.base_url}/embeddings"
            result = _post_json(url, self._headers, payload, timeout=self.timeout)

            # 按 index 排序确保顺序一致
            data = sorted(result["data"], key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in data]
            all_embeddings.extend(batch_embeddings)

            # 缓存维度
            if self._dim is None and batch_embeddings:
                self._dim = len(batch_embeddings[0])

            if len(texts) > self.batch_size:
                print(f"    [嵌入] {start+len(batch)}/{len(texts)} ...")

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单条查询文本"""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        """向量维度（首次嵌入后可用）"""
        if self._dim is None:
            # 发送一次探针请求获取维度
            probe = self.embed(["维度探测"])
            return self._dim or 0
        return self._dim

    # ============================================================
    # Sentence-Transformers 兼容接口
    # ============================================================
    def encode(self, texts: List[str],
               normalize_embeddings: bool = True,
               show_progress_bar: bool = False,
               batch_size: Optional[int] = None) -> "np.ndarray":
        """
        兼容 Sentence-Transformers 的 encode 方法。

        参数：
          texts:               文本列表
          normalize_embeddings: 是否归一化（API 返回的已是归一化向量，此参数仅做兼容）
          show_progress_bar:   是否显示进度条（API 调用自带日志，忽略此参数）
          batch_size:          批量大小（覆盖默认值）

        返回：
          numpy.ndarray，形状 (N, dim)
        """
        import numpy as np

        if batch_size:
            old_bs = self.batch_size
            self.batch_size = batch_size
            try:
                embeddings = self.embed(texts)
            finally:
                self.batch_size = old_bs
        else:
            embeddings = self.embed(texts)

        arr = np.array(embeddings, dtype=np.float32)

        # API 返回的向量已归一化，但保险起见再归一化一次
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms

        return arr


# ============================================================
# 重排序客户端
# ============================================================
class SiliconFlowReranker:
    """
    硅基流动重排序 API 客户端。

    兼容 Cross-Encoder 的 predict() 接口，可直接替换 retrieval_api.py 中的本地 reranker。
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_RERANK_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 timeout: int = DEFAULT_TIMEOUT,
                 max_chunks_per_doc: int = 1024):
        self.api_key = api_key or os.environ.get("SILICONFLOW_RERANK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "未找到重排序 API Key。请通过 api_key 参数传入，"
                "或设置环境变量 SILICONFLOW_RERANK_API_KEY"
            )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_chunks_per_doc = max_chunks_per_doc
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        print(f"  [SiliconFlow] 重排序客户端就绪: model={model}")

    def rerank(self, query: str, documents: List[str],
               top_n: Optional[int] = None,
               return_documents: bool = False) -> List[Dict[str, Any]]:
        """
        对文档列表按与查询的相关性重排序。

        参数：
          query:            查询文本
          documents:        文档文本列表
          top_n:            返回前 N 条（None 则返回全部）
          return_documents: 是否在结果中包含文档原文

        返回：
          [{"index": int, "relevance_score": float, "document": {"text": str}(可选)}, ...]
          按 relevance_score 降序排列
        """
        if not documents:
            return []

        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        url = f"{self.base_url}/rerank"
        result = _post_json(url, self._headers, payload, timeout=self.timeout)

        # 解析结果
        results = []
        for item in result.get("results", []):
            entry = {
                "index": item["index"],
                "score": float(item["relevance_score"]),
            }
            if return_documents and "document" in item:
                entry["document"] = item["document"]
            results.append(entry)

        return results

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """
        兼容 Cross-Encoder 的 predict 方法。

        参数：
          pairs: [[query, document], [query, document], ...]

        返回：
          相关性分数列表 List[float]
        """
        if not pairs:
            return []

        # 硅基流动 rerank API 接受一个 query + 多个 documents
        # 而非多个 query-document pair，所以需要分组
        # 但如果每个 pair 的 query 不同，需要分别调用
        # 优化：相同 query 的 pair 合并为一次调用
        from collections import defaultdict

        query_groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for idx, (q, doc) in enumerate(pairs):
            query_groups[q].append((idx, doc))

        scores: List[Optional[float]] = [None] * len(pairs)

        for query, items in query_groups.items():
            docs = [doc for _, doc in items]
            rerank_results = self.rerank(query, docs)

            # rerank 结果按 score 降序，需要映射回原始顺序
            score_map = {r["index"]: r["score"] for r in rerank_results}
            for local_idx, (orig_idx, _) in enumerate(items):
                scores[orig_idx] = score_map.get(local_idx, 0.0)

        return [s if s is not None else 0.0 for s in scores]

    def rerank_with_scores(self, query: str, documents: List[str],
                           top_n: Optional[int] = None) -> List[float]:
        """
        简化接口：返回与 documents 等长的分数列表（按原始顺序）。

        参数：
          query:     查询文本
          documents: 文档文本列表
          top_n:     （可选）只返回前 N 条的分数

        返回：
          分数列表 List[float]，与 documents 等长
        """
        if not documents:
            return []

        rerank_results = self.rerank(query, documents, top_n=top_n)

        # 构建 index → score 映射
        score_map = {r["index"]: r["score"] for r in rerank_results}
        return [score_map.get(i, 0.0) for i in range(len(documents))]
