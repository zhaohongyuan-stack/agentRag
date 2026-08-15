"""
行业分区向量分库 — 基于目录结构的三分区检索路由

设计思路：
  regulatory_docs/
    ├── 银行业/    → 独立的 RetrievalAPI 实例（自己的 DB、向量索引、BM25...）
    ├── 保险业/    → 独立的 RetrievalAPI 实例
    └── 其他/      → 独立的 RetrievalAPI 实例

  每个分区是完全独立的检索系统，路由层负责：
    1. 加载三个分区
    2. 根据 filters / target_industries 决定搜哪些分区
    3. 合并多分区结果，按 score 降序排列

使用方式：
    from industry_partition import IndustryPartitionAPI

    api = IndustryPartitionAPI("regulatory_docs/")
    api.load()

    # 自动推断行业
    hits = api.search("不良贷款率", filters={"applicable_scope": "商业银行"})

    # 显式指定分区
    hits = api.search("偿付能力", target_industries=["保险业"])

    # 搜全部
    hits = api.search("金融监管")
"""

import os
import time as _time
from pathlib import Path
from typing import List, Dict, Optional, Any

from .retrieval_api import RetrievalAPI
from .retrieval_request import RetrievalRequest, RetrievalHit, RetrievalStrategy


# ============================================================
# 分区名称常量
# ============================================================

PARTITION_NAMES = ("银行业", "保险业", "其他")

# 用于从 filters 中推断行业的关键词
INDUSTRY_FILTER_KEYWORDS: Dict[str, List[str]] = {
    "银行业": [
        "银行", "商业银行", "政策性银行", "外资银行", "民营银行",
        "农村商业银行", "城市商业银行", "大型商业银行", "股份制商业银行",
        "银行业金融机构", "存款", "贷款", "资本管理", "资本充足",
        "流动性风险", "大额风险暴露", "关联交易", "并表管理",
        "资本管理办法", "资本工具", "信用风险", "操作风险", "市场风险",
        "巴塞尔", "拨备", "不良贷款", "存贷比", "杠杆率",
    ],
    "保险业": [
        "保险", "保险公司", "财产保险", "人身保险", "再保险",
        "保险资产管理", "偿付能力", "保险资金", "保费",
        "保险代理", "保险经纪", "保险公估", "保险保障基金",
    ],
    "其他": [
        "信托", "证券", "基金", "金融租赁", "消费金融", "汽车金融",
        "理财", "资产管理", "财务公司", "支付", "反洗钱", "外汇",
        "债券", "同业", "票据", "衍生品", "金融控股",
        "小额贷款", "融资担保", "融资租赁", "典当", "保理",
    ],
}


# ============================================================
# IndustryPartitionAPI — 多分区检索路由
# ============================================================

class IndustryPartitionAPI:
    """
    行业分区检索路由。

    内部维护 3 个独立的 RetrievalAPI 实例（银行业/保险业/其他），
    每个分区从自己的子目录加载数据，有独立的缓存和索引。
    """

    def __init__(self,
                 base_dir: str = "regulatory_docs",
                 embed_model: str = "BAAI/bge-small-zh-v1.5",
                 use_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 reranker_api_key: Optional[str] = None,
                 use_embed_api: bool = False,
                 embed_api_key: Optional[str] = None,
                 embed_api_model: Optional[str] = None):
        """
        参数：
          base_dir:        数据根目录（内含 银行业/ 保险业/ 其他/ 三个子目录）
          embed_model:     嵌入模型名
          use_reranker:    是否启用重排序
          reranker_model:  重排序模型名
          reranker_api_key: 重排序 API Key
          use_embed_api:   是否使用 API 嵌入
          embed_api_key:   嵌入 API Key
          embed_api_model: 嵌入 API 模型名
        """
        self._base_dir = Path(base_dir)
        self._embed_model = embed_model
        self._use_reranker = use_reranker
        self._reranker_model = reranker_model
        self._reranker_api_key = reranker_api_key
        self._use_embed_api = use_embed_api
        self._embed_api_key = embed_api_key
        self._embed_api_model = embed_api_model

        # 三个分区各自独立的 RetrievalAPI
        self._partitions: Dict[str, RetrievalAPI] = {}
        self._loaded = False

    # ════════════════════════════════════════════════════════════
    # 加载
    # ════════════════════════════════════════════════════════════

    def load(self, lightweight: bool = False) -> "IndustryPartitionAPI":
        """
        加载三个分区的数据。

        每个分区从 {base_dir}/{行业名}/ 子目录加载 JSONL 文件，
        缓存放在各子目录的 .cache/ 下，互不干扰。
        """
        print("=" * 60)
        print("  行业分区检索 — 加载三个分区（银行业 / 保险业 / 其他）")
        print("=" * 60)

        for name in PARTITION_NAMES:
            partition_dir = self._base_dir / name
            if not partition_dir.exists():
                print(f"\n  [警告] 分区目录不存在: {partition_dir}，跳过")
                continue

            # 每个分区使用自己的 SQLite 数据库（放在分区目录的 .cache/ 下）
            cache_dir = partition_dir / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "retrieval.db")

            print(f"\n{'─' * 60}")
            print(f"  [{name}] 加载分区: {partition_dir}")
            print(f"{'─' * 60}")

            api = RetrievalAPI(
                embed_model=self._embed_model,
                use_reranker=self._use_reranker,
                reranker_model=self._reranker_model,
                reranker_api_key=self._reranker_api_key,
                use_embed_api=self._use_embed_api,
                embed_api_key=self._embed_api_key,
                embed_api_model=self._embed_api_model,
                db_path=db_path,
            )
            api.load(str(partition_dir), lightweight=lightweight)
            self._partitions[name] = api

        self._loaded = True
        self._print_summary()
        return self

    def _print_summary(self):
        """打印分区加载摘要"""
        print(f"\n{'=' * 60}")
        print(f"  行业分区检索就绪")
        print(f"{'=' * 60}")
        total_chunks = 0
        total_docs = 0
        for name in PARTITION_NAMES:
            api = self._partitions.get(name)
            if api:
                chunks = api.chunk_count
                docs = api.doc_count
                total_chunks += chunks
                total_docs += docs
                print(f"  {name}: {docs} 文档, {chunks} chunks")
            else:
                print(f"  {name}: 未加载")
        print(f"  合计: {total_docs} 文档, {total_chunks} chunks")
        print(f"{'=' * 60}\n")

    # ════════════════════════════════════════════════════════════
    # 行业推断
    # ════════════════════════════════════════════════════════════

    @classmethod
    def infer_industry(cls, filters: Dict[str, Any]) -> Optional[str]:
        """
        从 filters 中推断目标行业。

        拼接所有 filter value 做关键词匹配，返回第一个匹配的行业名。
        无法推断时返回 None。
        """
        text = " ".join(str(v) for v in filters.values() if v)
        if not text.strip():
            return None

        # 银行业优先（避免"保险"误匹配"银行存款保险"类文档）
        for kw in INDUSTRY_FILTER_KEYWORDS["银行业"]:
            if kw in text:
                return "银行业"

        for kw in INDUSTRY_FILTER_KEYWORDS["保险业"]:
            if kw in text:
                return "保险业"

        for kw in INDUSTRY_FILTER_KEYWORDS["其他"]:
            if kw in text:
                return "其他"

        return None

    # ════════════════════════════════════════════════════════════
    # 检索
    # ════════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 10,
               strategy: str = "hybrid",
               filters: Optional[Dict[str, Any]] = None,
               target_industries: Optional[List[str]] = None,
               bm25_k: int = 20, vector_k: int = 20,
               expand_context: bool = False) -> List[Dict[str, Any]]:
        """
        分区检索入口（便捷方法，返回 dict 列表）。

        参数：
          query:             查询文本
          top_k:             返回条数
          strategy:          检索策略
          filters:           元数据过滤条件
          target_industries: 指定搜哪些分区，None=自动推断或全搜
          bm25_k:            BM25 候选数
          vector_k:          向量候选数
          expand_context:    是否扩展邻域

        返回：
          List[dict]: 合并排序后的命中列表
        """
        req = RetrievalRequest(
            query=query,
            strategy=RetrievalStrategy(strategy),
            top_k=top_k,
            filters=filters or {},
            bm25_k=bm25_k,
            vector_k=vector_k,
            expand_context=expand_context,
            target_industries=target_industries,
        )
        hits = self.search_request(req)
        return [h.to_dict() for h in hits]

    def search_request(self, req: RetrievalRequest) -> List[RetrievalHit]:
        """
        分区检索入口（强类型，返回 RetrievalHit 列表）。

        路由逻辑：
          1. req.target_industries 非空 → 只搜指定分区
          2. 从 req.filters 推断行业 → 只搜推断出的分区
          3. 无法推断 → 搜全部三个分区
        """
        if not self._loaded:
            raise RuntimeError("请先调用 .load() 加载数据")

        _t0 = _time.time()

        # 确定要搜的分区
        target = req.target_industries
        if not target:
            inferred = self.infer_industry(req.filters)
            if inferred:
                target = [inferred]

        if target:
            partition_names = [n for n in target if n in self._partitions]
        else:
            partition_names = [n for n in PARTITION_NAMES if n in self._partitions]

        if not partition_names:
            return []

        # 打印路由信息
        print(f"\n{'─' * 60}")
        print(f"  [分区路由] query=\"{req.query[:60]}\"  "
              f"target={partition_names}  strategy={req.strategy.value}  top_k={req.top_k}")
        print(f"{'─' * 60}")

        # 多分区检索
        all_hits: List[RetrievalHit] = []
        for name in partition_names:
            api = self._partitions[name]
            _t_part = _time.time()

            # 每个分区用自己的 top_k 检索
            part_req = RetrievalRequest(
                query=req.query,
                strategy=req.strategy,
                top_k=req.top_k * 2,  # 多取一些，合并时再截断
                filters=req.filters,
                bm25_k=req.bm25_k,
                vector_k=req.vector_k,
                rrf_k=req.rrf_k,
                rerank=req.rerank,
                rerank_k=req.rerank_k,
                exact_mode=req.exact_mode,
                expand_context=req.expand_context,
                include_content_raw=req.include_content_raw,
                include_evidence=req.include_evidence,
                max_chars_per_hit=req.max_chars_per_hit,
            )
            hits = api.search_request(part_req)
            print(f"  [{name}] 返回 {len(hits)} 条  ({_time.time() - _t_part:.3f}s)")
            all_hits.extend(hits)

        # 合并排序：按 score 降序
        all_hits.sort(key=lambda h: h.score, reverse=True)

        # 重新编号
        merged = all_hits[:req.top_k]
        for i, h in enumerate(merged, 1):
            h.rank = i

        _total = _time.time() - _t0
        print(f"  [分区合并] {len(partition_names)} 个分区 → {len(merged)} 条命中  "
              f"({_total:.3f}s)")
        print(f"{'─' * 60}\n")

        return merged

    # ════════════════════════════════════════════════════════════
    # 委托方法 — 透传到各分区
    # ════════════════════════════════════════════════════════════

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """在所有分区中查找 chunk"""
        for api in self._partitions.values():
            result = api.get_chunk(chunk_id)
            if result:
                return result
        return None

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        for api in self._partitions.values():
            result = api.get_document(doc_id)
            if result:
                return result
        return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有分区的文档"""
        all_docs = []
        for name, api in self._partitions.items():
            for doc in api.list_documents():
                doc["_partition"] = name
                all_docs.append(doc)
        return all_docs

    def search_chunks(self, **filters) -> List[Dict[str, Any]]:
        """跨分区查询 chunks（元数据过滤）"""
        limit = filters.pop("limit", 100)
        all_results = []
        for api in self._partitions.values():
            results = api.search_chunks(**filters, limit=limit)
            all_results.extend(results)
        return all_results[:limit]

    def format_for_llm(self, results, max_chars: int = 3000) -> str:
        """格式化检索结果为 LLM 上下文"""
        # 使用第一个可用分区的 format_for_llm
        for api in self._partitions.values():
            return api.format_for_llm(results, max_chars)
        return ""

    # ════════════════════════════════════════════════════════════
    # 属性
    # ════════════════════════════════════════════════════════════

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def partitions(self) -> Dict[str, RetrievalAPI]:
        """各分区的 RetrievalAPI 实例"""
        return self._partitions

    @property
    def chunk_count(self) -> int:
        return sum(api.chunk_count for api in self._partitions.values())

    @property
    def doc_count(self) -> int:
        return sum(api.doc_count for api in self._partitions.values())

    @property
    def stats(self) -> Dict[str, Any]:
        """分区统计"""
        return {
            name: {
                "docs": api.doc_count,
                "chunks": api.chunk_count,
                "loaded": api.is_loaded,
            }
            for name, api in self._partitions.items()
        }


# ============================================================
# 便捷入口
# ============================================================

def load_partitions(base_dir: str = "regulatory_docs",
                    **kwargs) -> IndustryPartitionAPI:
    """一行加载三个分区"""
    api = IndustryPartitionAPI(base_dir=base_dir, **kwargs)
    api.load()
    return api