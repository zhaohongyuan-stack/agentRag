"""
执行路径表 — ExecutionPath 数据结构与内置默认值

定义 P0-P4 执行路径的数据结构和默认配置。
与 execution_paths.yaml 对齐。

分离此模块以避免 route_policy.py 与 policy_loader.py 之间的循环导入。
模式参考: rule_router/route_table.py
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExecutionPath:
    """
    执行路径定义

    描述一条完整的检索执行路径，包含通道、预算、重排、拆解等配置。
    """

    path_id: str  # P0-P4
    description: str
    channels: List[str] = field(default_factory=list)
    top_k: int = 10
    rerank: bool = False
    rerank_top_n: int = 0
    budget_ms: int = 5000
    max_retries: int = 1
    need_decomposition: bool = False
    cache_first: bool = False
    retrieval: bool = True

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "description": self.description,
            "channels": self.channels,
            "top_k": self.top_k,
            "rerank": self.rerank,
            "rerank_top_n": self.rerank_top_n,
            "budget_ms": self.budget_ms,
            "max_retries": self.max_retries,
            "need_decomposition": self.need_decomposition,
            "cache_first": self.cache_first,
            "retrieval": self.retrieval,
        }


# ============================================================
# 内置默认执行路径
# 与 execution_paths.yaml 对齐
# ============================================================
DEFAULT_EXECUTION_PATHS: Dict[str, ExecutionPath] = {
    "P0": ExecutionPath(
        path_id="P0",
        description="高置信度重复问题",
        channels=[],
        top_k=0,
        rerank=False,
        rerank_top_n=0,
        budget_ms=100,
        max_retries=0,
        need_decomposition=False,
        cache_first=True,
        retrieval=False,
    ),
    "P1": ExecutionPath(
        path_id="P1",
        description="精确字段明确",
        channels=["exact", "metadata"],
        top_k=5,
        rerank=False,
        rerank_top_n=0,
        budget_ms=2000,
        max_retries=1,
        need_decomposition=False,
        cache_first=False,
        retrieval=True,
    ),
    "P2": ExecutionPath(
        path_id="P2",
        description="普通事实查询",
        channels=["lexical", "dense", "metadata"],
        top_k=20,
        rerank=True,
        rerank_top_n=8,
        budget_ms=5000,
        max_retries=1,
        need_decomposition=False,
        cache_first=False,
        retrieval=True,
    ),
    "P3": ExecutionPath(
        path_id="P3",
        description="多路检索",
        channels=["lexical", "dense", "metadata", "table"],
        top_k=30,
        rerank=True,
        rerank_top_n=10,
        budget_ms=8000,
        max_retries=2,
        need_decomposition=False,
        cache_first=False,
        retrieval=True,
    ),
    "P4": ExecutionPath(
        path_id="P4",
        description="复杂分析（合规判断/多跳比较）",
        channels=["lexical", "dense", "metadata", "table", "relation", "neighborhood"],
        top_k=50,
        rerank=True,
        rerank_top_n=15,
        budget_ms=15000,
        max_retries=3,
        need_decomposition=True,
        cache_first=False,
        retrieval=True,
    ),
}
