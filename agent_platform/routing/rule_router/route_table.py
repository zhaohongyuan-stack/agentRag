"""
路由决策表 — YAML 配置，可热更新

定义意图 → 检索通道、预算、复杂度的映射关系。
Phase 1 使用内置默认值，Phase 2+ 可从 YAML 文件加载。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class RouteDecision:
    """
    路由决策结果

    包含检索通道、预算配置、复杂度级别等信息。
    """

    intent: str
    level: str  # L0-L4
    channels: List[str] = field(default_factory=list)
    top_k: int = 10
    rerank: bool = False
    rerank_top_n: int = 0
    budget_ms: int = 5000
    need_clarification: bool = False
    need_decomposition: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "level": self.level,
            "channels": self.channels,
            "top_k": self.top_k,
            "rerank": self.rerank,
            "rerank_top_n": self.rerank_top_n,
            "budget_ms": self.budget_ms,
            "need_clarification": self.need_clarification,
            "need_decomposition": self.need_decomposition,
            "description": self.description,
        }


# ============================================================
# 默认路由决策表
# 与 contracts/budget/budget_config.yaml 对齐
# ============================================================
DEFAULT_ROUTE_TABLE: Dict[str, RouteDecision] = {
    "greeting": RouteDecision(
        intent="greeting",
        level="L0",
        channels=[],
        top_k=0,
        rerank=False,
        budget_ms=1000,
        description="问候/打招呼，直接回复",
    ),
    "clause_query": RouteDecision(
        intent="clause_query",
        level="L1",
        channels=["exact", "metadata"],
        top_k=5,
        rerank=False,
        budget_ms=2000,
        description="条款查询，精确文号/条款号",
    ),
    "definition": RouteDecision(
        intent="definition",
        level="L2",
        channels=["lexical", "dense", "metadata"],
        top_k=20,
        rerank=True,
        rerank_top_n=8,
        budget_ms=5000,
        description="定义查询，普通事实查询",
    ),
    "threshold": RouteDecision(
        intent="threshold",
        level="L2",
        channels=["lexical", "dense", "metadata"],
        top_k=20,
        rerank=True,
        rerank_top_n=8,
        budget_ms=5000,
        description="阈值查询，普通事实查询",
    ),
    "table_lookup": RouteDecision(
        intent="table_lookup",
        level="L2",
        channels=["table", "metadata"],
        top_k=10,
        rerank=False,
        budget_ms=3000,
        description="表格取数，结构化查询",
    ),
    "comparison": RouteDecision(
        intent="comparison",
        level="L3",
        channels=["hybrid", "table", "metadata"],
        top_k=20,
        rerank=True,
        rerank_top_n=10,
        budget_ms=15000,
        need_decomposition=True,
        description="比较查询，需拆解为多个子问题",
    ),
    "compliance": RouteDecision(
        intent="compliance",
        level="L4",
        channels=["hybrid", "exact", "table", "neighborhood", "relation"],
        top_k=30,
        rerank=True,
        rerank_top_n=10,
        budget_ms=30000,
        need_decomposition=True,
        description="合规查询，多跳推理",
    ),
    "overview": RouteDecision(
        intent="overview",
        level="L2",
        channels=["dense", "metadata"],
        top_k=15,
        rerank=False,
        budget_ms=5000,
        description="概览查询，文档摘要",
    ),
    "unknown": RouteDecision(
        intent="unknown",
        level="L2",
        channels=["hybrid", "metadata"],
        top_k=20,
        rerank=True,
        rerank_top_n=8,
        budget_ms=5000,
        description="未知意图，默认混合检索",
    ),
}


class RouteTable:
    """
    路由决策表管理器

    支持从 YAML 文件加载或使用内置默认值。
    Phase 1 使用内置默认值，Phase 2+ 支持热更新。
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 配置文件路径，为 None 时使用内置默认值
        """
        self._table: Dict[str, RouteDecision] = dict(DEFAULT_ROUTE_TABLE)
        if config_path and os.path.exists(config_path):
            self._load_from_yaml(config_path)

    def get(self, intent: str) -> RouteDecision:
        """
        根据意图获取路由决策

        Args:
            intent: 意图类型

        Returns:
            RouteDecision 对象，未匹配时返回 unknown 的默认决策
        """
        return self._table.get(intent, self._table["unknown"])

    def update(self, intent: str, decision: RouteDecision) -> None:
        """更新路由决策（支持热更新）"""
        self._table[intent] = decision

    def list_intents(self) -> List[str]:
        """列出所有已配置的意图类型"""
        return sorted(self._table.keys())

    def _load_from_yaml(self, path: str) -> None:
        """从 YAML 文件加载路由决策表"""
        if not _HAS_YAML:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "route_table" not in data:
            return
        for intent, cfg in data["route_table"].items():
            self._table[intent] = RouteDecision(
                intent=intent,
                level=cfg.get("level", "L2"),
                channels=cfg.get("channels", ["hybrid"]),
                top_k=cfg.get("top_k", 10),
                rerank=cfg.get("rerank", False),
                rerank_top_n=cfg.get("rerank_top_n", 0),
                budget_ms=cfg.get("budget_ms", 5000),
                need_clarification=cfg.get("need_clarification", False),
                need_decomposition=cfg.get("need_decomposition", False),
                description=cfg.get("description", ""),
            )
