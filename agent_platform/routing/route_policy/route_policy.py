"""
综合路由策略 — 规则路由 + 风险路由 → 执行路径（P0-P4）

组合三层路由决策:
  第一层 — 规则路由（RuleRouter）: 意图 + 复杂度（L0-L4）
  第二层 — 风险路由（RiskRouter）: 风险级别（low/medium/high）
  第三层 — 执行路径映射: 复杂度 + 风险 → P0-P4

执行路径定义:
  P0: 高置信度重复问题 → 缓存优先，无需检索
  P1: 精确字段明确 → 单路精确检索
  P2: 普通事实查询 → 多路检索 + 重排
  P3: 多路检索 → 扩展通道 + 重排
  P4: 复杂分析 → 全通道 + 拆解 + 重排

映射逻辑:
  L0 greeting           → P0
  L1 + low risk         → P1
  L1 + medium/high risk → P2
  L2 + low risk         → P2
  L2 + medium/high risk → P3
  L3 + low risk         → P3
  L3 + medium/high risk → P4
  L4                    → P4
  need_clarification    → 仍然路由但标记需澄清

来源: M2.2 开发计划 + 问题确认.md
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..risk_router.risk_router import RiskAssessment, RiskRouter
from ..rule_router.route_table import RouteDecision
from ..rule_router.router import RuleRouter
from .path_table import DEFAULT_EXECUTION_PATHS, ExecutionPath
from .policy_loader import PolicyLoader


# ============================================================
# ComprehensiveRouteDecision 数据结构
# ============================================================
@dataclass
class ComprehensiveRouteDecision:
    """
    综合路由决策结果

    包含规则路由决策 + 风险评估 + 执行路径的完整信息。
    """

    # ── 来自 RuleRouter 的字段 ──
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

    # ── 风险评估字段 ──
    risk_level: str = "medium"  # low/medium/high
    risk_assessment: Optional[RiskAssessment] = None

    # ── 执行路径字段 ──
    path_id: str = "P2"  # P0-P4
    execution_path: Optional[ExecutionPath] = None

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
            "risk_level": self.risk_level,
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
            "path_id": self.path_id,
            "execution_path": self.execution_path.to_dict() if self.execution_path else None,
        }


# ============================================================
# 风险级别排序辅助
# ============================================================
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _is_high_risk(risk_level: str) -> bool:
    """判断是否为高风险（medium 或 high）"""
    return _RISK_ORDER.get(risk_level, 1) >= 1


def _is_medium_or_high_risk(risk_level: str) -> bool:
    """判断是否为中等或高风险"""
    return risk_level in ("medium", "high")


# ============================================================
# RoutePolicy 主类
# ============================================================
class RoutePolicy:
    """
    综合路由策略

    组合 RuleRouter（意图+复杂度）和 RiskRouter（风险评级），
    输出包含执行路径（P0-P4）的综合路由决策。

    用法:
        policy = RoutePolicy()
        decision = policy.decide(query_spec)
        print(decision.path_id)  # P0-P4
    """

    def __init__(
        self,
        rule_router: Optional[RuleRouter] = None,
        risk_router: Optional[RiskRouter] = None,
        policy_loader: Optional[PolicyLoader] = None,
    ):
        """
        Args:
            rule_router: 规则路由器，为 None 时使用默认 RuleRouter
            risk_router: 风险路由器，为 None 时使用默认 RiskRouter
            policy_loader: 策略加载器，为 None 时使用默认 PolicyLoader
        """
        self._rule_router = rule_router or RuleRouter()
        self._risk_router = risk_router or RiskRouter()
        self._policy_loader = policy_loader or PolicyLoader()
        # 加载执行路径
        self._paths: Dict[str, ExecutionPath] = self._policy_loader.load()

    def decide(self, query_spec: Any) -> ComprehensiveRouteDecision:
        """
        综合路由决策

        Args:
            query_spec: QuerySpec 对象（或兼容的 dict）

        Returns:
            ComprehensiveRouteDecision 对象
        """
        # 1. 规则路由: 意图 + 复杂度
        route_decision: RouteDecision = self._rule_router.route(query_spec)

        # 2. 风险路由: 风险评级
        risk_assessment: RiskAssessment = self._risk_router.assess(query_spec)

        # 3. 执行路径映射: 复杂度 + 风险 → P0-P4
        path_id = self._map_to_path(
            level=route_decision.level,
            risk_level=risk_assessment.level,
            intent=route_decision.intent,
        )
        execution_path = self._paths.get(path_id, self._paths.get("P2"))

        # 4. 构建综合决策
        # 执行路径的配置覆盖规则路由的通道/预算（执行路径优先级更高）
        # 但保留 need_clarification 标记
        need_clarification = route_decision.need_clarification
        need_decomposition = execution_path.need_decomposition or route_decision.need_decomposition

        return ComprehensiveRouteDecision(
            # 规则路由字段
            intent=route_decision.intent,
            level=route_decision.level,
            channels=execution_path.channels if execution_path else route_decision.channels,
            top_k=execution_path.top_k if execution_path else route_decision.top_k,
            rerank=execution_path.rerank if execution_path else route_decision.rerank,
            rerank_top_n=execution_path.rerank_top_n if execution_path else route_decision.rerank_top_n,
            budget_ms=execution_path.budget_ms if execution_path else route_decision.budget_ms,
            need_clarification=need_clarification,
            need_decomposition=need_decomposition,
            description=execution_path.description if execution_path else route_decision.description,
            # 风险评估字段
            risk_level=risk_assessment.level,
            risk_assessment=risk_assessment,
            # 执行路径字段
            path_id=path_id,
            execution_path=execution_path,
        )

    # ----------------------------------------------------------
    # 内部方法: 复杂度 + 风险 → 执行路径映射
    # ----------------------------------------------------------
    def _map_to_path(self, level: str, risk_level: str, intent: str) -> str:
        """
        根据复杂度级别和风险级别映射到执行路径

        映射规则:
          L0 greeting           → P0
          L1 + low risk         → P1
          L1 + medium/high risk → P2
          L2 + low risk         → P2
          L2 + medium/high risk → P3
          L3 + low risk         → P3
          L3 + medium/high risk → P4
          L4                    → P4

        Args:
            level: 复杂度级别（L0-L4）
            risk_level: 风险级别（low/medium/high）
            intent: 查询意图

        Returns:
            执行路径 ID（P0-P4）
        """
        # L0 → P0（问候/打招呼，无需检索）
        if level == "L0":
            return "P0"

        # L4 → P4（合规/复杂查询，多跳推理）
        if level == "L4":
            return "P4"

        # L1: 精确字段明确
        if level == "L1":
            if risk_level == "low":
                return "P1"
            else:
                # medium/high risk → 升级到 P2
                return "P2"

        # L2: 普通事实查询
        if level == "L2":
            if risk_level == "low":
                return "P2"
            else:
                # medium/high risk → 升级到 P3
                return "P3"

        # L3: 比较/多路检索
        if level == "L3":
            if risk_level == "low":
                return "P3"
            else:
                # medium/high risk → 升级到 P4
                return "P4"

        # 默认兜底
        return "P2"

    # ----------------------------------------------------------
    # 热更新支持
    # ----------------------------------------------------------
    def reload(self) -> None:
        """从文件重新加载执行路径配置（热更新）"""
        self._paths = self._policy_loader.reload()

    def get_path(self, path_id: str) -> Optional[ExecutionPath]:
        """获取指定执行路径"""
        return self._paths.get(path_id)

    def list_paths(self) -> List[str]:
        """列出所有执行路径 ID"""
        return sorted(self._paths.keys())
