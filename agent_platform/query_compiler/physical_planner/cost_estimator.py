"""
算子成本估计 — 估算物理计划的执行成本

职责:
  1. 基于静态成本表为每个检索算子（通道）估算延迟、成功率、证据增益
  2. 将阶段内并行通道与重排算子聚合为阶段级成本
  3. 将多阶段成本聚合为整计划成本（PlanCost）
  4. 输出加权总成本，供 PlanOptimizer 选择最优计划

成本模型说明:
  - 延迟（latency_ms）: 阶段内通道并行取 max，重排串行叠加；阶段间累加
  - 证据增益（estimated_gain）: 通道间按 OR 组合（1 - Π(1-g)），
    重排按残差增益叠加；阶段间再按 OR 组合，封顶 1.0
  - 成功率（success_rate）: 通道间按 OR 组合（至少一个成功），
    重排成功率相乘（重排必须成功）；阶段间相乘（所有阶段须成功）
  - 总成本（total_cost）: latency + 增益不足惩罚 + 失败率惩罚，
    数值越低越优

模式参考: physical_planner/planner.py 的 dataclass + to_dict 风格
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .planner import PhysicalPlan


logger = logging.getLogger(__name__)


# ============================================================
# 静态成本表（第一版固定估值，后续可接入实测统计）
# ============================================================
OPERATOR_COSTS: Dict[str, Dict[str, float]] = {
    "exact":    {"latency_ms": 50,  "success_rate": 0.95, "evidence_gain": 0.8},
    "lexical":  {"latency_ms": 100, "success_rate": 0.85, "evidence_gain": 0.6},
    "dense":    {"latency_ms": 200, "success_rate": 0.80, "evidence_gain": 0.5},
    "metadata": {"latency_ms": 30,  "success_rate": 0.99, "evidence_gain": 0.3},
    "table":    {"latency_ms": 150, "success_rate": 0.90, "evidence_gain": 0.7},
    "relation": {"latency_ms": 300, "success_rate": 0.75, "evidence_gain": 0.6},
    "rerank":   {"latency_ms": 500, "success_rate": 0.95, "evidence_gain": 0.2},
}

# 成本权重（用于加权计算 total_cost）
LATENCY_WEIGHT: float = 1.0        # 延迟权重（毫秒，1:1 计入）
GAIN_PENALTY_WEIGHT: float = 200.0  # 增益不足惩罚权重（毫秒等价）
FAILURE_PENALTY_WEIGHT: float = 500.0  # 失败率惩罚权重（毫秒等价）

# 未知算子的兜底成本
DEFAULT_OPERATOR_COST: Dict[str, float] = {
    "latency_ms": 200,
    "success_rate": 0.80,
    "evidence_gain": 0.4,
}


@dataclass
class PlanCost:
    """
    物理计划成本估计结果

    Attributes:
        latency_ms: 整计划端到端延迟（毫秒）
        estimated_gain: 整计划证据增益（0~1，越高越好）
        success_rate: 整计划成功率（0~1，越高越好）
        total_cost: 加权总成本（越低越优），由延迟/增益/成功率加权得到
        breakdown: 每个算子的成本明细，键为 "{stage}:{channel}"
    """

    latency_ms: int = 0
    estimated_gain: float = 0.0
    success_rate: float = 0.0
    total_cost: float = 0.0
    breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "estimated_gain": self.estimated_gain,
            "success_rate": self.success_rate,
            "total_cost": self.total_cost,
            "breakdown": {k: dict(v) for k, v in self.breakdown.items()},
        }


class CostEstimator:
    """
    算子成本估计器

    根据静态成本表估算物理计划的延迟、证据增益、成功率与加权总成本。

    用法:
        estimator = CostEstimator()
        cost = estimator.estimate(physical_plan)
        print(cost.latency_ms, cost.estimated_gain, cost.total_cost)
    """

    def __init__(
        self,
        operator_costs: Dict[str, Dict[str, float]] = None,
        latency_weight: float = LATENCY_WEIGHT,
        gain_penalty_weight: float = GAIN_PENALTY_WEIGHT,
        failure_penalty_weight: float = FAILURE_PENALTY_WEIGHT,
    ):
        """
        Args:
            operator_costs: 自定义算子成本表，为 None 时使用默认 OPERATOR_COSTS。
                            传入时会与默认表合并（自定义覆盖同名算子）。
            latency_weight: 延迟权重
            gain_penalty_weight: 增益不足惩罚权重
            failure_penalty_weight: 失败率惩罚权重
        """
        self._costs: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in OPERATOR_COSTS.items()
        }
        if operator_costs:
            for key, val in operator_costs.items():
                self._costs[key] = dict(val)
        self._latency_weight = latency_weight
        self._gain_penalty_weight = gain_penalty_weight
        self._failure_penalty_weight = failure_penalty_weight

    def estimate(self, plan: PhysicalPlan) -> PlanCost:
        """
        估算物理计划的成本

        遍历 plan.stages 中的所有 channels 与 rerank 操作，累加/聚合得到
        延迟、证据增益、成功率，并计算加权总成本。

        Args:
            plan: 物理执行计划

        Returns:
            PlanCost 对象
        """
        breakdown: Dict[str, Dict[str, float]] = {}
        stage_latencies: List[int] = []
        stage_gains: List[float] = []
        stage_successes: List[float] = []

        for stage in plan.stages:
            # 通道成本（阶段内并行检索）
            channel_latencies: List[int] = []
            channel_gains: List[float] = []
            channel_successes: List[float] = []

            for ch in stage.channels:
                cost = self._lookup_cost(ch)
                breakdown[f"{stage.name}:{ch}"] = dict(cost)
                channel_latencies.append(int(cost["latency_ms"]))
                channel_gains.append(float(cost["evidence_gain"]))
                channel_successes.append(float(cost["success_rate"]))

            # 重排算子（串行叠加在检索之后）
            rerank_cost = None
            if stage.rerank:
                rerank_cost = self._lookup_cost("rerank")
                breakdown[f"{stage.name}:rerank"] = dict(rerank_cost)

            # 阶段延迟：通道并行取 max，重排串行叠加
            channel_latency = max(channel_latencies) if channel_latencies else 0
            rerank_latency = (
                int(rerank_cost["latency_ms"]) if rerank_cost else 0
            )
            stage_latency = channel_latency + rerank_latency
            stage_latencies.append(stage_latency)

            # 阶段证据增益：通道 OR 组合 + 重排残差增益
            channel_gain = self._or_combine(channel_gains)
            if rerank_cost is not None:
                rerank_gain = float(rerank_cost["evidence_gain"])
                # 重排对剩余不确定性提供残差增益
                channel_gain = channel_gain + (1.0 - channel_gain) * rerank_gain
            stage_gains.append(channel_gain)

            # 阶段成功率：通道 OR 组合（至少一个成功），重排成功率相乘
            channel_success = self._or_combine(channel_successes)
            if rerank_cost is not None:
                channel_success = channel_success * float(
                    rerank_cost["success_rate"]
                )
            stage_successes.append(channel_success)

        # 整计划聚合
        total_latency = sum(stage_latencies)
        # 阶段间证据增益按 OR 组合（多阶段检索互补）
        estimated_gain = min(self._or_combine(stage_gains), 1.0)
        # 阶段间成功率相乘（所有阶段须成功）
        success_rate = self._multiply(stage_successes)

        total_cost = self._compute_total_cost(
            total_latency, estimated_gain, success_rate
        )

        cost = PlanCost(
            latency_ms=total_latency,
            estimated_gain=estimated_gain,
            success_rate=success_rate,
            total_cost=total_cost,
            breakdown=breakdown,
        )
        logger.info(
            "成本估计完成: latency=%dms, gain=%.4f, success=%.4f, cost=%.2f, "
            "stages=%d, operators=%d",
            total_latency,
            estimated_gain,
            success_rate,
            total_cost,
            len(plan.stages),
            len(breakdown),
        )
        return cost

    # ============================================================
    # 内部方法
    # ============================================================

    def _lookup_cost(self, operator: str) -> Dict[str, float]:
        """
        查询算子成本

        未知算子回退到 DEFAULT_OPERATOR_COST 并记录警告。
        """
        cost = self._costs.get(operator)
        if cost is None:
            logger.warning("未知算子 '%s'，使用兜底成本", operator)
            return dict(DEFAULT_OPERATOR_COST)
        return dict(cost)

    @staticmethod
    def _or_combine(values: List[float]) -> float:
        """
        OR 组合（概率并集）

        1 - Π(1 - v_i)，表示至少一个成功的概率。空列表返回 0.0。
        """
        if not values:
            return 0.0
        product = 1.0
        for v in values:
            product *= (1.0 - v)
        return 1.0 - product

    @staticmethod
    def _multiply(values: List[float]) -> float:
        """
        连乘组合

        空列表返回 1.0（无约束）。
        """
        result = 1.0
        for v in values:
            result *= v
        return result

    def _compute_total_cost(
        self,
        latency_ms: int,
        estimated_gain: float,
        success_rate: float,
    ) -> float:
        """
        计算加权总成本

        total_cost = latency * w_latency
                   + gain_penalty * (1 - gain)
                   + failure_penalty * (1 - success)

        延迟越低、增益越高、成功率越高 → 总成本越低（越优）。
        """
        return (
            self._latency_weight * latency_ms
            + self._gain_penalty_weight * (1.0 - estimated_gain)
            + self._failure_penalty_weight * (1.0 - success_rate)
        )
