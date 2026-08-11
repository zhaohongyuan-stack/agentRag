"""
计划优化器 — 生成多个候选物理计划并选择成本最优者

职责:
  1. 基于 QueryIR 生成多个候选物理计划（基础 / 精简 / 激进）
  2. 使用 CostEstimator 估算每个候选的成本
  3. 选择 total_cost 最低的候选作为最终计划

候选策略:
  - 基础计划: 使用默认 PLAN_TEMPLATES
  - 精简计划: 去除低证据增益通道（evidence_gain < 阈值），降低延迟
  - 激进计划: 启用全部检索通道 + 重排，最大化召回（成本最高）

模式参考: physical_planner/planner.py 的 dataclass + to_dict 风格
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..logical_planner.planner import LogicalPlan, LogicalPlanner
from ..query_ir.ir_builder import QueryIR
from .cost_estimator import OPERATOR_COSTS, CostEstimator, PlanCost
from .planner import (
    DEFAULT_TEMPLATE,
    PLAN_TEMPLATES,
    PhysicalPlan,
    PhysicalPlanner,
)


logger = logging.getLogger(__name__)


# 全部检索通道（激进计划使用）
ALL_RETRIEVAL_CHANNELS: List[str] = [
    "exact",
    "lexical",
    "dense",
    "metadata",
    "table",
]

# 精简计划的证据增益阈值：低于此值的通道会被裁剪
LEAN_GAIN_THRESHOLD: float = 0.5


class PlanOptimizer:
    """
    计划优化器

    生成多个候选物理计划，基于成本估计选择最优者。

    用法:
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(query_ir)
        best = optimizer.select_best(candidates)
        print(best.plan_id, best.stages)
    """

    def __init__(self, cost_estimator: CostEstimator = None):
        """
        Args:
            cost_estimator: 成本估计器，为 None 时使用默认 CostEstimator。
        """
        self._cost_estimator = cost_estimator or CostEstimator()
        self._logical_planner = LogicalPlanner()

    def generate_candidates(self, query_ir: QueryIR) -> List[PhysicalPlan]:
        """
        生成多个候选物理计划

        策略:
          1. 基础计划: 默认模板
          2. 精简计划: 裁剪低增益通道（与基础计划通道不同时才生成）
          3. 激进计划: 全部检索通道 + 重排

        Args:
            query_ir: 查询中间表示

        Returns:
            候选物理计划列表（2~3 个）
        """
        logical_plan = self._logical_planner.plan(query_ir)
        intent = query_ir.intent

        base_template = self._base_template(intent)
        candidates: List[PhysicalPlan] = []

        # 1. 基础计划：使用默认模板
        base_plan = PhysicalPlanner().plan(logical_plan, query_ir)
        candidates.append(base_plan)
        logger.debug("生成基础计划: channels=%s", base_template["channels"])

        # 2. 精简计划：去除低增益通道
        lean_channels = self._lean_channels(base_template["channels"])
        if lean_channels and lean_channels != list(base_template["channels"]):
            lean_template = {
                intent: {
                    **base_template,
                    "channels": lean_channels,
                }
            }
            lean_plan = PhysicalPlanner(templates=lean_template).plan(
                logical_plan, query_ir
            )
            candidates.append(lean_plan)
            logger.debug("生成精简计划: channels=%s", lean_channels)

        # 3. 激进计划：全部检索通道 + 重排
        aggressive_channels = ALL_RETRIEVAL_CHANNELS
        if aggressive_channels != list(base_template["channels"]):
            aggressive_template = {
                intent: {
                    "channels": list(aggressive_channels),
                    "top_k": base_template["top_k"],
                    "rerank": True,
                    "budget_ms": base_template["budget_ms"],
                }
            }
            aggressive_plan = PhysicalPlanner(
                templates=aggressive_template
            ).plan(logical_plan, query_ir)
            candidates.append(aggressive_plan)
            logger.debug("生成激进计划: channels=%s", aggressive_channels)

        logger.info(
            "候选计划生成完成: intent=%s, candidates=%d",
            intent,
            len(candidates),
        )
        return candidates

    def select_best(
        self, candidates: List[PhysicalPlan]
    ) -> Optional[PhysicalPlan]:
        """
        选择成本最低的候选计划

        计算每个候选的 PlanCost，返回 total_cost 最低者。
        候选为空时返回 None；成本相同时保留首个最低者（稳定）。

        Args:
            candidates: 候选物理计划列表

        Returns:
            成本最优的 PhysicalPlan，无候选时返回 None
        """
        if not candidates:
            logger.warning("候选计划为空，无法选择最优计划")
            return None

        best_plan: Optional[PhysicalPlan] = None
        best_cost: float = float("inf")

        for plan in candidates:
            cost = self._cost_estimator.estimate(plan)
            logger.debug(
                "候选计划成本: plan_id=%s, total_cost=%.2f",
                plan.plan_id,
                cost.total_cost,
            )
            if cost.total_cost < best_cost:
                best_cost = cost.total_cost
                best_plan = plan

        if best_plan is not None:
            logger.info(
                "选择最优计划: plan_id=%s, total_cost=%.2f",
                best_plan.plan_id,
                best_cost,
            )
        return best_plan

    def optimize(self, query_ir: QueryIR) -> Optional[PhysicalPlan]:
        """
        一站式优化：生成候选并选择最优

        Args:
            query_ir: 查询中间表示

        Returns:
            成本最优的 PhysicalPlan
        """
        candidates = self.generate_candidates(query_ir)
        return self.select_best(candidates)

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _base_template(intent: str) -> Dict[str, Any]:
        """
        获取基础模板

        优先使用 PLAN_TEMPLATES[intent]，未知意图回退到 DEFAULT_TEMPLATE。
        """
        template = PLAN_TEMPLATES.get(intent)
        if template is None:
            template = dict(DEFAULT_TEMPLATE)
        return dict(template)

    @staticmethod
    def _lean_channels(channels: List[str]) -> List[str]:
        """
        裁剪低增益通道

        保留 evidence_gain >= LEAN_GAIN_THRESHOLD 的通道。
        若裁剪后为空（全部低增益），则保留增益最高的单个通道，
        确保精简计划仍可执行。
        """
        kept = [
            ch
            for ch in channels
            if OPERATOR_COSTS.get(ch, {}).get("evidence_gain", 0.0)
            >= LEAN_GAIN_THRESHOLD
        ]
        if kept:
            return kept

        # 全部低增益时，保留增益最高的通道
        if not channels:
            return []
        best = max(
            channels,
            key=lambda ch: OPERATOR_COSTS.get(ch, {}).get(
                "evidence_gain", 0.0
            ),
        )
        return [best]
