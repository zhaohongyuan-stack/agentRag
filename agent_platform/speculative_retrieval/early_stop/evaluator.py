"""
早停评估器 — EarlyStopEvaluator

在推测式检索过程中评估当前证据是否已足够，判断是否应提前停止
后续分支的执行，从而节省检索预算与延迟。

停止判据（优先级从高到低）:
  1. sufficient: 证据充分性已达阈值（默认 0.85）
  2. budget_exhausted: 检索预算已耗尽
  3. marginal_gain_low: 边际证据增益低于阈值（默认 0.05）

设计要点:
  1. 纯函数式评估: 不修改分支状态，仅返回 EarlyStopResult
  2. 边际增益守卫: 仅在存在上一轮评分（previous_score > 0）时计算
     边际增益判断，避免首轮误判
  3. 可配置阈值: 阈值为类属性，子类或实例可覆盖

模式参考: orchestration/budget_controller 的 dataclass + to_dict 风格
"""

import logging
from dataclasses import dataclass
from typing import List

from ..branch_launcher.launcher import RetrievalBranch

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopResult:
    """
    早停评估结果

    Attributes:
        should_stop: 是否应停止检索
        reason: 停止原因（sufficient / marginal_gain_low / budget_exhausted /
                none）
        current_sufficiency: 当前充分性评分
        marginal_gain: 本轮相对上一轮的边际增益
    """

    should_stop: bool
    reason: str
    current_sufficiency: float
    marginal_gain: float

    def to_dict(self) -> dict:
        """转换为字典，用于日志输出与序列化"""
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "current_sufficiency": self.current_sufficiency,
            "marginal_gain": self.marginal_gain,
        }


class EarlyStopEvaluator:
    """
    早停评估器

    判断是否应该停止检索:
      1. 证据充分性已达标
      2. 边际证据增益低于阈值
      3. 预算耗尽

    用法:
        evaluator = EarlyStopEvaluator()
        result = evaluator.evaluate(branches, sufficiency_score=0.9)
        if result.should_stop:
            ...  # 停止后续检索
    """

    # 边际增益阈值：低于该值认为继续检索收益甚微
    MARGINAL_GAIN_THRESHOLD = 0.05
    # 充分性阈值：达到该值认为证据已足够
    SUFFICIENCY_THRESHOLD = 0.85

    def evaluate(
        self,
        branches: List[RetrievalBranch],
        sufficiency_score: float = 0.0,
        previous_score: float = 0.0,
        budget_exhausted: bool = False,
    ) -> EarlyStopResult:
        """
        评估是否应该早停

        判定优先级: sufficient > budget_exhausted > marginal_gain_low。
        任一条件满足即返回 should_stop=True，并给出对应原因。

        Args:
            branches: 所有检索分支（用于判断是否有可用结果）
            sufficiency_score: 当前充分性评分
            previous_score: 上一轮评分（用于计算边际增益）
            budget_exhausted: 预算是否已耗尽

        Returns:
            EarlyStopResult 评估结果
        """
        marginal_gain = self.estimate_marginal_gain(
            sufficiency_score, previous_score
        )

        # 1. 证据充分性达标 → 停止
        if sufficiency_score >= self.SUFFICIENCY_THRESHOLD:
            logger.info(
                "早停：证据充分 sufficiency=%.4f >= %.2f",
                sufficiency_score,
                self.SUFFICIENCY_THRESHOLD,
            )
            return EarlyStopResult(
                should_stop=True,
                reason="sufficient",
                current_sufficiency=sufficiency_score,
                marginal_gain=marginal_gain,
            )

        # 2. 预算耗尽 → 停止
        if budget_exhausted:
            logger.info(
                "早停：预算耗尽 sufficiency=%.4f", sufficiency_score
            )
            return EarlyStopResult(
                should_stop=True,
                reason="budget_exhausted",
                current_sufficiency=sufficiency_score,
                marginal_gain=marginal_gain,
            )

        # 3. 边际增益过低 → 停止（仅在有上一轮评分时判断，避免首轮误判）
        if (
            previous_score > 0.0
            and marginal_gain < self.MARGINAL_GAIN_THRESHOLD
        ):
            logger.info(
                "早停：边际增益过低 gain=%.4f < %.2f",
                marginal_gain,
                self.MARGINAL_GAIN_THRESHOLD,
            )
            return EarlyStopResult(
                should_stop=True,
                reason="marginal_gain_low",
                current_sufficiency=sufficiency_score,
                marginal_gain=marginal_gain,
            )

        logger.debug(
            "继续检索 sufficiency=%.4f gain=%.4f branches=%d",
            sufficiency_score,
            marginal_gain,
            len(branches),
        )
        return EarlyStopResult(
            should_stop=False,
            reason="none",
            current_sufficiency=sufficiency_score,
            marginal_gain=marginal_gain,
        )

    def estimate_marginal_gain(
        self, current_score: float, previous_score: float
    ) -> float:
        """
        计算边际增益

        边际增益 = 当前评分 - 上一轮评分，下限为 0.0（不出现负增益）。

        Args:
            current_score: 当前充分性评分
            previous_score: 上一轮评分

        Returns:
            边际增益值（>= 0.0）
        """
        gain = current_score - previous_score
        return max(0.0, gain)
