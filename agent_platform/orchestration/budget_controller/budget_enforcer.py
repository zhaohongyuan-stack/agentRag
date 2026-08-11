"""
预算执行器 — BudgetEnforcer

将 BudgetController 产出的 BudgetAction 转化为可执行的策略决策，
并在超预算或降级时记录详细日志，供审计与可观测性使用。

设计要点:
  1. 动作转换: enforce() 将枚举动作转为字符串信号，供调度器消费
  2. 状态缓存: 缓存最近一次动作，便于 should_stop / should_downgrade 快速判定
  3. 详细日志: STOP 与 DOWNGRADE 均记录 WARNING 级别日志，含上下文信息
  4. 解耦: 不直接操作预算数值，只消费 BudgetAction，与 BudgetController 解耦

使用方式:
    from agent_platform.orchestration.budget_controller import (
        BudgetController, BudgetEnforcer,
    )

    bc = BudgetController("P3")
    enforcer = BudgetEnforcer()
    action = bc.consume_retrieval_round()
    signal = enforcer.enforce(action, context={"path": "P3"})
    if enforcer.should_stop():
        ...  # 终止执行
"""

import logging
from typing import Any, Dict, Optional

from .budget_models import BudgetAction

logger = logging.getLogger(__name__)


class BudgetEnforcer:
    """
    预算执行器 — 将预算动作转化为执行策略决策

    接收 BudgetController.check_budget() 或各 consume_* 方法返回的
    BudgetAction，转换为字符串信号并记录日志。调度器据此决定是否
    停止执行或降级策略。

    Attributes:
        last_action: 最近一次处理的预算动作
        stop_count: 触发 STOP 的累计次数
        downgrade_count: 触发 DOWNGRADE 的累计次数
    """

    def __init__(self):
        """初始化预算执行器"""
        self.last_action: BudgetAction = BudgetAction.CONTINUE
        self.stop_count: int = 0
        self.downgrade_count: int = 0
        # 缓存最近一次超预算时的上下文，便于诊断
        self._last_context: Optional[Dict[str, Any]] = None

    # ============================================================
    # 动作执行
    # ============================================================

    def enforce(
        self,
        action: BudgetAction,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        执行预算动作，返回对应的字符串信号

        - CONTINUE: 返回 "continue"，不打日志
        - STOP: 返回 "stop"，记录 WARNING 日志（硬限制触发）
        - DOWNGRADE: 返回 "downgrade"，记录 WARNING 日志（软阈值触发）

        Args:
            action: 预算控制器产出的动作
            context: 附加上下文（如路径、维度、消耗详情），用于日志诊断

        Returns:
            字符串信号: "continue" / "stop" / "downgrade"
        """
        self.last_action = action
        self._last_context = context

        if action == BudgetAction.CONTINUE:
            # 正常执行无需告警，保持 debug 级别
            logger.debug("预算检查通过，继续执行 context=%s", context)
            return "continue"

        if action == BudgetAction.STOP:
            self.stop_count += 1
            logger.warning(
                "预算硬限制触发，必须停止执行。action=stop count=%d context=%s",
                self.stop_count,
                context,
            )
            return "stop"

        if action == BudgetAction.DOWNGRADE:
            self.downgrade_count += 1
            logger.warning(
                "预算接近上限，建议降级执行策略。action=downgrade count=%d context=%s",
                self.downgrade_count,
                context,
            )
            return "downgrade"

        # 理论上不会到达，防御性兜底
        logger.error("未知预算动作 action=%s context=%s", action, context)
        return "continue"

    # ============================================================
    # 状态查询
    # ============================================================

    def should_stop(self) -> bool:
        """
        是否应该停止执行

        Returns:
            最近一次动作是否为 STOP
        """
        return self.last_action == BudgetAction.STOP

    def should_downgrade(self) -> bool:
        """
        是否应该降级执行策略

        Returns:
            最近一次动作是否为 DOWNGRADE
        """
        return self.last_action == BudgetAction.DOWNGRADE

    def should_continue(self) -> bool:
        """
        是否应该继续执行

        Returns:
            最近一次动作是否为 CONTINUE
        """
        return self.last_action == BudgetAction.CONTINUE

    def get_last_context(self) -> Optional[Dict[str, Any]]:
        """
        获取最近一次超预算时的上下文

        Returns:
            上下文字典，若无则为 None
        """
        return self._last_context

    def get_stats(self) -> Dict[str, int]:
        """
        获取执行统计

        Returns:
            含 stop_count / downgrade_count 的统计字典
        """
        return {
            "stop_count": self.stop_count,
            "downgrade_count": self.downgrade_count,
        }

    def reset(self) -> None:
        """重置执行器状态，清空动作缓存与计数"""
        self.last_action = BudgetAction.CONTINUE
        self.stop_count = 0
        self.downgrade_count = 0
        self._last_context = None
        logger.debug("预算执行器状态已重置")

    def __repr__(self) -> str:
        return (
            f"BudgetEnforcer(last_action={self.last_action.value}, "
            f"stop_count={self.stop_count}, "
            f"downgrade_count={self.downgrade_count})"
        )
