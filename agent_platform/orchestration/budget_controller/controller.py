"""
预算控制器核心 — BudgetController

负责根据执行路径（P0-P4）分配预算，并在执行过程中跟踪各维度资源消耗，
据此给出 CONTINUE / STOP / DOWNGRADE 决策。

设计要点:
  1. 路径绑定: 每个控制器实例绑定一条执行路径，路径决定预算上限
  2. 细粒度计量: 检索轮次、改写、子任务、工具调用、token、耗时均可计量
  3. 软硬限制: 达到硬限制返回 STOP；达到软阈值（默认 80%）返回 DOWNGRADE
  4. 非破坏性: consume_* 方法只累加 consumed 并返回动作，不抛异常，
     由 BudgetEnforcer 决定如何处理

使用方式:
    from agent_platform.orchestration.budget_controller import BudgetController

    bc = BudgetController("P2")
    action = bc.consume_retrieval_round()
    if action == BudgetAction.STOP:
        ...  # 停止执行
"""

import logging
from typing import Dict

from .budget_models import (
    BUDGET_BY_PATH,
    BudgetAction,
    BudgetConsumed,
    ExecutionBudget,
)

logger = logging.getLogger(__name__)

# 降级阈值：当某维度消耗达到上限的该比例时，返回 DOWNGRADE
# 硬限制为 100%，软阈值用于提前降级以留出余量
_DOWNGRADE_RATIO = 0.8


class BudgetController:
    """
    预算控制器 — 分配与跟踪执行预算

    根据执行路径初始化预算上限，提供各维度的消耗方法，并在每次消耗后
    检查预算状态，返回对应的 BudgetAction。

    Attributes:
        path_id: 当前绑定的执行路径（P0-P4）
        budget: 当前执行预算
    """

    def __init__(self, path_id: str = "P2"):
        """
        初始化预算控制器

        Args:
            path_id: 执行路径标识（P0-P4），默认 P2
        """
        self.path_id: str = path_id
        self.budget: ExecutionBudget = self.allocate(path_id)
        logger.debug(
            "预算控制器已初始化 path=%s, budget=%s",
            path_id,
            self.budget.to_dict(),
        )

    # ============================================================
    # 预算分配
    # ============================================================

    def allocate(self, path_id: str) -> ExecutionBudget:
        """
        分配指定执行路径的预算

        重新绑定路径并重置消耗记录。若路径不存在则回退到 P2 并记录告警。

        Args:
            path_id: 执行路径标识（P0-P4）

        Returns:
            对应路径的 ExecutionBudget（全新实例，消耗记录清零）
        """
        self.path_id = path_id
        template = BUDGET_BY_PATH.get(path_id)
        if template is None:
            logger.warning(
                "未知执行路径 path_id=%s，回退到 P2 默认预算", path_id
            )
            template = BUDGET_BY_PATH["P2"]
        # 深拷贝：重置 consumed 为新的空记录，避免路径间共享状态
        self.budget = ExecutionBudget(
            max_retrieval_rounds=template.max_retrieval_rounds,
            max_rewrites=template.max_rewrites,
            max_subtasks=template.max_subtasks,
            max_tool_calls=template.max_tool_calls,
            total_timeout_ms=template.total_timeout_ms,
            total_token_budget=template.total_token_budget,
            per_step_timeout_ms=template.per_step_timeout_ms,
            consumed=BudgetConsumed(),
        )
        return self.budget

    # ============================================================
    # 资源消耗
    # ============================================================

    def consume_retrieval_round(self) -> BudgetAction:
        """
        消耗一次检索轮次

        Returns:
            消耗后的预算动作
        """
        self.budget.consumed.retrieval_rounds += 1
        action = self.check_budget()
        logger.debug(
            "消耗检索轮次 consumed=%d/%d action=%s",
            self.budget.consumed.retrieval_rounds,
            self.budget.max_retrieval_rounds,
            action.value,
        )
        return action

    def consume_rewrite(self) -> BudgetAction:
        """
        消耗一次查询改写

        Returns:
            消耗后的预算动作
        """
        self.budget.consumed.rewrites += 1
        action = self.check_budget()
        logger.debug(
            "消耗查询改写 consumed=%d/%d action=%s",
            self.budget.consumed.rewrites,
            self.budget.max_rewrites,
            action.value,
        )
        return action

    def consume_subtask(self) -> BudgetAction:
        """
        消耗一次子任务

        Returns:
            消耗后的预算动作
        """
        self.budget.consumed.subtasks += 1
        action = self.check_budget()
        logger.debug(
            "消耗子任务 consumed=%d/%d action=%s",
            self.budget.consumed.subtasks,
            self.budget.max_subtasks,
            action.value,
        )
        return action

    def consume_tool_call(self) -> BudgetAction:
        """
        消耗一次工具调用

        Returns:
            消耗后的预算动作
        """
        self.budget.consumed.tool_calls += 1
        action = self.check_budget()
        logger.debug(
            "消耗工具调用 consumed=%d/%d action=%s",
            self.budget.consumed.tool_calls,
            self.budget.max_tool_calls,
            action.value,
        )
        return action

    def consume_tokens(self, tokens: int) -> BudgetAction:
        """
        消耗指定数量的 token

        Args:
            tokens: 本次消耗的 token 数

        Returns:
            消耗后的预算动作
        """
        if tokens < 0:
            logger.warning("消耗 token 数为负值 tokens=%d，已忽略", tokens)
            return self.check_budget()
        self.budget.consumed.tokens += tokens
        action = self.check_budget()
        logger.debug(
            "消耗 token delta=%d consumed=%d/%d action=%s",
            tokens,
            self.budget.consumed.tokens,
            self.budget.total_token_budget,
            action.value,
        )
        return action

    def consume_time(self, ms: int) -> BudgetAction:
        """
        消耗指定时长（毫秒）

        Args:
            ms: 本次消耗的毫秒数

        Returns:
            消耗后的预算动作
        """
        if ms < 0:
            logger.warning("消耗时长为负值 ms=%d，已忽略", ms)
            return self.check_budget()
        self.budget.consumed.elapsed_ms += ms
        action = self.check_budget()
        logger.debug(
            "消耗耗时 delta=%d consumed=%d/%d action=%s",
            ms,
            self.budget.consumed.elapsed_ms,
            self.budget.total_timeout_ms,
            action.value,
        )
        return action

    # ============================================================
    # 预算检查
    # ============================================================

    def check_budget(self) -> BudgetAction:
        """
        检查当前预算状态

        判定优先级：硬限制（STOP）> 软阈值（DOWNGRADE）> 正常（CONTINUE）。
        任一维度达到硬限制即返回 STOP；任一维度达到软阈值即返回 DOWNGRADE。

        Returns:
            预算动作
        """
        c = self.budget.consumed
        b = self.budget

        # --- 硬限制：任一维度超限则必须停止 ---
        if c.retrieval_rounds > b.max_retrieval_rounds:
            return BudgetAction.STOP
        if c.rewrites > b.max_rewrites:
            return BudgetAction.STOP
        if c.subtasks > b.max_subtasks:
            return BudgetAction.STOP
        if c.tool_calls > b.max_tool_calls:
            return BudgetAction.STOP
        if c.tokens > b.total_token_budget:
            return BudgetAction.STOP
        if c.elapsed_ms > b.total_timeout_ms:
            return BudgetAction.STOP

        # --- 软阈值：达到 80% 即建议降级 ---
        if b.max_retrieval_rounds > 0 and _ratio(
            c.retrieval_rounds, b.max_retrieval_rounds
        ) >= _DOWNGRADE_RATIO:
            return BudgetAction.DOWNGRADE
        if b.max_rewrites > 0 and _ratio(
            c.rewrites, b.max_rewrites
        ) >= _DOWNGRADE_RATIO:
            return BudgetAction.DOWNGRADE
        if b.max_subtasks > 0 and _ratio(
            c.subtasks, b.max_subtasks
        ) >= _DOWNGRADE_RATIO:
            return BudgetAction.DOWNGRADE
        if b.max_tool_calls > 0 and _ratio(
            c.tool_calls, b.max_tool_calls
        ) >= _DOWNGRADE_RATIO:
            return BudgetAction.DOWNGRADE
        if b.total_token_budget > 0 and _ratio(
            c.tokens, b.total_token_budget
        ) >= _DOWNGRADE_RATIO:
            return BudgetAction.DOWNGRADE
        if b.total_timeout_ms > 0 and _ratio(
            c.elapsed_ms, b.total_timeout_ms
        ) >= _DOWNGRADE_RATIO:
            return BudgetAction.DOWNGRADE

        return BudgetAction.CONTINUE

    # ============================================================
    # 查询
    # ============================================================

    def get_remaining(self) -> Dict[str, int]:
        """
        获取各维度剩余预算

        Returns:
            各维度剩余量的字典，键与 BudgetConsumed 字段对齐
        """
        c = self.budget.consumed
        b = self.budget
        return {
            "retrieval_rounds": max(0, b.max_retrieval_rounds - c.retrieval_rounds),
            "rewrites": max(0, b.max_rewrites - c.rewrites),
            "subtasks": max(0, b.max_subtasks - c.subtasks),
            "tool_calls": max(0, b.max_tool_calls - c.tool_calls),
            "tokens": max(0, b.total_token_budget - c.tokens),
            "elapsed_ms": max(0, b.total_timeout_ms - c.elapsed_ms),
        }

    def get_summary(self) -> Dict[str, dict]:
        """
        获取预算消耗摘要

        包含路径、上限、已消耗、剩余三个维度，便于日志和监控输出。

        Returns:
            摘要字典，含 path / limits / consumed / remaining
        """
        return {
            "path": self.path_id,
            "limits": {
                "max_retrieval_rounds": self.budget.max_retrieval_rounds,
                "max_rewrites": self.budget.max_rewrites,
                "max_subtasks": self.budget.max_subtasks,
                "max_tool_calls": self.budget.max_tool_calls,
                "total_timeout_ms": self.budget.total_timeout_ms,
                "total_token_budget": self.budget.total_token_budget,
                "per_step_timeout_ms": self.budget.per_step_timeout_ms,
            },
            "consumed": self.budget.consumed.to_dict(),
            "remaining": self.get_remaining(),
        }

    def __repr__(self) -> str:
        c = self.budget.consumed
        return (
            f"BudgetController(path={self.path_id}, "
            f"retrieval={c.retrieval_rounds}/{self.budget.max_retrieval_rounds}, "
            f"rewrites={c.rewrites}/{self.budget.max_rewrites}, "
            f"tokens={c.tokens}/{self.budget.total_token_budget}, "
            f"elapsed_ms={c.elapsed_ms}/{self.budget.total_timeout_ms})"
        )


def _ratio(consumed: int, limit: int) -> float:
    """计算消耗比例，limit 为 0 时返回 0.0（由调用方预判）"""
    if limit <= 0:
        return 0.0
    return consumed / limit
