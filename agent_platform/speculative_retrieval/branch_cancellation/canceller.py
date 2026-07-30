"""
动态分支取消器 — BranchCanceller

在推测式检索过程中，根据当前检索结果动态取消不再需要的分支，
避免无谓的计算开销。

取消规则（优先级从高到低）:
  1. 证据已充分 → 取消所有未完成分支
  2. 精确检索找到唯一有效条款 → 取消 Dense 和 relation
  3. BM25(lexical) 与 Dense 命中相同父条款 → 取消 table

设计要点:
  1. 状态感知: 仅取消未完成（LAUNCHED / RUNNING）的分支，已完成分支
     保留结果不被取消
  2. 原地更新: 直接修改分支状态并返回原列表，便于调用方继续使用
  3. 启发式判断: 父条款从 parent_id 或 chunk_id 前缀推导

模式参考: orchestration/budget_controller 的控制器风格
"""

import logging
from typing import List, Set

from ..branch_launcher.launcher import BranchStatus, RetrievalBranch

logger = logging.getLogger(__name__)

# 充分性阈值：达到该值认为证据已足够，可取消所有未完成分支
_SUFFICIENCY_THRESHOLD = 0.85
# 唯一条款判定得分阈值
_CLAUSE_SCORE_THRESHOLD = 0.8


class BranchCanceller:
    """
    动态分支取消器

    根据当前检索结果判断是否取消某些分支:
      1. 精确检索已找到唯一有效条款 → 取消 Dense 和 relation
      2. BM25 与 Dense 命中相同父条款 → 取消 table
      3. 证据已充分 → 取消所有未完成分支

    用法:
        canceller = BranchCanceller()
        canceller.evaluate_and_cancel(branches, sufficiency_score=0.9)
    """

    def evaluate_and_cancel(
        self,
        branches: List[RetrievalBranch],
        sufficiency_score: float = 0.0,
    ) -> List[RetrievalBranch]:
        """
        评估并取消不需要的分支

        按优先级依次应用取消规则，被取消的分支 status 变为 CANCELLED。
        仅取消未完成（LAUNCHED / RUNNING）的分支。

        Args:
            branches: 所有检索分支
            sufficiency_score: 当前充分性评分

        Returns:
            更新后的分支列表（与入参为同一列表，分支状态已更新）
        """
        # 1. 证据充分 → 取消所有未完成分支
        if sufficiency_score >= _SUFFICIENCY_THRESHOLD:
            self.cancel_all_pending(branches)
            logger.info(
                "证据充分（%.4f），已取消全部未完成分支", sufficiency_score
            )
            return branches

        # 2. 精确检索找到唯一有效条款 → 取消 dense 和 relation
        if self._found_unique_clause(branches):
            n = self._cancel_by_channels(branches, {"dense", "relation"})
            logger.info(
                "精确检索命中唯一条款，已取消 dense/relation 分支 count=%d",
                n,
            )
            return branches

        # 3. lexical 与 dense 命中相同父条款 → 取消 table
        if self._same_parent_hits(branches):
            n = self._cancel_by_channels(branches, {"table"})
            logger.info(
                "lexical 与 dense 命中相同父条款，已取消 table 分支 count=%d",
                n,
            )
            return branches

        logger.debug(
            "无需取消分支 sufficiency=%.4f", sufficiency_score
        )
        return branches

    def _found_unique_clause(self, branches: List[RetrievalBranch]) -> bool:
        """
        检查精确检索是否找到唯一有效条款

        判定条件: exact 通道分支存在且已完成，且仅有 1 条结果，且该结果
        得分不低于 _CLAUSE_SCORE_THRESHOLD。

        Args:
            branches: 所有检索分支

        Returns:
            是否找到唯一有效条款
        """
        for b in branches:
            if b.channel != "exact":
                continue
            if b.status != BranchStatus.COMPLETED:
                continue
            if len(b.results) != 1:
                return False
            score = float(b.results[0].get("score", 0.0))
            return score >= _CLAUSE_SCORE_THRESHOLD
        return False

    def _same_parent_hits(self, branches: List[RetrievalBranch]) -> bool:
        """
        检查 lexical 和 dense 是否命中相同父条款

        分别提取 lexical 与 dense 分支结果的父条款集合，判断是否存在交集。
        父条款优先取结果中的 parent_id，否则取 chunk_id 的首个 '-' 前段。

        Args:
            branches: 所有检索分支

        Returns:
            是否命中相同父条款
        """
        lexical_parents = self._parent_ids(branches, "lexical")
        if not lexical_parents:
            return False
        dense_parents = self._parent_ids(branches, "dense")
        if not dense_parents:
            return False
        return bool(lexical_parents & dense_parents)

    def cancel_all_pending(self, branches: List[RetrievalBranch]) -> List[RetrievalBranch]:
        """
        取消所有未完成的分支

        将状态为 LAUNCHED 或 RUNNING 的分支标记为 CANCELLED。
        已完成 / 已取消 / 已失败的分支保持不变。

        Args:
            branches: 所有检索分支

        Returns:
            更新后的分支列表（与入参为同一列表）
        """
        cancelled = 0
        for b in branches:
            if b.status in (BranchStatus.LAUNCHED, BranchStatus.RUNNING):
                b.status = BranchStatus.CANCELLED
                cancelled += 1
                logger.debug(
                    "取消未完成分支 branch_id=%s channel=%s",
                    b.branch_id,
                    b.channel,
                )
        logger.debug("cancel_all_pending 共取消 %d 个分支", cancelled)
        return branches

    # ============================================================
    # 内部方法
    # ============================================================

    def _cancel_by_channels(
        self, branches: List[RetrievalBranch], channels: Set[str]
    ) -> int:
        """
        取消指定通道集合中未完成的分支

        Args:
            branches: 所有检索分支
            channels: 需取消的通道名称集合

        Returns:
            被取消的分支数量
        """
        cancelled = 0
        for b in branches:
            if b.channel not in channels:
                continue
            if b.status in (BranchStatus.LAUNCHED, BranchStatus.RUNNING):
                b.status = BranchStatus.CANCELLED
                cancelled += 1
                logger.debug(
                    "取消分支 branch_id=%s channel=%s",
                    b.branch_id,
                    b.channel,
                )
        return cancelled

    @staticmethod
    def _parent_ids(
        branches: List[RetrievalBranch], channel: str
    ) -> Set[str]:
        """
        提取指定通道分支结果的父条款 ID 集合

        父条款优先取结果中的 parent_id 字段，否则从 chunk_id 取首个 '-'
        前段作为父条款推导值。

        Args:
            branches: 所有检索分支
            channel: 通道名称

        Returns:
            父条款 ID 集合
        """
        parents: Set[str] = set()
        for b in branches:
            if b.channel != channel:
                continue
            if b.status != BranchStatus.COMPLETED:
                continue
            for r in b.results:
                parent = r.get("parent_id")
                if not parent:
                    chunk_id = r.get("chunk_id", "")
                    parent = (
                        chunk_id.split("-", 1)[0]
                        if "-" in chunk_id
                        else chunk_id
                    )
                if parent:
                    parents.add(parent)
        return parents
