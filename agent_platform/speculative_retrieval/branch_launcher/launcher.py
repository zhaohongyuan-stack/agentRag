"""
推测式检索启动器 — SpeculativeLauncher

负责按分层策略并行启动多个检索分支（通道），为推测式检索提供
分支管理与执行入口。

分层启动策略:
  - T0: 低成本通道立即启动（exact, lexical, metadata）
  - T1: 延迟启动 Dense（100ms 后，由调用方控制时机）
  - T2: 条件启动 table/relation（证据不足时触发）

设计要点:
  1. 同步实现: 第一版不使用 asyncio，retrieval_func 为同步函数，
     便于在单线程 DAG 执行器中集成
  2. 分层标记: 每个分支携带 tier 信息，供上层调度器决定执行时机
  3. 非破坏性: 单个分支执行失败时记录 error 并标记 FAILED，不抛异常，
     不影响其他分支

模式参考: orchestration/budget_controller 的 dataclass + to_dict 风格
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BranchStatus(str, Enum):
    """
    检索分支状态枚举

    继承 str 以便于序列化与日志输出。

    状态流转:
        LAUNCHED → RUNNING → COMPLETED
                            → FAILED
        LAUNCHED / RUNNING → CANCELLED
    """

    LAUNCHED = "launched"  # 已创建，尚未执行
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成并返回结果
    CANCELLED = "cancelled"  # 被动态取消
    FAILED = "failed"  # 执行失败


@dataclass
class RetrievalBranch:
    """
    检索分支 — 描述一个通道的一次检索执行

    Attributes:
        branch_id: 分支唯一标识（格式 {channel}-{short_uuid}）
        channel: 检索通道（exact / lexical / dense / metadata / table / relation）
        tier: 启动层级（0=T0立即, 1=T1延迟, 2=T2条件）
        status: 分支当前状态
        results: 检索结果列表
        score: 分支得分（聚合其结果得分，取最高）
        started_at: 开始时间（ISO 字符串）
        completed_at: 完成时间（ISO 字符串）
        error: 失败时的错误信息
    """

    branch_id: str
    channel: str
    tier: int
    status: BranchStatus = BranchStatus.LAUNCHED
    results: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典，用于日志输出与序列化"""
        return {
            "branch_id": self.branch_id,
            "channel": self.channel,
            "tier": self.tier,
            "status": self.status.value,
            "results": list(self.results),
            "score": self.score,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class SpeculativeLauncher:
    """
    推测式检索启动器

    分层启动策略:
      - T0: 低成本通道立即启动（exact, cache, lexical, metadata）
      - T1: 延迟启动 Dense（100ms 后）
      - T2: 条件启动 table/relation（证据不足时）

    第一版采用同步实现，retrieval_func 为同步函数。分层启动时 T0 立即执行，
    T1/T2 仅创建分支并标记为 LAUNCHED，由调用方根据调度策略决定何时触发执行。

    用法:
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["exact", "dense"], retrieval_func)
        for b in branches:
            print(b.channel, b.status, len(b.results))
    """

    # T0 通道（立即启动）
    T0_CHANNELS = ["exact", "lexical", "metadata"]
    # T1 通道（延迟启动）
    T1_CHANNELS = ["dense"]
    # T2 通道（条件启动）
    T2_CHANNELS = ["table", "relation"]

    # ============================================================
    # 内部辅助
    # ============================================================

    def _tier_of(self, channel: str) -> int:
        """
        确定通道所属的启动层级

        未配置的通道默认归入 T1（延迟启动），并记录调试日志。
        """
        if channel in self.T0_CHANNELS:
            return 0
        if channel in self.T1_CHANNELS:
            return 1
        if channel in self.T2_CHANNELS:
            return 2
        logger.debug("通道 '%s' 未配置层级，默认归入 T1", channel)
        return 1

    @staticmethod
    def _make_branch_id(channel: str) -> str:
        """生成分支唯一标识，格式为 {channel}-{short_uuid}"""
        return f"{channel}-{uuid.uuid4().hex[:8]}"

    # ============================================================
    # 启动接口
    # ============================================================

    def launch(
        self,
        plan_channels: List[str],
        retrieval_func: Optional[Callable[[str], List[dict]]] = None,
    ) -> List[RetrievalBranch]:
        """
        同步启动检索分支（简化版，不使用 asyncio）

        对每个通道创建分支并立即执行 retrieval_func（若提供），
        执行成功后标记为 COMPLETED，失败标记为 FAILED。未提供检索函数时
        分支结果为空列表，状态保持 LAUNCHED。

        Args:
            plan_channels: 物理计划中配置的通道列表
            retrieval_func: (channel: str) -> List[dict]，检索函数。
                            为 None 时返回空结果列表。

        Returns:
            所有启动的分支列表
        """
        branches: List[RetrievalBranch] = []
        for channel in plan_channels:
            tier = self._tier_of(channel)
            branch = RetrievalBranch(
                branch_id=self._make_branch_id(channel),
                channel=channel,
                tier=tier,
                status=BranchStatus.LAUNCHED,
            )

            if retrieval_func is None:
                # 无检索函数，保留 LAUNCHED 状态，结果为空
                branches.append(branch)
                logger.debug(
                    "分支已创建（无检索函数）branch_id=%s channel=%s tier=%d",
                    branch.branch_id,
                    channel,
                    tier,
                )
                continue

            # 同步执行检索
            branch.status = BranchStatus.RUNNING
            branch.started_at = _now_iso()
            try:
                results = retrieval_func(channel) or []
                branch.results = list(results)
                branch.score = _aggregate_score(results)
                branch.status = BranchStatus.COMPLETED
                branch.completed_at = _now_iso()
                logger.debug(
                    "分支执行完成 branch_id=%s channel=%s results=%d score=%.4f",
                    branch.branch_id,
                    channel,
                    len(results),
                    branch.score,
                )
            except Exception as exc:  # noqa: BLE001
                branch.status = BranchStatus.FAILED
                branch.error = str(exc)
                branch.completed_at = _now_iso()
                logger.warning(
                    "分支执行失败 branch_id=%s channel=%s error=%s",
                    branch.branch_id,
                    channel,
                    exc,
                )
            branches.append(branch)

        logger.info(
            "同步启动完成 channels=%d completed=%d failed=%d",
            len(branches),
            sum(1 for b in branches if b.status == BranchStatus.COMPLETED),
            sum(1 for b in branches if b.status == BranchStatus.FAILED),
        )
        return branches

    def launch_tiered(
        self,
        plan_channels: List[str],
        retrieval_func: Optional[Callable[[str], List[dict]]] = None,
    ) -> Dict[str, List[RetrievalBranch]]:
        """
        分层启动，返回 {tier: branches} 字典

        T0 通道立即启动（若提供 retrieval_func 则同步执行），
        T1/T2 通道仅创建分支并标记为 LAUNCHED，不执行检索，
        由调用方根据调度策略决定何时触发执行。

        Args:
            plan_channels: 物理计划中配置的通道列表
            retrieval_func: (channel: str) -> List[dict]，检索函数。
                            为 None 时所有分支结果为空。

        Returns:
            按 tier 分组的分支字典，键为 "T0" / "T1" / "T2"
        """
        grouped: Dict[str, List[RetrievalBranch]] = {"T0": [], "T1": [], "T2": []}
        for channel in plan_channels:
            tier = self._tier_of(channel)
            branch = RetrievalBranch(
                branch_id=self._make_branch_id(channel),
                channel=channel,
                tier=tier,
                status=BranchStatus.LAUNCHED,
            )
            tier_key = f"T{tier}"
            if tier_key not in grouped:
                grouped[tier_key] = []

            if tier == 0 and retrieval_func is not None:
                # T0 立即执行
                branch.status = BranchStatus.RUNNING
                branch.started_at = _now_iso()
                try:
                    results = retrieval_func(channel) or []
                    branch.results = list(results)
                    branch.score = _aggregate_score(results)
                    branch.status = BranchStatus.COMPLETED
                    branch.completed_at = _now_iso()
                except Exception as exc:  # noqa: BLE001
                    branch.status = BranchStatus.FAILED
                    branch.error = str(exc)
                    branch.completed_at = _now_iso()
                    logger.warning(
                        "T0 分支执行失败 channel=%s error=%s", channel, exc
                    )
            # T1/T2 保持 LAUNCHED，等待调用方触发
            grouped[tier_key].append(branch)

        logger.info(
            "分层启动完成 T0=%d T1=%d T2=%d",
            len(grouped.get("T0", [])),
            len(grouped.get("T1", [])),
            len(grouped.get("T2", [])),
        )
        return grouped


# ============================================================
# 模块级辅助函数
# ============================================================


def _now_iso() -> str:
    """返回当前时间的 ISO 格式字符串（毫秒精度）"""
    return datetime.now().isoformat(timespec="milliseconds")


def _aggregate_score(results: List[dict]) -> float:
    """
    聚合分支结果得分

    取结果中 score 字段的最大值作为分支得分，无结果或无得分时返回 0.0。
    """
    if not results:
        return 0.0
    scores = [
        float(r.get("score", 0.0))
        for r in results
        if r.get("score") is not None
    ]
    if not scores:
        return 0.0
    return max(scores)
