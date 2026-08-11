"""
检索结果流收集器 — ResultStream

收集多个检索分支的检索结果，按得分排序，支持增量更新与去重。

设计要点:
  1. 增量收集: add_results 可多次调用，每次添加一个分支的结果
  2. 通道过滤: 通过 branch_id 前缀解析通道，支持按通道查询结果
  3. 去重合并: merge_and_dedupe 基于 chunk_id 去重，保留得分最高者
  4. 非破坏性: 不修改传入的结果字典（去重时取副本）

模式参考: orchestration/budget_controller 的 dataclass 风格
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ResultStream:
    """
    检索结果流收集器

    收集多个分支的检索结果，按得分排序，支持增量更新。

    Attributes:
        _branch_results: 分支 ID 到其结果的映射
        _branch_channels: 分支 ID 到通道的映射（从 branch_id 前缀解析）
        _results: 所有结果的扁平缓存（按 score 降序）
        _dirty: 缓存是否需要重建
    """

    def __init__(self):
        """初始化空的结果收集器"""
        self._branch_results: Dict[str, List[dict]] = {}
        self._branch_channels: Dict[str, str] = {}
        self._results: List[dict] = []
        self._dirty: bool = False

    def add_results(self, branch_id: str, results: List[dict]):
        """
        添加一个分支的结果

        会解析 branch_id 前缀作为通道名（格式 {channel}-{uuid}），
        用于后续按通道过滤。

        Args:
            branch_id: 分支唯一标识
            results: 该分支的检索结果列表
        """
        self._branch_results[branch_id] = list(results)
        self._branch_channels[branch_id] = self._channel_from_branch_id(branch_id)
        self._dirty = True
        logger.debug(
            "添加分支结果 branch_id=%s channel=%s count=%d",
            branch_id,
            self._branch_channels[branch_id],
            len(results),
        )

    def get_all_results(self) -> List[dict]:
        """
        获取所有分支的结果，按 score 降序

        Returns:
            所有结果的扁平列表，按 score 从高到低排序
        """
        self._rebuild_if_dirty()
        return list(self._results)

    def get_results_by_channel(self, channel: str) -> List[dict]:
        """
        获取指定通道的结果

        通过 branch_id 前缀匹配通道，若结果内含 channel 字段则一并匹配。

        Args:
            channel: 检索通道名称

        Returns:
            该通道的所有结果，按 score 降序
        """
        collected: List[dict] = []
        for branch_id, results in self._branch_results.items():
            branch_channel = self._branch_channels.get(branch_id, "")
            if branch_channel == channel:
                collected.extend(results)
                continue
            # 兜底：检查结果内 channel 字段
            for r in results:
                if r.get("channel") == channel:
                    collected.append(r)
        collected.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return collected

    def merge_and_dedupe(self) -> List[dict]:
        """
        合并所有结果并去重（基于 chunk_id）

        相同 chunk_id 的结果保留得分最高者；无 chunk_id 的结果视为唯一，
        全部保留。

        Returns:
            去重后的结果列表，按 score 降序
        """
        merged: Dict[str, dict] = {}
        no_id: List[dict] = []
        for results in self._branch_results.values():
            for r in results:
                chunk_id = r.get("chunk_id")
                if chunk_id is None:
                    no_id.append(dict(r))
                    continue
                existing = merged.get(chunk_id)
                if existing is None:
                    merged[chunk_id] = dict(r)
                else:
                    if float(r.get("score", 0.0)) > float(
                        existing.get("score", 0.0)
                    ):
                        merged[chunk_id] = dict(r)
        combined = list(merged.values()) + no_id
        combined.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        logger.debug(
            "合并去重完成 total=%d unique=%d",
            self.get_result_count(),
            len(combined),
        )
        return combined

    def get_result_count(self) -> int:
        """
        获取总结果数（未去重）

        Returns:
            所有分支结果的累计数量
        """
        return sum(len(rs) for rs in self._branch_results.values())

    def clear(self):
        """清空所有结果"""
        self._branch_results.clear()
        self._branch_channels.clear()
        self._results.clear()
        self._dirty = False
        logger.debug("结果流已清空")

    # ============================================================
    # 内部方法
    # ============================================================

    def _rebuild_if_dirty(self):
        """脏标记触发时重建扁平结果缓存"""
        if not self._dirty:
            return
        flat: List[dict] = []
        for results in self._branch_results.values():
            flat.extend(results)
        flat.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        self._results = flat
        self._dirty = False

    @staticmethod
    def _channel_from_branch_id(branch_id: str) -> str:
        """
        从 branch_id 解析通道名

        branch_id 格式约定为 {channel}-{uuid}，取首个 '-' 之前的部分。
        无 '-' 时返回整个 branch_id。
        """
        if not branch_id or "-" not in branch_id:
            return branch_id or ""
        return branch_id.split("-", 1)[0]
