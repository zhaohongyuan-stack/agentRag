"""
父文档聚合器 — 将同一父条款下的子 chunk 聚合为单条证据

职责:
  1. 按 parent_chunk_id（来自 metadata 或顶层字段）或 hierarchy_path 前缀分组
  2. 每组保留得分最高的 hit，同时合并元数据（收集所有子 chunk_id）
  3. 无父级信息的 hit 原样透传，不参与聚合

设计要点:
  - 防止同一父条款的多个子 chunk 冗余占据 Top-K 名额
  - 输入/输出均为 RetrievalHit dict 列表
  - 对保留的 hit 做深拷贝，不修改原始数据
  - 合并的子 chunk_id 列表写入 metadata["merged_child_chunk_ids"]
"""

import copy
from typing import Any, Dict, List, Optional, Tuple


class ParentAggregator:
    """
    父文档聚合器

    将同一父级下的子 chunk 聚合为一条，保留最高分 hit 并合并子 chunk 信息。

    分组键优先级:
      1. metadata["parent_chunk_id"]
      2. hit 顶层 "parent_chunk_id"
      3. hierarchy_path 前缀（去掉最后一级的父路径）
      4. 以上均无 → 不分组，原样透传

    Attributes:
        path_separator: hierarchy_path 中的层级分隔符，默认 " > "
    """

    def __init__(self, path_separator: str = " > "):
        """
        Args:
            path_separator: hierarchy_path 的层级分隔符，
                            用于切分路径并提取父级前缀
        """
        self._sep = path_separator

    # ============================================================
    # 公共方法
    # ============================================================

    def aggregate(self, hits: List[dict]) -> List[dict]:
        """
        对检索结果执行父文档聚合

        流程:
          1. 为每条 hit 计算分组键（parent_chunk_id 或 hierarchy_path 前缀）
          2. 按分组键分组
          3. 每组保留得分最高的 hit，合并所有子 chunk_id 到 metadata
          4. 无分组键的 hit 原样透传
          5. 按最高分降序返回

        Args:
            hits: RetrievalHit dict 列表

        Returns:
            聚合后的 RetrievalHit dict 列表（按得分降序）
        """
        if not hits:
            return []

        # 1. 计算分组键并分组
        groups: Dict[str, List[Tuple[int, dict]]] = {}
        ungrouped: List[Tuple[int, dict]] = []

        for idx, hit in enumerate(hits):
            group_key = self._get_group_key(hit)
            if group_key is None:
                ungrouped.append((idx, hit))
            else:
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append((idx, hit))

        result: List[Tuple[float, dict]] = []

        # 2. 无分组键的 hit 原样透传
        for idx, hit in ungrouped:
            score = hit.get("score", 0.0)
            result.append((score, copy.deepcopy(hit)))

        # 3. 每组保留最高分 hit，合并子 chunk_id
        for group_key, members in groups.items():
            best_hit = self._merge_group(members)
            score = best_hit.get("score", 0.0)
            result.append((score, best_hit))

        # 4. 按得分降序排列
        result.sort(key=lambda x: x[0], reverse=True)

        return [hit for _, hit in result]

    # ============================================================
    # 内部方法
    # ============================================================

    def _get_group_key(self, hit: dict) -> Optional[str]:
        """
        为 hit 计算父文档分组键

        优先级:
          1. metadata["parent_chunk_id"]
          2. hit 顶层 "parent_chunk_id"
          3. hierarchy_path 前缀（父路径，需至少 2 级）

        Returns:
            分组键字符串，无可用父级信息时返回 None
        """
        # 1. 从 metadata 中获取 parent_chunk_id
        metadata = hit.get("metadata", {})
        if isinstance(metadata, dict):
            parent_id = metadata.get("parent_chunk_id", "")
            if parent_id:
                return f"pid:{parent_id}"

        # 2. 从 hit 顶层获取 parent_chunk_id
        parent_id = hit.get("parent_chunk_id", "")
        if parent_id:
            return f"pid:{parent_id}"

        # 3. 从 hierarchy_path 提取父级前缀
        hierarchy_path = hit.get("hierarchy_path", "")
        if hierarchy_path and self._sep in hierarchy_path:
            segments = hierarchy_path.split(self._sep)
            # 至少需要 2 级才能提取父路径
            if len(segments) >= 2:
                parent_path = self._sep.join(segments[:-1])
                if parent_path.strip():
                    return f"path:{parent_path}"

        # 4. 无可用父级信息
        return None

    def _merge_group(self, members: List[Tuple[int, dict]]) -> dict:
        """
        合并同一父级下的多个 hit

        - 保留得分最高的 hit（得分相同保留先出现的）
        - 收集所有子 chunk_id 到 metadata["merged_child_chunk_ids"]
        - 收集所有子 chunk 的 score 列表到 metadata["merged_child_scores"]
        - 记录被合并的子 chunk 数量到 metadata["merged_child_count"]

        Args:
            members: [(原始索引, hit), ...] 列表

        Returns:
            合并后的 hit dict（深拷贝，不修改原始数据）
        """
        # 找到得分最高的 hit（得分相同保留先出现的）
        best_idx_pos = 0
        best_score = members[0][1].get("score", 0.0)
        for i in range(1, len(members)):
            score = members[i][1].get("score", 0.0)
            if score > best_score:
                best_score = score
                best_idx_pos = i

        # 深拷贝最佳 hit，避免修改原始数据
        best_hit = copy.deepcopy(members[best_idx_pos][1])

        # 收集所有子 chunk_id（去重，保持顺序）
        child_chunk_ids: List[str] = []
        child_scores: List[float] = []
        seen_ids: set = set()

        for _, hit in members:
            chunk_id = hit.get("chunk_id", "")
            score = hit.get("score", 0.0)
            if chunk_id and chunk_id not in seen_ids:
                child_chunk_ids.append(chunk_id)
                child_scores.append(round(score, 6))
                seen_ids.add(chunk_id)

        # 写入合并信息到 metadata
        metadata = best_hit.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["merged_child_chunk_ids"] = child_chunk_ids
        metadata["merged_child_scores"] = child_scores
        metadata["merged_child_count"] = len(child_chunk_ids)
        best_hit["metadata"] = metadata

        return best_hit

    def __repr__(self) -> str:
        return f"ParentAggregator(path_separator={self._sep!r})"
