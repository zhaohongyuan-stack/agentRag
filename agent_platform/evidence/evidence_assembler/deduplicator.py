"""
证据去重器 — 对检索结果（RetrievalHit）进行去重

职责:
  1. 按 content_hash 去重（内容相同的 hit 只保留得分最高的）
  2. 按 chunk_id 去重（同一 chunk_id 只保留首次出现）
  3. 近似重复检测（内容互为子串时，保留更长更完整的那条）

设计要点:
  - 输入/输出均为 RetrievalHit dict 列表，不依赖 EvidenceItem
  - 去重顺序: chunk_id → content_hash → 近似重复
  - 不修改原始 hit dict（浅拷贝列表，hit dict 本身按需深拷贝）
"""

import hashlib
from typing import Any, Dict, List, Optional


class Deduplicator:
    """
    检索结果去重器

    对 RetrievalHit 列表执行三级去重:
      1. chunk_id 精确去重（保留首次出现）
      2. content_hash 精确去重（保留得分最高）
      3. 近似重复检测（子串包含关系，保留更长者）

    Attributes:
        min_content_len: 参与近似重复检测的最小内容长度，短于此长度的内容跳过子串检测
    """

    def __init__(self, min_content_len: int = 10):
        """
        Args:
            min_content_len: 近似重复检测的最小内容长度阈值，
                             内容短于此值时不参与子串去重（避免误删短文本）
        """
        self._min_content_len = min_content_len

    # ============================================================
    # 公共方法
    # ============================================================

    def deduplicate(self, hits: List[dict]) -> List[dict]:
        """
        对检索结果列表执行去重

        执行顺序:
          1. 按 chunk_id 去重（同一 chunk_id 保留首次出现）
          2. 按 content_hash 去重（同一内容哈希保留得分最高）
          3. 近似重复检测（内容为另一条子串时移除较短者）

        Args:
            hits: RetrievalHit dict 列表

        Returns:
            去重后的 RetrievalHit dict 列表（保持原始相对顺序）
        """
        if not hits:
            return []

        # 第1级: 按 chunk_id 去重
        deduped = self._dedup_by_chunk_id(hits)

        # 第2级: 按 content_hash 去重
        deduped = self._dedup_by_content_hash(deduped)

        # 第3级: 近似重复检测（子串包含）
        deduped = self._dedup_near_duplicates(deduped)

        return deduped

    # ============================================================
    # 内部方法 — 各级去重
    # ============================================================

    def _dedup_by_chunk_id(self, hits: List[dict]) -> List[dict]:
        """
        按 chunk_id 精确去重

        同一 chunk_id 的 hit 只保留首次出现的那条。

        Args:
            hits: RetrievalHit dict 列表

        Returns:
            去重后的列表
        """
        seen_ids: set = set()
        result: List[dict] = []

        for hit in hits:
            chunk_id = hit.get("chunk_id", "")
            if chunk_id:
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
            result.append(hit)

        return result

    def _dedup_by_content_hash(self, hits: List[dict]) -> List[dict]:
        """
        按 content_hash 精确去重

        对每条 hit 的 content 计算哈希，内容相同的只保留得分最高的。
        得分相同时保留先出现的那条（稳定去重）。

        Args:
            hits: RetrievalHit dict 列表

        Returns:
            去重后的列表（保持原始相对顺序）
        """
        # content_hash -> (score, index, hit)
        best: Dict[str, tuple] = {}

        for idx, hit in enumerate(hits):
            content = hit.get("content", "")
            content_hash = self._compute_hash(content)
            score = hit.get("score", 0.0)

            if content_hash not in best:
                best[content_hash] = (score, idx, hit)
            else:
                existing_score = best[content_hash][0]
                # 得分更高时替换；得分相同保留先出现的（稳定）
                if score > existing_score:
                    best[content_hash] = (score, idx, hit)

        # 按原始索引排序，保持相对顺序
        sorted_items = sorted(best.values(), key=lambda x: x[1])
        return [item[2] for item in sorted_items]

    def _dedup_near_duplicates(self, hits: List[dict]) -> List[dict]:
        """
        近似重复检测 — 基于子串包含关系

        如果 hit A 的内容是 hit B 内容的子串，且 B 更长，
        则认为 A 是 B 的冗余片段，移除 A。

        边界处理:
          - 内容为空或过短（< min_content_len）的 hit 不参与检测
          - 内容完全相同的情况已由 content_hash 去重处理
          - 仅当 len(A) < len(B) 且 A in B 时才移除 A

        Args:
            hits: RetrievalHit dict 列表

        Returns:
            去重后的列表
        """
        if len(hits) <= 1:
            return list(hits)

        n = len(hits)
        removed: set = set()

        for i in range(n):
            if i in removed:
                continue
            content_i = hits[i].get("content", "")
            if len(content_i) < self._min_content_len:
                continue

            for j in range(n):
                if i == j or j in removed:
                    continue
                content_j = hits[j].get("content", "")
                if len(content_j) < self._min_content_len:
                    continue

                # 如果 i 的内容是 j 的子串，且 j 更长 → 移除 i
                if len(content_i) < len(content_j) and content_i in content_j:
                    removed.add(i)
                    break

        return [hit for idx, hit in enumerate(hits) if idx not in removed]

    # ============================================================
    # 内部工具
    # ============================================================

    @staticmethod
    def _compute_hash(content: str) -> str:
        """
        计算内容的哈希值

        对内容做 strip 后取 MD5，用于快速判断内容是否相同。
        忽略首尾空白，使 "  abc  " 与 "abc" 视为相同内容。

        Args:
            content: 文本内容

        Returns:
            16 进制哈希字符串
        """
        normalized = content.strip() if content else ""
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return f"Deduplicator(min_content_len={self._min_content_len})"
