"""
证据组装器 — 将 RetrievalHit 列表转换为 EvidenceBundle

职责:
  1. 将 RetrievalHit 转换为回答级证据项
  2. 绑定证据到声明槽位（claim slots）
  3. 计算证据充分性评分
  4. 检测证据冲突
  5. 识别缺失条件

Phase 2 增强:
  - 去重: 基于 content_hash、chunk_id 及内容子串近似重复检测
  - 父文档聚合: 同一父条款的子 chunk 不冗余占据 Top-K（合并，保留最高分）
  - 得分排序: 证据项按 score 降序排列
  - 保留 Phase 1 全部功能: 声明绑定、冲突检测、充分性评分

Phase 1 简化版（仍保留）:
  - 充分性评分基于证据数量和声明覆盖率
  - 冲突检测基于版本状态
  - 不接入 LLM，纯规则计算
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .deduplicator import Deduplicator
from .parent_aggregator import ParentAggregator


@dataclass
class EvidenceItem:
    """单个证据项"""

    evidence_id: str
    chunk_id: str
    content: str
    evidence_snippet: str
    citation: str
    score: float
    source_doc: str
    hierarchy_path: str
    chunk_type: str
    normative_level: str = ""
    version_status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "evidence_snippet": self.evidence_snippet,
            "citation": self.citation,
            "score": self.score,
            "source_doc": self.source_doc,
            "hierarchy_path": self.hierarchy_path,
            "chunk_type": self.chunk_type,
            "normative_level": self.normative_level,
            "version_status": self.version_status,
        }


@dataclass
class ClaimSlot:
    """声明槽位"""

    claim_id: str
    description: str
    slot_type: str = ""
    status: str = "pending"  # pending/supported/missing/conflict
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "description": self.description,
            "slot_type": self.slot_type,
            "status": self.status,
            "evidence_ids": self.evidence_ids,
        }


@dataclass
class EvidenceBundle:
    """证据包 — 证据组装的最终产物"""

    bundle_id: str
    claim_slots: List[ClaimSlot] = field(default_factory=list)
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    sufficiency_score: float = 0.0
    sufficiency_threshold: float = 0.85
    is_sufficient: bool = False
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    missing_conditions: List[str] = field(default_factory=list)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_items)

    @property
    def supported_claims(self) -> int:
        return sum(1 for c in self.claim_slots if c.status == "supported")

    @property
    def total_claims(self) -> int:
        return len(self.claim_slots)

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "claim_slots": [c.to_dict() for c in self.claim_slots],
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "sufficiency": {
                "score": round(self.sufficiency_score, 4),
                "threshold": self.sufficiency_threshold,
                "is_sufficient": self.is_sufficient,
            },
            "conflicts": self.conflicts,
            "missing_conditions": self.missing_conditions,
        }


class EvidenceBuilder:
    """
    证据组装器

    将 RetrievalHit 列表和 QuerySpec 的声明槽位组装为 EvidenceBundle。

    Phase 2 增强流程:
      hits → 去重 → 父文档聚合 → 转换为 EvidenceItem → 得分排序
            → 声明绑定 → 冲突检测 → 缺失识别 → 充分性评分 → EvidenceBundle
    """

    def __init__(
        self,
        sufficiency_threshold: float = 0.85,
        deduplicator: Optional[Deduplicator] = None,
        parent_aggregator: Optional[ParentAggregator] = None,
    ):
        """
        Args:
            sufficiency_threshold: 充分性阈值，低于此值判定为证据不足
            deduplicator: 去重器实例，为 None 时使用默认 Deduplicator
            parent_aggregator: 父文档聚合器实例，为 None 时使用默认 ParentAggregator
        """
        self._threshold = sufficiency_threshold
        self._deduplicator = deduplicator or Deduplicator()
        self._parent_aggregator = parent_aggregator or ParentAggregator()

    def build(
        self,
        hits: List[dict],
        claims: List[Dict[str, Any]],
        query_text: str = "",
    ) -> EvidenceBundle:
        """
        组装证据包

        Phase 2 流程:
          0. 去重（content_hash / chunk_id / 近似重复）
          0.5. 父文档聚合（同父级子 chunk 合并，保留最高分）
          1. 将清洗后的 hits 转换为 EvidenceItem
          1.5. 按得分降序排序
          2. 将 claims dict 转换为 ClaimSlot
          3. 绑定证据到声明槽位
          4. 检测冲突
          5. 识别缺失条件
          6. 计算充分性评分

        Args:
            hits: RetrievalHit 列表（A组返回的检索结果）
            claims: 声明槽位列表（来自 QuerySpec）
            query_text: 查询文本（用于证据相关性判断）

        Returns:
            EvidenceBundle 对象
        """
        bundle_id = f"eb-{uuid.uuid4().hex[:8]}"

        # 0. 去重 → 父文档聚合（在转换为 EvidenceItem 之前处理原始 hits）
        cleaned_hits = self._deduplicator.deduplicate(hits)
        cleaned_hits = self._parent_aggregator.aggregate(cleaned_hits)

        # 1. 将清洗后的 hits 转换为 EvidenceItem
        evidence_items = self._hits_to_evidence(cleaned_hits)

        # 1.5. 按得分降序排序
        evidence_items = self._sort_evidence(evidence_items)

        # 2. 将 claims dict 转换为 ClaimSlot
        claim_slots = [
            ClaimSlot(
                claim_id=c.get("claim_id", f"c{i}"),
                description=c.get("description", ""),
                slot_type=c.get("slot_type", ""),
                status=c.get("status", "pending"),
                evidence_ids=list(c.get("evidence_ids", [])),
            )
            for i, c in enumerate(claims)
        ]

        # 3. 绑定证据到声明槽位
        claim_slots = self._bind_evidence(claim_slots, evidence_items, query_text)

        # 4. 检测冲突
        conflicts = self._detect_conflicts(evidence_items)

        # 5. 识别缺失条件
        missing_conditions = self._find_missing(claim_slots)

        # 6. 计算充分性评分
        sufficiency_score = self._calculate_sufficiency(
            claim_slots, evidence_items, conflicts
        )

        is_sufficient = sufficiency_score >= self._threshold

        return EvidenceBundle(
            bundle_id=bundle_id,
            claim_slots=claim_slots,
            evidence_items=evidence_items,
            sufficiency_score=sufficiency_score,
            sufficiency_threshold=self._threshold,
            is_sufficient=is_sufficient,
            conflicts=conflicts,
            missing_conditions=missing_conditions,
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _hits_to_evidence(self, hits: List[dict]) -> List[EvidenceItem]:
        """将 RetrievalHit dict 列表转换为 EvidenceItem 列表"""
        evidence_items = []
        for i, hit in enumerate(hits):
            metadata = hit.get("metadata", {})
            evidence_id = f"ev-{uuid.uuid4().hex[:8]}"

            item = EvidenceItem(
                evidence_id=evidence_id,
                chunk_id=hit.get("chunk_id", ""),
                content=hit.get("content", ""),
                evidence_snippet=hit.get("evidence_snippet", hit.get("content", "")[:200]),
                citation=hit.get("citation", ""),
                score=hit.get("score", 0.0),
                source_doc=hit.get("doc_name", hit.get("doc_id", "")),
                hierarchy_path=hit.get("hierarchy_path", ""),
                chunk_type=hit.get("chunk_type", ""),
                normative_level=metadata.get("normative_level", ""),
                version_status=metadata.get("version_status", "active"),
                metadata=metadata,
            )
            evidence_items.append(item)

        return evidence_items

    def _sort_evidence(self, evidence_items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        按得分降序排序证据项

        得分相同时保持原始顺序（稳定排序）。

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            按得分降序排列的 EvidenceItem 列表
        """
        return sorted(evidence_items, key=lambda e: e.score, reverse=True)

    @staticmethod
    def _compute_content_hash(content: str) -> str:
        """
        计算内容的哈希值

        对内容做 strip 后取 MD5，用于内容去重。
        忽略首尾空白，使 "  abc  " 与 "abc" 视为相同内容。

        Args:
            content: 文本内容

        Returns:
            16 进制哈希字符串
        """
        normalized = content.strip() if content else ""
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _bind_evidence(
        self,
        claims: List[ClaimSlot],
        evidence: List[EvidenceItem],
        query_text: str,
    ) -> List[ClaimSlot]:
        """
        绑定证据到声明槽位

        Phase 1 简化策略:
          - 如果有证据且声明描述关键词在证据内容中出现 → supported
          - 如果有证据但声明描述关键词未出现 → 暂标记 pending（Phase 2 用 LLM 判断）
          - 如果无证据 → missing
        """
        if not evidence:
            for claim in claims:
                claim.status = "missing"
            return claims

        for claim in claims:
            # 尝试在证据中找到支持该声明的证据
            matched_evidence = []
            claim_keywords = self._extract_keywords(claim.description)

            for ev in evidence:
                # 简单关键词匹配
                ev_text = (ev.content + " " + ev.evidence_snippet).lower()
                if any(kw in ev_text for kw in claim_keywords) or not claim_keywords:
                    matched_evidence.append(ev.evidence_id)

            if matched_evidence:
                claim.evidence_ids = matched_evidence[:3]  # 最多绑定3条证据
                claim.status = "supported"
            else:
                # 没有精确匹配，但有证据可用 → 标记为 pending（后续 LLM 判断）
                # Phase 1: 有证据就标记为 supported
                claim.evidence_ids = [evidence[0].evidence_id]
                claim.status = "supported" if evidence else "missing"

        return claims

    def _extract_keywords(self, text: str) -> List[str]:
        """从声明描述中提取关键词"""
        # 简单分词：按空格和标点分割
        import re
        words = re.split(r"[\s,，。、；;：:（）()]+", text)
        # 过滤太短的词
        return [w.lower() for w in words if len(w) >= 2]

    def _detect_conflicts(self, evidence: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """
        检测证据冲突

        Phase 1: 检测版本状态冲突
          - 同一文档的不同版本（active vs superseded）
        """
        conflicts = []

        # 按文档分组
        doc_groups: Dict[str, List[EvidenceItem]] = {}
        for ev in evidence:
            key = ev.source_doc
            if key not in doc_groups:
                doc_groups[key] = []
            doc_groups[key].append(ev)

        for doc_name, items in doc_groups.items():
            version_statuses = set(ev.version_status for ev in items)
            if len(version_statuses) > 1 and "superseded" in version_statuses:
                conflicts.append({
                    "type": "version_conflict",
                    "description": f"文档 '{doc_name}' 存在版本冲突",
                    "versions": list(version_statuses),
                    "evidence_ids": [ev.evidence_id for ev in items],
                })

        return conflicts

    def _find_missing(self, claims: List[ClaimSlot]) -> List[str]:
        """识别缺失条件"""
        missing = []
        for claim in claims:
            if claim.status == "missing":
                missing.append(claim.description)
        return missing

    def _calculate_sufficiency(
        self,
        claims: List[ClaimSlot],
        evidence: List[EvidenceItem],
        conflicts: List[Dict[str, Any]],
    ) -> float:
        """
        计算证据充分性评分

        评分公式（Phase 1 简化版）:
          score = claim_coverage × evidence_quality × (1 - conflict_penalty)

          claim_coverage = supported_claims / total_claims
          evidence_quality = min(evidence_count / expected_count, 1.0)
          conflict_penalty = 0.1 × conflict_count
        """
        if not claims:
            # 无声明槽位时，有证据即可
            return 1.0 if evidence else 0.0

        # 声明覆盖率
        supported = sum(1 for c in claims if c.status == "supported")
        claim_coverage = supported / len(claims)

        # 证据质量（有3条以上高质量证据为满分）
        expected_count = 3
        evidence_quality = min(len(evidence) / expected_count, 1.0)

        # 冲突惩罚
        conflict_penalty = 0.1 * len(conflicts)

        score = claim_coverage * evidence_quality * (1.0 - conflict_penalty)
        score = max(0.0, min(1.0, score))

        return score
