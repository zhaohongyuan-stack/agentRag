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
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .deduplicator import Deduplicator
from .parent_aggregator import ParentAggregator

logger = logging.getLogger(__name__)


# ============================================================
# 表格分区关键词库（方案三：证据层硬过滤）
# 分层优先级匹配：Tier 1 > Tier 2 > Tier 3
# 同层内按关键词长度降序排列（长词优先，避免短词截断长词）
# ============================================================

# Tier 1: 精确分区名（最高优先级，直接对应表格分区）
_TIER1_KEYWORDS: List[Tuple[str, str]] = [
    ("银行业金融机构", "1. 银行业金融机构"),
    ("商业银行合计", "其中：商业银行合计"),
    ("大型商业银行", "2. 大型商业银行"),
    ("股份制商业银行", "3. 股份制商业银行"),
    ("城市商业银行", "4. 城市商业银行"),
    ("农村金融机构", "5. 农村金融机构"),
    ("其他类金融机构", "6. 其他类金融机构"),
    ("外资银行", "6. 其他类金融机构"),
    ("民营银行", "6. 其他类金融机构"),
]

# Tier 2: 机构类型简称（中等优先级，需较独特才能匹配）
_TIER2_KEYWORDS: List[Tuple[str, str]] = [
    ("股份制", "3. 股份制商业银行"),
    ("城市商业", "4. 城市商业银行"),
    ("农商", "5. 农村金融机构"),
    ("农信", "5. 农村金融机构"),
    ("其他类", "6. 其他类金融机构"),
]

# Tier 3: 宽泛关键词（最低优先级，仅当 Tier 1/2 均未匹配时生效）
# 注意："银行业"会匹配文档标题中的"银行业"，所以放最低优先级
_TIER3_KEYWORDS: List[Tuple[str, str]] = [
    ("银行业", "1. 银行业金融机构"),
    ("商业银行", "其中：商业银行合计"),
    ("农村", "5. 农村金融机构"),
]


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
        retrieval_filters: Optional[Dict[str, Any]] = None,
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

        # 4.5 证据分区过滤（方案三：证据层硬过滤）
        evidence_items = self._filter_by_primary_partition(
            evidence_items, conflicts, query_text, retrieval_filters
        )

        # 5. 识别缺失条件
        missing_conditions = self._find_missing(claim_slots)

        # 6. 计算充分性评分
        sufficiency_score = self._calculate_sufficiency(
            claim_slots, evidence_items, conflicts, missing_conditions
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
                # 没有精确匹配，标记为 pending（后续 Evaluator Agent 判断）
                claim.evidence_ids = []
                claim.status = "pending"

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

        # 表格分区冲突检测：同一指标在同一文档的多个 table_name 分区出现
        conflicts.extend(self._detect_table_partition_conflicts(evidence))

        return conflicts

    def _detect_table_partition_conflicts(
        self, evidence: List[EvidenceItem]
    ) -> List[Dict[str, Any]]:
        """
        表格分区冲突检测（builder 内部版本）

        同一指标（metric_name）在同一文档的多个 table_name 分区中出现时，
        标记为分区冲突。返回 dict 格式，与 _detect_conflicts 保持一致。
        """
        conflicts: List[Dict[str, Any]] = []

        # metric_name + source_doc → table_name → [EvidenceItem, ...]
        metric_table_map: Dict[Tuple[str, str], Dict[str, List[EvidenceItem]]] = {}

        for ev in evidence:
            metadata = ev.metadata or {}
            metric_name = metadata.get("metric_name", "")
            table_name = metadata.get("table_name", "")
            if not metric_name or not table_name:
                continue

            key = (metric_name, ev.source_doc)
            if key not in metric_table_map:
                metric_table_map[key] = {}
            metric_table_map[key].setdefault(table_name, []).append(ev)

        for (metric_name, doc_name), table_groups in metric_table_map.items():
            if len(table_groups) < 2:
                continue

            partitions = list(table_groups.keys())
            evidence_ids = [
                ev.evidence_id for evs in table_groups.values() for ev in evs
            ]

            conflicts.append({
                "type": "table_partition_conflict",
                "description": (
                    f"指标 '{metric_name}' 在文档 '{doc_name}' 的 "
                    f"{len(partitions)} 个表格分区中均有数据："
                    f"{', '.join(partitions)}。需确认问题所指的具体分区。"
                ),
                "partitions": partitions,
                "metric_name": metric_name,
                "source_doc": doc_name,
                "evidence_ids": evidence_ids,
            })

        return conflicts

    def _find_missing(self, claims: List[ClaimSlot]) -> List[str]:
        """识别缺失条件（包括 missing 和 pending 状态的声明）"""
        missing = []
        for claim in claims:
            if claim.status in ("missing", "pending"):
                missing.append(claim.description)
        return missing

    def _calculate_sufficiency(
        self,
        claims: List[ClaimSlot],
        evidence: List[EvidenceItem],
        conflicts: List[Dict[str, Any]],
        missing_conditions: List[str],
    ) -> float:
        """
        计算证据充分性评分

        使用五维度加权评分器（SufficiencyScorer）替代简化公式：
          - 覆盖率 (30%)
          - 来源权威性 (15%)
          - 版本有效性 (20%)
          - 条件完整性 (15%)
          - 多通道一致性 (20%)
          - 冲突惩罚 + 缺失惩罚
        """
        # 无证据时直接返回0分
        if not evidence:
            return 0.0

        from agent_platform.evidence.sufficiency_scorer.scorer import SufficiencyScorer

        # 构建临时 EvidenceBundle 供 SufficiencyScorer 使用
        temp_bundle = EvidenceBundle(
            bundle_id="temp",
            claim_slots=claims,
            evidence_items=evidence,
            conflicts=conflicts,
            missing_conditions=missing_conditions,
            sufficiency_threshold=self._threshold,
        )

        scorer = SufficiencyScorer(threshold=self._threshold)
        result = scorer.score(temp_bundle)
        return result.score

    # ============================================================
    # 方案三：证据层分区过滤
    # ============================================================

    def _filter_by_primary_partition(
        self,
        evidence_items: List[EvidenceItem],
        conflicts: List[Dict[str, Any]],
        query_text: str,
        retrieval_filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """
        当检测到分区冲突时，只保留与查询最相关分区的证据。

        检索结果优先原则：如果检索请求已携带明确的 table_name 过滤条件，
        直接信任检索层决策，跳过分区推断和过滤。
        """
        # ── 检索结果优先 ──
        if retrieval_filters and retrieval_filters.get("table_name"):
            logger.info(
                "[证据层] 检索结果优先：table_name='%s' 已在检索层过滤，"
                "跳过证据层分区推断",
                retrieval_filters["table_name"],
            )
            return evidence_items

        # ── 无分区冲突时原样返回 ──
        partition_conflicts = [
            c for c in conflicts
            if c.get("type") == "table_partition_conflict"
        ]
        if not partition_conflicts:
            return evidence_items

        logger.info(
            "[证据层] 检测到 %d 个分区冲突，启动证据分区过滤",
            len(partition_conflicts),
        )

        # ── 推断主分区 ──
        target_partition = self._infer_target_partition(query_text)

        if target_partition:
            # 保留主分区证据 + 无分区标记的证据
            filtered = [
                ev for ev in evidence_items
                if not ev.metadata.get("table_name")
                or ev.metadata.get("table_name") == target_partition
            ]
            if filtered:
                logger.info(
                    "[证据层] 证据分区过滤：目标分区='%s'，"
                    "过滤前 %d 条 → 过滤后 %d 条",
                    target_partition, len(evidence_items), len(filtered),
                )
                return filtered
            # 安全兜底：推断出分区但过滤后为空，回退到过滤前
            logger.warning(
                "[证据层] 推断分区='%s' 但过滤后证据为空，回退到完整证据集",
                target_partition,
            )
            return evidence_items

        # ── 无法推断主分区：每分区保留 top-1 ──
        logger.info(
            "[证据层] 无法推断目标分区，执行按分区去重（每分区保留 top-1）"
        )
        return self._deduplicate_by_partition(evidence_items)

    def _infer_target_partition(self, query_text: str) -> Optional[str]:
        """
        从查询文本推断目标表格分区。

        采用分层优先级匹配：
          Tier 1: 精确分区名（如"大型商业银行"）— 命中即返回
          Tier 2: 机构类型简称（如"股份制"）— Tier 1 未命中时检查
          Tier 3: 宽泛关键词（如"银行业"）— 仅当 Tier 1/2 均未命中

        分层设计避免文档标题"银行业"覆盖问题中的"大型商业银行"。
        """
        if not query_text:
            return None

        # 按优先级依次检查三层关键词
        for tier_name, tier_keywords in (
            ("Tier1", _TIER1_KEYWORDS),
            ("Tier2", _TIER2_KEYWORDS),
            ("Tier3", _TIER3_KEYWORDS),
        ):
            # 同层内按关键词长度降序排列（长词优先，避免短词截断长词）
            sorted_keywords = sorted(
                tier_keywords, key=lambda x: len(x[0]), reverse=True
            )
            for keyword, partition in sorted_keywords:
                if keyword in query_text:
                    logger.debug(
                        "[证据层] 分区推断(%s)：query 中匹配到 '%s' → '%s'",
                        tier_name, keyword, partition,
                    )
                    return partition

        logger.debug("[证据层] 分区推断：query 中未匹配到任何分区关键词")
        return None

    def _deduplicate_by_partition(
        self, evidence_items: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        """无法推断目标分区时，每个 table_name 分区只保留得分最高的一条"""
        seen_partitions: Dict[str, EvidenceItem] = {}
        no_partition: List[EvidenceItem] = []

        for ev in evidence_items:  # 已按 score 降序排列
            table_name = ev.metadata.get("table_name", "")
            if not table_name:
                no_partition.append(ev)
                continue
            if table_name not in seen_partitions:
                seen_partitions[table_name] = ev

        result = no_partition + list(seen_partitions.values())
        logger.info(
            "[证据层] 按分区去重：%d 条 → %d 条（%d 个分区各保留 top-1，无分区 %d 条）",
            len(evidence_items), len(result),
            len(seen_partitions), len(no_partition),
        )
        return result
