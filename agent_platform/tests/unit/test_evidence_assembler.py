"""
Evidence Assembler 单元测试

测试范围:
  - Deduplicator: content_hash 去重、chunk_id 去重、近似重复检测
  - ParentAggregator: 父文档聚合、无父级透传、最高分保留
  - EvidenceBuilder: 正常组装、去重集成、父聚合集成、排序、空结果
"""

import pytest

from agent_platform.evidence.evidence_assembler.builder import (
    ClaimSlot,
    EvidenceBuilder,
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.evidence.evidence_assembler.deduplicator import Deduplicator
from agent_platform.evidence.evidence_assembler.parent_aggregator import ParentAggregator


# ============================================================
# 测试数据工厂
# ============================================================

def make_hit(
    chunk_id: str = "",
    content: str = "",
    score: float = 0.5,
    doc_name: str = "《商业银行资本管理办法》",
    chunk_type: str = "clause",
    citation: str = "",
    hierarchy_path: str = "",
    metadata: dict = None,
) -> dict:
    """构造 RetrievalHit dict"""
    return {
        "chunk_id": chunk_id or f"chunk-{score}",
        "content": content,
        "evidence_snippet": content[:200],
        "score": score,
        "doc_name": doc_name,
        "doc_id": doc_name,
        "chunk_type": chunk_type,
        "citation": citation or f"{doc_name} 第1条",
        "hierarchy_path": hierarchy_path,
        "metadata": metadata or {},
    }


def make_hits(count: int = 5) -> list:
    """生成 count 个不同 chunk 的 hits"""
    return [
        make_hit(
            chunk_id=f"chunk-{i}",
            content=f"这是第{i}条法规的内容，关于资本充足率的要求。",
            score=0.9 - i * 0.1,
            citation=f"《商业银行资本管理办法》第{i + 1}条",
        )
        for i in range(count)
    ]


# ============================================================
# Deduplicator 测试
# ============================================================

class TestDeduplicator:
    """去重器测试"""

    def test_no_duplicates(self):
        """无重复时全部保留"""
        hits = make_hits(5)
        dedup = Deduplicator()
        result = dedup.deduplicate(hits)
        assert len(result) == 5

    def test_dedup_by_chunk_id(self):
        """相同 chunk_id 去重"""
        hits = [
            make_hit(chunk_id="chunk-1", content="内容A", score=0.9),
            make_hit(chunk_id="chunk-1", content="内容A", score=0.5),
        ]
        result = Deduplicator().deduplicate(hits)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "chunk-1"

    def test_dedup_by_content_hash(self):
        """相同内容去重，保留得分最高"""
        hits = [
            make_hit(chunk_id="chunk-a", content="相同内容", score=0.3),
            make_hit(chunk_id="chunk-b", content="相同内容", score=0.9),
        ]
        result = Deduplicator().deduplicate(hits)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_dedup_near_duplicate(self):
        """近似重复（子串包含）去重"""
        long_content = "商业银行资本管理办法第四十三条规定，核心一级资本充足率不得低于百分之五"
        short_content = "核心一级资本充足率不得低于百分之五"
        hits = [
            make_hit(chunk_id="chunk-short", content=short_content, score=0.8),
            make_hit(chunk_id="chunk-long", content=long_content, score=0.7),
        ]
        result = Deduplicator().deduplicate(hits)
        # 短内容是长内容的子串，应被移除
        assert len(result) == 1
        assert result[0]["chunk_id"] == "chunk-long"

    def test_empty_hits(self):
        """空列表处理"""
        assert Deduplicator().deduplicate([]) == []

    def test_preserves_order(self):
        """去重后保持相对顺序"""
        hits = [
            make_hit(chunk_id="chunk-1", content="内容一", score=0.5),
            make_hit(chunk_id="chunk-2", content="内容二", score=0.6),
            make_hit(chunk_id="chunk-1", content="内容一", score=0.9),  # 重复
            make_hit(chunk_id="chunk-3", content="内容三", score=0.7),
        ]
        result = Deduplicator().deduplicate(hits)
        assert len(result) == 3
        assert result[0]["chunk_id"] == "chunk-1"
        assert result[1]["chunk_id"] == "chunk-2"
        assert result[2]["chunk_id"] == "chunk-3"


# ============================================================
# ParentAggregator 测试
# ============================================================

class TestParentAggregator:
    """父文档聚合器测试"""

    def test_aggregate_same_parent(self):
        """同父级子 chunk 聚合"""
        hits = [
            make_hit(chunk_id="child-1", content="子内容1", score=0.7,
                     metadata={"parent_chunk_id": "parent-1"}),
            make_hit(chunk_id="child-2", content="子内容2", score=0.9,
                     metadata={"parent_chunk_id": "parent-1"}),
            make_hit(chunk_id="child-3", content="子内容3", score=0.5,
                     metadata={"parent_chunk_id": "parent-1"}),
        ]
        result = ParentAggregator().aggregate(hits)
        assert len(result) == 1
        # 保留得分最高的
        assert result[0]["score"] == 0.9
        # 合并了所有子 chunk_id
        merged_ids = result[0]["metadata"]["merged_child_chunk_ids"]
        assert len(merged_ids) == 3

    def test_no_parent_passthrough(self):
        """无父级信息的 hit 原样透传"""
        hits = [
            make_hit(chunk_id="chunk-1", content="内容1", score=0.5),
            make_hit(chunk_id="chunk-2", content="内容2", score=0.7),
        ]
        result = ParentAggregator().aggregate(hits)
        assert len(result) == 2

    def test_aggregate_by_hierarchy_path(self):
        """通过 hierarchy_path 前缀分组"""
        hits = [
            make_hit(chunk_id="c-1", content="内容1", score=0.6,
                     hierarchy_path="《办法》 > 第一章 > 第1条"),
            make_hit(chunk_id="c-2", content="内容2", score=0.8,
                     hierarchy_path="《办法》 > 第一章 > 第2条"),
            make_hit(chunk_id="c-3", content="内容3", score=0.5,
                     hierarchy_path="《办法》 > 第二章 > 第3条"),
        ]
        result = ParentAggregator().aggregate(hits)
        # c-1 和 c-2 同属"《办法》 > 第一章"，应聚合
        # c-3 属于"《办法》 > 第二章"，独立
        assert len(result) == 2

    def test_highest_score_kept(self):
        """聚合时保留得分最高的 hit"""
        hits = [
            make_hit(chunk_id="c-1", content="低分", score=0.3,
                     metadata={"parent_chunk_id": "p-1"}),
            make_hit(chunk_id="c-2", content="高分", score=0.95,
                     metadata={"parent_chunk_id": "p-1"}),
            make_hit(chunk_id="c-3", content="中分", score=0.6,
                     metadata={"parent_chunk_id": "p-1"}),
        ]
        result = ParentAggregator().aggregate(hits)
        assert len(result) == 1
        assert result[0]["score"] == 0.95

    def test_empty_hits(self):
        """空列表处理"""
        assert ParentAggregator().aggregate([]) == []


# ============================================================
# EvidenceBuilder 测试
# ============================================================

class TestEvidenceBuilder:
    """证据组装器测试"""

    def test_normal_assembly(self):
        """正常组装: 5 个不同 chunk → 5 个 evidence_items"""
        hits = make_hits(5)
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=[], query_text="测试")

        assert bundle.evidence_count == 5
        assert isinstance(bundle, EvidenceBundle)

    def test_dedup_integration(self):
        """去重集成: 2 个相同 content_hash → 只保留 1 个"""
        hits = [
            make_hit(chunk_id="c-1", content="相同内容", score=0.3),
            make_hit(chunk_id="c-2", content="相同内容", score=0.9),
            make_hit(chunk_id="c-3", content="不同内容", score=0.5),
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=[], query_text="测试")

        assert bundle.evidence_count == 2

    def test_parent_aggregation_integration(self):
        """父聚合集成: 3 个同父 chunk → 合并为 1 个"""
        hits = [
            make_hit(chunk_id="c-1", content="子内容1", score=0.7,
                     metadata={"parent_chunk_id": "p-1"}),
            make_hit(chunk_id="c-2", content="子内容2", score=0.9,
                     metadata={"parent_chunk_id": "p-1"}),
            make_hit(chunk_id="c-3", content="子内容3", score=0.5,
                     metadata={"parent_chunk_id": "p-1"}),
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=[], query_text="测试")

        assert bundle.evidence_count == 1

    def test_sorting_by_score(self):
        """排序: 证据按得分降序"""
        hits = [
            make_hit(chunk_id="c-1", content="低分内容AAA", score=0.3),
            make_hit(chunk_id="c-2", content="高分内容BBB", score=0.9),
            make_hit(chunk_id="c-3", content="中分内容CCC", score=0.7),
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=[], query_text="测试")

        scores = [e.score for e in bundle.evidence_items]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.9

    def test_empty_result(self):
        """空结果 → 空 EvidenceBundle"""
        builder = EvidenceBuilder()
        bundle = builder.build(hits=[], claims=[], query_text="测试")

        assert bundle.evidence_count == 0
        assert bundle.sufficiency_score == 0.0
        assert not bundle.is_sufficient

    def test_claim_binding(self):
        """声明槽位绑定"""
        hits = make_hits(3)
        claims = [
            {"claim_id": "c0", "description": "资本充足率要求", "slot_type": "metric"},
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=claims, query_text="资本充足率")

        assert len(bundle.claim_slots) == 1
        assert bundle.claim_slots[0].status in ("supported", "pending")

    def test_conflict_detection(self):
        """版本冲突检测"""
        hits = [
            make_hit(chunk_id="c-1", content="旧版内容", score=0.8,
                     metadata={"version_status": "superseded"}),
            make_hit(chunk_id="c-2", content="新版内容", score=0.9,
                     metadata={"version_status": "active"}),
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=[], query_text="测试")

        assert len(bundle.conflicts) > 0
        assert bundle.conflicts[0]["type"] == "version_conflict"

    def test_sufficiency_score(self):
        """充分性评分"""
        hits = make_hits(3)
        claims = [
            {"claim_id": "c0", "description": "资本充足率", "slot_type": "metric"},
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=claims, query_text="资本充足率")

        assert bundle.sufficiency_score > 0
        assert bundle.is_sufficient == (bundle.sufficiency_score >= bundle.sufficiency_threshold)

    def test_to_dict(self):
        """EvidenceBundle 序列化"""
        hits = make_hits(2)
        builder = EvidenceBuilder()
        bundle = builder.build(hits=hits, claims=[], query_text="测试")

        data = bundle.to_dict()
        assert "bundle_id" in data
        assert "evidence_items" in data
        assert "sufficiency" in data
        assert "conflicts" in data

    def test_backward_compatibility(self):
        """向后兼容: 不传 deduplicator/parent_aggregator 也能工作"""
        builder = EvidenceBuilder()
        hits = make_hits(3)
        bundle = builder.build(hits=hits, claims=[], query_text="测试")
        assert bundle.evidence_count == 3
