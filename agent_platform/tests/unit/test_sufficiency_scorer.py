"""
SufficiencyScorer 单元测试 — 证据充分性评分模块

测试范围:
  - SufficiencyScorer.score(): 五类典型场景的综合评分（参考开发计划测试用例表）
      * 全部覆盖 6/6 supported
      * 缺 1 非必填 5/6 supported（缺失 optional）
      * 缺 1 必填 5/6 supported（缺失 required）
      * 有冲突 6/6 supported + 1 个冲突
      * 空证据 0/6 supported
  - calculate_coverage / calculate_required_coverage: 覆盖率计算
  - calculate_authority: 不同 normative_level 的加权
  - calculate_version_validity: active / superseded 加权
  - calculate_condition_completeness: 缺失条件数量
  - calculate_channel_consistency: 多通道一致性
  - calculate_conflict_penalty: 不同冲突类型的惩罚值
  - calculate_missing_penalty: 多个缺失条件的惩罚（含上限）
  - 阈值边界测试: score 正好等于阈值
  - SufficiencyScore 数据结构与序列化

评分公式（来自 scorer.py）:
    score = coverage * 0.30
          + authority * 0.15
          + version_validity * 0.20
          + condition_completeness * 0.15
          + multi_channel_consistency * 0.20
          - conflict_penalty
          - missing_penalty
    最终限制在 [0.0, 1.0]，is_sufficient = (score >= threshold)，默认阈值 0.85
"""

import pytest

from agent_platform.evidence.conflict_detector.conflict_types import (
    Conflict,
    ConflictType,
)
from agent_platform.evidence.evidence_assembler.builder import (
    ClaimSlot,
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.evidence.sufficiency_scorer import (
    AUTHORITY_WEIGHTS,
    CONFLICT_PENALTIES,
    DEFAULT_CONFLICT_PENALTY,
    VERSION_WEIGHTS,
    SufficiencyScore,
    SufficiencyScorer,
    calculate_authority,
    calculate_channel_consistency,
    calculate_condition_completeness,
    calculate_conflict_penalty,
    calculate_coverage,
    calculate_missing_penalty,
    calculate_required_coverage,
    calculate_version_validity,
)


# ============================================================
# 测试数据工厂
# ============================================================

# 6 个声明槽位规格: (slot_type, description)
# 前 3 个为 required，后 3 个为 optional
CLAIM_SPECS = [
    ("required", "资本充足率最低要求"),
    ("required", "核心一级资本充足率"),
    ("required", "一级资本充足率"),
    ("optional", "杠杆率要求"),
    ("optional", "流动性覆盖率"),
    ("optional", "净稳定资金比例"),
]


def make_evidence_item(
    evidence_id: str = "ev-1",
    chunk_id: str = "chunk-1",
    content: str = "商业银行资本充足率不得低于百分之八。",
    normative_level: str = "法律",
    version_status: str = "active",
    channel: str = "regulation",
    score: float = 0.9,
    source_doc: str = "《商业银行资本管理办法》",
) -> EvidenceItem:
    """构造单个证据项，可指定权威性层级、版本状态与来源通道"""
    metadata = {"channel": channel} if channel else {}
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        content=content,
        evidence_snippet=content[:200],
        citation=f"{source_doc} 第1条",
        score=score,
        source_doc=source_doc,
        hierarchy_path="",
        chunk_type="clause",
        normative_level=normative_level,
        version_status=version_status,
        metadata=metadata,
    )


def make_claim_slots(missing_indices=()):
    """
    构造 6 个声明槽位（3 required + 3 optional）

    Args:
        missing_indices: 标记为 "missing" 的槽位下标集合，其余为 "supported"
    """
    slots = []
    for i, (slot_type, desc) in enumerate(CLAIM_SPECS):
        status = "missing" if i in missing_indices else "supported"
        slots.append(
            ClaimSlot(
                claim_id=f"c{i}",
                description=desc,
                slot_type=slot_type,
                status=status,
                evidence_ids=[],
            )
        )
    return slots


def make_bundle(
    claim_slots,
    evidence_items,
    conflicts=None,
    missing_conditions=None,
    bundle_id: str = "eb-test",
) -> EvidenceBundle:
    """构造证据包"""
    return EvidenceBundle(
        bundle_id=bundle_id,
        claim_slots=claim_slots,
        evidence_items=evidence_items,
        conflicts=conflicts or [],
        missing_conditions=missing_conditions or [],
    )


def make_high_quality_evidence(channel_a="regulation", channel_b="gazette"):
    """
    高质量证据对: 法律(1.0) + 行政法规(0.9)，均 active
    - 两通道不同 → multi_channel_consistency = 1.0
    - 同通道 → multi_channel_consistency = 0.7
    - authority = (1.0 + 0.9) / 2 = 0.95
    - version_validity = 1.0
    """
    return [
        make_evidence_item("ev-1", "chunk-1", normative_level="法律", channel=channel_a),
        make_evidence_item("ev-2", "chunk-2", normative_level="行政法规", channel=channel_b),
    ]


def make_low_authority_evidence():
    """
    低权威性证据对: 其他(0.4) + 规范性文件(0.6)，均 active，同通道
    - authority = (0.4 + 0.6) / 2 = 0.5
    - version_validity = 1.0
    - multi_channel_consistency = 0.7（单通道）
    """
    return [
        make_evidence_item("ev-1", "chunk-1", normative_level="其他", channel="regulation"),
        make_evidence_item("ev-2", "chunk-2", normative_level="规范性文件", channel="regulation"),
    ]


# ============================================================
# SufficiencyScorer 综合评分场景测试（对应开发计划测试用例表）
# ============================================================

class TestSufficiencyScorerScenarios:
    """五类典型场景的综合充分性评分"""

    def test_all_covered(self):
        """
        全部覆盖: 6/6 supported，无冲突
        - coverage=1.0, authority=0.95, version=1.0, completeness=1.0, consistency=1.0
        - score = 0.30 + 0.1425 + 0.20 + 0.15 + 0.20 = 0.9925
        - 预期: 0.90-1.0, is_sufficient=True
        """
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices=()),
            evidence_items=make_high_quality_evidence(),
        )
        result = SufficiencyScorer().score(bundle)

        assert 0.90 <= result.score <= 1.0
        assert result.is_sufficient is True
        # 必填覆盖率也应为 1.0
        assert result.components["required_coverage"] == pytest.approx(1.0)
        assert result.components["coverage"] == pytest.approx(1.0)
        # 无惩罚
        assert result.penalties["conflict_penalty"] == 0.0
        assert result.penalties["missing_penalty"] == 0.0

    def test_missing_one_optional(self):
        """
        缺 1 非必填: 5/6 supported（缺失 optional c5），无冲突，单通道证据
        - coverage=5/6, authority=0.95, version=1.0, completeness=1.0, consistency=0.7
        - score = 0.25 + 0.1425 + 0.20 + 0.15 + 0.14 = 0.8825
        - 预期: 0.80-0.90；is_sufficient 取决于阈值（默认 0.85 → True）
        """
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={5}),  # c5 为 optional
            evidence_items=make_high_quality_evidence("regulation", "regulation"),
        )
        result = SufficiencyScorer().score(bundle)

        assert 0.80 <= result.score <= 0.90
        # 缺失的是 optional，required_coverage 仍为 1.0
        assert result.components["required_coverage"] == pytest.approx(1.0)
        assert result.components["coverage"] == pytest.approx(5 / 6)
        # 默认阈值 0.85 → 0.8825 充分
        assert result.is_sufficient is True

    def test_missing_one_optional_threshold_dependent(self):
        """
        缺 1 非必填的 is_sufficient 取决于阈值:
        - 同一 bundle，阈值 0.85 → True；阈值 0.90 → False
        """
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={5}),
            evidence_items=make_high_quality_evidence("regulation", "regulation"),
        )
        # 阈值 0.85 → 充分
        assert SufficiencyScorer(threshold=0.85).score(bundle).is_sufficient is True
        # 阈值 0.90 → 不足
        assert SufficiencyScorer(threshold=0.90).score(bundle).is_sufficient is False

    def test_missing_one_required(self):
        """
        缺 1 必填: 5/6 supported（缺失 required c0），无冲突，低权威性单通道证据，
        且将该必填声明列入 missing_conditions
        - coverage=5/6, authority=0.5, version=1.0, completeness=0.85, consistency=0.7
        - missing_penalty=0.05
        - score = 0.25 + 0.075 + 0.20 + 0.1275 + 0.14 - 0.05 = 0.7425
        - 预期: 0.60-0.75, is_sufficient=False
        """
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={0}),  # c0 为 required
            evidence_items=make_low_authority_evidence(),
            missing_conditions=[CLAIM_SPECS[0][1]],
        )
        result = SufficiencyScorer().score(bundle)

        assert 0.60 <= result.score <= 0.75
        assert result.is_sufficient is False
        # 必填覆盖率: 2/3 required supported
        assert result.components["required_coverage"] == pytest.approx(2 / 3)
        assert result.components["coverage"] == pytest.approx(5 / 6)
        assert result.penalties["missing_penalty"] == pytest.approx(0.05)

    def test_with_conflict(self):
        """
        有冲突: 6/6 supported，1 个效力冲突（AUTHORITY_CONFLICT，惩罚 0.15）
        - coverage=1.0, authority=0.95, version=1.0, completeness=1.0, consistency=1.0
        - conflict_penalty=0.15
        - score = 0.9925 - 0.15 = 0.8425
        - 预期: 0.70-0.85, is_sufficient=False（0.8425 < 0.85）
        """
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices=()),
            evidence_items=make_high_quality_evidence(),
            conflicts=[{"type": "authority_conflict", "description": "效力冲突"}],
        )
        result = SufficiencyScorer().score(bundle)

        assert 0.70 <= result.score <= 0.85
        assert result.is_sufficient is False
        assert result.penalties["conflict_penalty"] == pytest.approx(0.15)
        assert result.components["coverage"] == pytest.approx(1.0)

    def test_empty_evidence(self):
        """
        空证据: 0/6 supported，无证据项，6 个声明全部缺失
        - coverage=0.0, authority=0.0, version=0.0, consistency=0.0
        - completeness=0.10（6 个缺失）, missing_penalty=0.30（上限）
        - score = 0.10*0.15 - 0.30 = -0.285 → 限制为 0.0
        - 预期: 0.0-0.1, is_sufficient=False
        """
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={0, 1, 2, 3, 4, 5}),
            evidence_items=[],
            missing_conditions=[desc for _, desc in CLAIM_SPECS],
        )
        result = SufficiencyScorer().score(bundle)

        assert 0.0 <= result.score <= 0.1
        assert result.is_sufficient is False
        assert result.components["coverage"] == 0.0
        assert result.components["authority"] == 0.0
        assert result.components["version_validity"] == 0.0
        assert result.components["multi_channel_consistency"] == 0.0
        assert result.penalties["missing_penalty"] == pytest.approx(0.30)


# ============================================================
# calculate_coverage / calculate_required_coverage 测试
# ============================================================

class TestCalculateCoverage:
    """声明槽位覆盖率计算"""

    def test_all_supported(self):
        """6/6 supported → 1.0"""
        slots = make_claim_slots(missing_indices=())
        assert calculate_coverage(slots) == pytest.approx(1.0)

    def test_half_supported(self):
        """3/6 supported → 0.5"""
        slots = make_claim_slots(missing_indices={3, 4, 5})
        assert calculate_coverage(slots) == pytest.approx(0.5)

    def test_none_supported(self):
        """0/6 supported → 0.0"""
        slots = make_claim_slots(missing_indices={0, 1, 2, 3, 4, 5})
        assert calculate_coverage(slots) == 0.0

    def test_empty_slots(self):
        """无声明槽位 → 1.0（按实现约定）"""
        assert calculate_coverage([]) == 1.0

    def test_pending_not_counted(self):
        """pending 状态不计入 supported"""
        slots = [
            ClaimSlot(claim_id="c0", description="d", status="supported"),
            ClaimSlot(claim_id="c1", description="d", status="pending"),
            ClaimSlot(claim_id="c2", description="d", status="missing"),
        ]
        assert calculate_coverage(slots) == pytest.approx(1 / 3)


class TestCalculateRequiredCoverage:
    """必填槽位覆盖率计算"""

    def test_all_required_supported(self):
        """3/3 required supported → 1.0"""
        slots = make_claim_slots(missing_indices={4})  # 仅 optional 缺失
        assert calculate_required_coverage(slots) == pytest.approx(1.0)

    def test_one_required_missing(self):
        """2/3 required supported → 2/3"""
        slots = make_claim_slots(missing_indices={0})  # c0 required 缺失
        assert calculate_required_coverage(slots) == pytest.approx(2 / 3)

    def test_no_required_slots(self):
        """无 required 槽位 → 1.0"""
        slots = [
            ClaimSlot(claim_id="c0", description="d", slot_type="optional", status="missing"),
        ]
        assert calculate_required_coverage(slots) == 1.0

    def test_all_required_missing(self):
        """0/3 required supported → 0.0"""
        slots = make_claim_slots(missing_indices={0, 1, 2})
        assert calculate_required_coverage(slots) == 0.0

    def test_slot_type_case_insensitive(self):
        """slot_type 包含 'required'（大小写不敏感）即视为必填"""
        slots = [
            ClaimSlot(claim_id="c0", description="d", slot_type="REQUIRED", status="supported"),
            ClaimSlot(claim_id="c1", description="d", slot_type="required_metric", status="missing"),
        ]
        assert calculate_required_coverage(slots) == pytest.approx(0.5)


# ============================================================
# calculate_authority 测试
# ============================================================

class TestCalculateAuthority:
    """来源权威性计算 — 不同 normative_level 的加权"""

    def test_law(self):
        """法律 → 1.0"""
        items = [make_evidence_item(normative_level="法律")]
        assert calculate_authority(items) == pytest.approx(1.0)

    def test_administrative_regulation(self):
        """行政法规 → 0.9"""
        items = [make_evidence_item(normative_level="行政法规")]
        assert calculate_authority(items) == pytest.approx(0.9)

    def test_department_rule(self):
        """部门规章 → 0.75"""
        items = [make_evidence_item(normative_level="部门规章")]
        assert calculate_authority(items) == pytest.approx(0.75)

    def test_normative_document(self):
        """规范性文件 → 0.6"""
        items = [make_evidence_item(normative_level="规范性文件")]
        assert calculate_authority(items) == pytest.approx(0.6)

    def test_other(self):
        """其他 → 0.4"""
        items = [make_evidence_item(normative_level="其他")]
        assert calculate_authority(items) == pytest.approx(0.4)

    def test_empty_normative_level(self):
        """空 normative_level → 默认权重 0.4"""
        items = [make_evidence_item(normative_level="")]
        assert calculate_authority(items) == pytest.approx(0.4)

    def test_unknown_level(self):
        """无法识别的层级 → 默认权重 0.4"""
        items = [make_evidence_item(normative_level="某未知层级")]
        assert calculate_authority(items) == pytest.approx(0.4)

    def test_keyword_match(self):
        """关键词包含匹配: '行政法规（国务院）' → 0.9"""
        items = [make_evidence_item(normative_level="行政法规（国务院）")]
        assert calculate_authority(items) == pytest.approx(0.9)

    def test_mixed_levels(self):
        """混合层级: 法律(1.0) + 规范性文件(0.6) → 0.8"""
        items = [
            make_evidence_item("ev-1", "chunk-1", normative_level="法律"),
            make_evidence_item("ev-2", "chunk-2", normative_level="规范性文件"),
        ]
        assert calculate_authority(items) == pytest.approx(0.8)

    def test_no_evidence(self):
        """无证据 → 0.0"""
        assert calculate_authority([]) == 0.0

    def test_authority_weights_table(self):
        """验证权威性映射表完整性"""
        assert AUTHORITY_WEIGHTS == {
            "法律": 1.0,
            "行政法规": 0.9,
            "部门规章": 0.75,
            "规范性文件": 0.6,
            "其他": 0.4,
        }


# ============================================================
# calculate_version_validity 测试
# ============================================================

class TestCalculateVersionValidity:
    """版本有效性计算 — active / superseded 加权"""

    def test_all_active(self):
        """全部 active → 1.0"""
        items = [
            make_evidence_item("ev-1", "chunk-1", version_status="active"),
            make_evidence_item("ev-2", "chunk-2", version_status="active"),
        ]
        assert calculate_version_validity(items) == pytest.approx(1.0)

    def test_all_superseded(self):
        """全部 superseded → 0.3"""
        items = [
            make_evidence_item("ev-1", "chunk-1", version_status="superseded"),
            make_evidence_item("ev-2", "chunk-2", version_status="superseded"),
        ]
        assert calculate_version_validity(items) == pytest.approx(0.3)

    def test_mixed_active_superseded(self):
        """混合 active + superseded → (1.0 + 0.3) / 2 = 0.65"""
        items = [
            make_evidence_item("ev-1", "chunk-1", version_status="active"),
            make_evidence_item("ev-2", "chunk-2", version_status="superseded"),
        ]
        assert calculate_version_validity(items) == pytest.approx(0.65)

    def test_unknown_status(self):
        """未识别的版本状态 → 默认权重 0.5"""
        items = [make_evidence_item(version_status="draft")]
        assert calculate_version_validity(items) == pytest.approx(0.5)

    def test_no_evidence(self):
        """无证据 → 0.0"""
        assert calculate_version_validity([]) == 0.0

    def test_version_weights_table(self):
        """验证版本状态权重映射表"""
        assert VERSION_WEIGHTS == {"active": 1.0, "superseded": 0.3}


# ============================================================
# calculate_condition_completeness 测试
# ============================================================

class TestCalculateConditionCompleteness:
    """条件完整性计算 — 缺失条件越多分越低"""

    def test_no_missing(self):
        """0 个缺失 → 1.0"""
        bundle = make_bundle(claim_slots=[], evidence_items=[], missing_conditions=[])
        assert calculate_condition_completeness(bundle) == pytest.approx(1.0)

    def test_one_missing(self):
        """1 个缺失 → 0.85"""
        bundle = make_bundle(claim_slots=[], evidence_items=[], missing_conditions=["c1"])
        assert calculate_condition_completeness(bundle) == pytest.approx(0.85)

    def test_two_missing(self):
        """2 个缺失 → 0.70"""
        bundle = make_bundle(claim_slots=[], evidence_items=[], missing_conditions=["c1", "c2"])
        assert calculate_condition_completeness(bundle) == pytest.approx(0.70)

    def test_many_missing_clamped(self):
        """7 个缺失 → 1 - 0.15*7 = -0.05 → 限制为 0.0"""
        bundle = make_bundle(
            claim_slots=[],
            evidence_items=[],
            missing_conditions=[f"c{i}" for i in range(7)],
        )
        assert calculate_condition_completeness(bundle) == 0.0


# ============================================================
# calculate_channel_consistency 测试
# ============================================================

class TestCalculateChannelConsistency:
    """多通道一致性计算"""

    def test_multi_channel(self):
        """2 个及以上不同通道 → 1.0"""
        items = [
            make_evidence_item("ev-1", "chunk-1", channel="regulation"),
            make_evidence_item("ev-2", "chunk-2", channel="gazette"),
        ]
        bundle = make_bundle(claim_slots=[], evidence_items=items)
        assert calculate_channel_consistency(bundle) == pytest.approx(1.0)

    def test_single_channel(self):
        """仅 1 个通道 → 0.7"""
        items = [
            make_evidence_item("ev-1", "chunk-1", channel="regulation"),
            make_evidence_item("ev-2", "chunk-2", channel="regulation"),
        ]
        bundle = make_bundle(claim_slots=[], evidence_items=items)
        assert calculate_channel_consistency(bundle) == pytest.approx(0.7)

    def test_no_evidence(self):
        """无证据 → 0.0"""
        bundle = make_bundle(claim_slots=[], evidence_items=[])
        assert calculate_channel_consistency(bundle) == 0.0

    def test_no_channel_field(self):
        """无 channel 字段的证据归入 'unknown'，单通道 → 0.7"""
        items = [
            make_evidence_item("ev-1", "chunk-1", channel=""),
            make_evidence_item("ev-2", "chunk-2", channel=""),
        ]
        bundle = make_bundle(claim_slots=[], evidence_items=items)
        assert calculate_channel_consistency(bundle) == pytest.approx(0.7)

    def test_three_channels(self):
        """3 个不同通道 → 1.0"""
        items = [
            make_evidence_item("ev-1", "chunk-1", channel="regulation"),
            make_evidence_item("ev-2", "chunk-2", channel="gazette"),
            make_evidence_item("ev-3", "chunk-3", channel="database"),
        ]
        bundle = make_bundle(claim_slots=[], evidence_items=items)
        assert calculate_channel_consistency(bundle) == pytest.approx(1.0)


# ============================================================
# calculate_conflict_penalty 测试
# ============================================================

class TestCalculateConflictPenalty:
    """冲突惩罚计算 — 不同冲突类型的惩罚值"""

    def test_empty(self):
        """无冲突 → 0.0"""
        assert calculate_conflict_penalty([]) == 0.0

    def test_authority_conflict_dict_type(self):
        """效力冲突（builder 风格 dict 'type'）→ 0.15"""
        conflicts = [{"type": "authority_conflict"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.15)

    def test_version_conflict(self):
        """版本冲突 → 0.10"""
        conflicts = [{"type": "version_conflict"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.10)

    def test_numeric_mismatch(self):
        """数值不一致 → 0.12"""
        conflicts = [{"type": "numeric_mismatch"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.12)

    def test_scope_overlap_default(self):
        """适用范围重叠（未登记）→ 默认 0.08"""
        conflicts = [{"type": "scope_overlap"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(DEFAULT_CONFLICT_PENALTY)
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.08)

    def test_temporal_conflict_default(self):
        """时效冲突（未登记）→ 默认 0.08"""
        conflicts = [{"type": "temporal_conflict"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.08)

    def test_multiple_conflicts_sum(self):
        """多个冲突惩罚累加: 0.15 + 0.10 + 0.12 = 0.37"""
        conflicts = [
            {"type": "authority_conflict"},
            {"type": "version_conflict"},
            {"type": "numeric_mismatch"},
        ]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.37)

    def test_conflict_object(self):
        """Conflict 对象（含 conflict_type 属性）→ 0.15"""
        conflict = Conflict(
            conflict_id="cf-1",
            conflict_type=ConflictType.AUTHORITY_CONFLICT,
            description="效力冲突",
        )
        assert calculate_conflict_penalty([conflict]) == pytest.approx(0.15)

    def test_dict_conflict_type_enum(self):
        """dict 含 'conflict_type' 键（枚举值）→ 0.15"""
        conflicts = [{"conflict_type": ConflictType.AUTHORITY_CONFLICT}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.15)

    def test_dict_conflict_type_chinese(self):
        """dict 含 'conflict_type' 键（中文字符串）→ 0.15"""
        conflicts = [{"conflict_type": "效力冲突"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.15)

    def test_dict_conflict_type_uppercase_name(self):
        """dict 含 'conflict_type' 键（枚举名大写）→ 0.10"""
        conflicts = [{"conflict_type": "VERSION_CONFLICT"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.10)

    def test_unknown_type_default(self):
        """无法识别的冲突类型 → 默认 0.08"""
        conflicts = [{"type": "unknown_conflict"}]
        assert calculate_conflict_penalty(conflicts) == pytest.approx(0.08)

    def test_conflict_penalties_table(self):
        """验证冲突惩罚映射表"""
        assert CONFLICT_PENALTIES == {
            ConflictType.AUTHORITY_CONFLICT: 0.15,
            ConflictType.VERSION_CONFLICT: 0.10,
            ConflictType.NUMERIC_MISMATCH: 0.12,
        }
        assert DEFAULT_CONFLICT_PENALTY == 0.08


# ============================================================
# calculate_missing_penalty 测试
# ============================================================

class TestCalculateMissingPenalty:
    """缺失条件惩罚计算 — 含上限"""

    def test_no_missing(self):
        """0 个缺失 → 0.0"""
        assert calculate_missing_penalty([]) == 0.0

    def test_one_missing(self):
        """1 个缺失 → 0.05"""
        assert calculate_missing_penalty(["c1"]) == pytest.approx(0.05)

    def test_two_missing(self):
        """2 个缺失 → 0.10"""
        assert calculate_missing_penalty(["c1", "c2"]) == pytest.approx(0.10)

    def test_six_missing_at_cap(self):
        """6 个缺失 → 0.30（恰好等于上限）"""
        penalty = calculate_missing_penalty([f"c{i}" for i in range(6)])
        assert penalty == pytest.approx(0.30)

    def test_ten_missing_capped(self):
        """10 个缺失 → 0.30（超过上限，被截断）"""
        penalty = calculate_missing_penalty([f"c{i}" for i in range(10)])
        assert penalty == pytest.approx(0.30)


# ============================================================
# 阈值边界测试
# ============================================================

class TestThresholdBoundary:
    """阈值边界: score 正好等于阈值"""

    def _build_bundle_with_known_score(self):
        """
        构造一个 score 已知的 bundle（有冲突场景，score = 0.8425）

        coverage=1.0, authority=0.95, version=1.0, completeness=1.0, consistency=1.0
        conflict_penalty=0.15 → score = 0.9925 - 0.15 = 0.8425
        """
        return make_bundle(
            claim_slots=make_claim_slots(missing_indices=()),
            evidence_items=make_high_quality_evidence(),
            conflicts=[{"type": "authority_conflict"}],
        )

    def test_score_above_threshold_is_sufficient(self):
        """score 高于阈值 → is_sufficient=True"""
        bundle = self._build_bundle_with_known_score()
        # 实际 score = 0.8425，阈值 0.80 → 充分
        result = SufficiencyScorer(threshold=0.80).score(bundle)
        assert result.is_sufficient is True

    def test_score_below_threshold_is_insufficient(self):
        """score 低于阈值 → is_sufficient=False（默认阈值 0.85）"""
        bundle = self._build_bundle_with_known_score()
        # 实际 score = 0.8425，默认阈值 0.85 → 不足
        result = SufficiencyScorer(threshold=0.85).score(bundle)
        assert result.is_sufficient is False

    def test_score_equals_threshold_is_sufficient(self):
        """score 正好等于阈值 → is_sufficient=True（>= 判定）"""
        bundle = self._build_bundle_with_known_score()
        # 先用零阈值取到真实分数
        actual_score = SufficiencyScorer(threshold=0.0).score(bundle).score

        # 阈值 == score → 充分（>=）
        border_result = SufficiencyScorer(threshold=actual_score).score(bundle)
        assert border_result.is_sufficient is True
        assert border_result.score == pytest.approx(actual_score)

    def test_threshold_just_above_score_is_insufficient(self):
        """阈值仅高于 score 一点点 → is_sufficient=False"""
        bundle = self._build_bundle_with_known_score()
        actual_score = SufficiencyScorer(threshold=0.0).score(bundle).score

        above_result = SufficiencyScorer(threshold=actual_score + 1e-6).score(bundle)
        assert above_result.is_sufficient is False

    def test_custom_threshold_applied(self):
        """自定义阈值生效: 同一 bundle 不同阈值结果不同"""
        bundle = self._build_bundle_with_known_score()
        # score = 0.8425
        assert SufficiencyScorer(threshold=0.84).score(bundle).is_sufficient is True
        assert SufficiencyScorer(threshold=0.85).score(bundle).is_sufficient is False

    def test_score_clamped_to_range(self):
        """score 限制在 [0.0, 1.0]"""
        # 上界: 全覆盖、高质量、无冲突、无缺失 → 接近 1.0
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices=()),
            evidence_items=make_high_quality_evidence(),
        )
        score = SufficiencyScorer().score(bundle).score
        assert 0.0 <= score <= 1.0
        # 下界: 空证据 + 大量缺失 → 0.0
        empty_bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={0, 1, 2, 3, 4, 5}),
            evidence_items=[],
            missing_conditions=[desc for _, desc in CLAIM_SPECS],
        )
        empty_score = SufficiencyScorer().score(empty_bundle).score
        assert 0.0 <= empty_score <= 1.0
        assert empty_score == 0.0


# ============================================================
# SufficiencyScore 数据结构与序列化测试
# ============================================================

class TestSufficiencyScoreDataclass:
    """SufficiencyScore 结果数据结构"""

    def test_to_dict_keys(self):
        """to_dict 包含全部字段"""
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices=()),
            evidence_items=make_high_quality_evidence(),
        )
        data = SufficiencyScorer().score(bundle).to_dict()
        assert set(data.keys()) == {"score", "is_sufficient", "components", "penalties"}

    def test_components_keys(self):
        """components 包含六个维度"""
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices=()),
            evidence_items=make_high_quality_evidence(),
        )
        components = SufficiencyScorer().score(bundle).components
        assert set(components.keys()) == {
            "coverage",
            "required_coverage",
            "authority",
            "version_validity",
            "condition_completeness",
            "multi_channel_consistency",
        }

    def test_penalties_keys(self):
        """penalties 包含冲突惩罚与缺失惩罚"""
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={0}),
            evidence_items=make_low_authority_evidence(),
            conflicts=[{"type": "version_conflict"}],
            missing_conditions=[CLAIM_SPECS[0][1]],
        )
        penalties = SufficiencyScorer().score(bundle).penalties
        assert set(penalties.keys()) == {"conflict_penalty", "missing_penalty"}
        assert penalties["conflict_penalty"] == pytest.approx(0.10)
        assert penalties["missing_penalty"] == pytest.approx(0.05)

    def test_to_dict_rounding(self):
        """to_dict 对分数与各分量四舍五入到 4 位小数"""
        bundle = make_bundle(
            claim_slots=make_claim_slots(missing_indices={5}),
            evidence_items=make_high_quality_evidence("regulation", "regulation"),
        )
        data = SufficiencyScorer().score(bundle).to_dict()
        # score 0.8825 → round(0.8825, 4) == 0.8825
        assert data["score"] == round(0.8825, 4)
        # 各分量也是 4 位小数
        for value in data["components"].values():
            assert round(value, 4) == value

    def test_default_threshold_is_085(self):
        """默认阈值为 0.85"""
        scorer = SufficiencyScorer()
        assert scorer._threshold == 0.85

    def test_scorer_repr(self):
        """__repr__ 显示阈值"""
        scorer = SufficiencyScorer(threshold=0.90)
        assert "0.9" in repr(scorer)
        assert "SufficiencyScorer" in repr(scorer)
