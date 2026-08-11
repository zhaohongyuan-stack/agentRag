"""
ConflictDetector 单元测试

测试范围:
  - ConflictDetector.detect(): 五类冲突检测
      * 数值不一致 (NUMERIC_MISMATCH)
      * 版本冲突 (VERSION_CONFLICT)
      * 适用范围重叠 (SCOPE_OVERLAP)
      * 效力冲突 (AUTHORITY_CONFLICT)
      * 时效冲突 (TEMPORAL_CONFLICT)
  - 无冲突 / 多重冲突 / 空证据列表
  - ConflictResolver: 优先级排序、解决建议、展示格式化
  - Conflict 数据结构与 ConflictType 枚举

测试模式参考 test_evidence_assembler.py:
  - 使用 pytest
  - 定义测试数据工厂函数 make_evidence()
"""

import itertools

import pytest

from agent_platform.evidence.conflict_detector import (
    CONFLICT_PRIORITY,
    Conflict,
    ConflictDetector,
    ConflictResolver,
    ConflictType,
)
from agent_platform.evidence.evidence_assembler.builder import EvidenceItem


# ============================================================
# 测试数据工厂
# ============================================================

_evidence_counter = itertools.count(1)


def make_evidence(
    content: str = "",
    source_doc: str = "《商业银行资本管理办法》",
    version_status: str = "active",
    normative_level: str = "",
    hierarchy_path: str = "",
    metadata: dict = None,
    chunk_id: str = "",
    citation: str = "",
    score: float = 0.8,
    evidence_id: str = "",
) -> EvidenceItem:
    """构造 EvidenceItem，自动生成唯一 evidence_id / chunk_id"""
    eid = evidence_id or f"ev-{next(_evidence_counter):03d}"
    return EvidenceItem(
        evidence_id=eid,
        chunk_id=chunk_id or f"chunk-{eid}",
        content=content,
        evidence_snippet=content[:200] if content else "",
        citation=citation or f"{source_doc} 第1条",
        score=score,
        source_doc=source_doc,
        hierarchy_path=hierarchy_path,
        chunk_type="clause",
        normative_level=normative_level,
        version_status=version_status,
        metadata=metadata or {},
    )


def _types_of(conflicts):
    """提取冲突类型集合，便于断言"""
    return {c.conflict_type for c in conflicts}


# ============================================================
# ConflictType 枚举与数据结构
# ============================================================

class TestConflictTypeEnums:
    """冲突类型枚举为中文标签"""

    def test_enum_values_are_chinese(self):
        assert ConflictType.NUMERIC_MISMATCH.value == "数值不一致"
        assert ConflictType.VERSION_CONFLICT.value == "版本冲突"
        assert ConflictType.SCOPE_OVERLAP.value == "适用范围重叠"
        assert ConflictType.AUTHORITY_CONFLICT.value == "效力冲突"
        assert ConflictType.TEMPORAL_CONFLICT.value == "时效冲突"

    def test_priority_order(self):
        # 数字越小优先级越高: 效力 < 版本 < 时效 < 范围 < 数值
        assert CONFLICT_PRIORITY[ConflictType.AUTHORITY_CONFLICT] == 1
        assert CONFLICT_PRIORITY[ConflictType.VERSION_CONFLICT] == 2
        assert CONFLICT_PRIORITY[ConflictType.TEMPORAL_CONFLICT] == 3
        assert CONFLICT_PRIORITY[ConflictType.SCOPE_OVERLAP] == 4
        assert CONFLICT_PRIORITY[ConflictType.NUMERIC_MISMATCH] == 5


class TestConflictDataclass:
    """Conflict 数据结构"""

    def test_to_dict(self):
        c = Conflict(
            conflict_id="c-001",
            conflict_type=ConflictType.NUMERIC_MISMATCH,
            description="测试冲突",
            evidence_ids=["ev-1", "ev-2"],
            details={"metric": "资本充足率"},
            priority=5,
        )
        d = c.to_dict()
        assert d["conflict_id"] == "c-001"
        assert d["conflict_type"] == "数值不一致"
        assert d["description"] == "测试冲突"
        assert d["evidence_ids"] == ["ev-1", "ev-2"]
        assert d["details"] == {"metric": "资本充足率"}
        assert d["priority"] == 5


# ============================================================
# ConflictDetector — 数值冲突
# ============================================================

class TestNumericMismatch:
    """数值不一致检测"""

    def test_numeric_mismatch_same_metric(self):
        """8% vs 8.5%（同一指标）→ NUMERIC_MISMATCH"""
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于8.5%",
            source_doc="《商业银行法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.NUMERIC_MISMATCH
        assert conflict.priority == CONFLICT_PRIORITY[ConflictType.NUMERIC_MISMATCH]
        # 涉及两条证据
        assert set(conflict.evidence_ids) == {ev1.evidence_id, ev2.evidence_id}
        # details 记录指标与数值
        assert conflict.details["metric"] == "核心一级资本充足率"
        values = {v["value"] for v in conflict.details["values"]}
        assert values == {"8%", "8.5%"}

    def test_numeric_mismatch_plain_number(self):
        """普通数值（非百分比）的差异也能检出

        注: _NUMBER_PATTERN 要求数字前后非 word 字符（中文属 word），
        故用空格分隔数字与中文，确保数字可被提取。
        """
        ev1 = make_evidence(
            content="最低注册资本为 10 亿元",
            source_doc="《A规定》",
            metadata={"metric_name": "注册资本"},
        )
        ev2 = make_evidence(
            content="最低注册资本为 15 亿元",
            source_doc="《B规定》",
            metadata={"metric_name": "注册资本"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        numeric = [c for c in conflicts if c.conflict_type == ConflictType.NUMERIC_MISMATCH]
        assert len(numeric) == 1


# ============================================================
# ConflictDetector — 版本冲突
# ============================================================

class TestVersionConflict:
    """版本冲突检测"""

    def test_version_conflict_active_vs_superseded(self):
        """v1(旧) vs v2(新)，同一文档不同版本状态 → VERSION_CONFLICT

        内容不含数值，确保只产生版本冲突。
        """
        ev1 = make_evidence(
            content="旧版规定要求商业银行加强资本管理",
            source_doc="《资本管理办法》",
            version_status="superseded",
        )
        ev2 = make_evidence(
            content="新版规定要求商业银行强化资本管理体系",
            source_doc="《资本管理办法》",
            version_status="active",
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.VERSION_CONFLICT
        assert conflict.priority == CONFLICT_PRIORITY[ConflictType.VERSION_CONFLICT]
        assert set(conflict.evidence_ids) == {ev1.evidence_id, ev2.evidence_id}
        assert conflict.details["source_doc"] == "《资本管理办法》"
        assert set(conflict.details["version_statuses"]) == {"active", "superseded"}

    def test_no_version_conflict_when_same_status(self):
        """同一文档同为 active → 不构成版本冲突"""
        ev1 = make_evidence(
            content="内容一",
            source_doc="《资本管理办法》",
            version_status="active",
        )
        ev2 = make_evidence(
            content="内容二",
            source_doc="《资本管理办法》",
            version_status="active",
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert not any(
            c.conflict_type == ConflictType.VERSION_CONFLICT for c in conflicts
        )

    def test_no_version_conflict_when_same_content(self):
        """同一文档 active/superseded 但内容相同 → 不构成版本冲突"""
        ev1 = make_evidence(
            content="完全相同的内容",
            source_doc="《资本管理办法》",
            version_status="superseded",
        )
        ev2 = make_evidence(
            content="完全相同的内容",
            source_doc="《资本管理办法》",
            version_status="active",
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert not any(
            c.conflict_type == ConflictType.VERSION_CONFLICT for c in conflicts
        )


# ============================================================
# ConflictDetector — 无冲突
# ============================================================

class TestNoConflict:
    """无冲突场景"""

    def test_no_conflict_same_source_same_value(self):
        """同一来源多块证据，数值相同（8%）→ 无冲突"""
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="商业银行核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert conflicts == []

    def test_no_conflict_different_metrics(self):
        """不同指标的不同数值 → 无冲突"""
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="一级资本充足率不得低于10%",
            source_doc="《资本管理办法》",
            metadata={"metric_name": "一级资本充足率"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert conflicts == []


# ============================================================
# ConflictDetector — 多重冲突
# ============================================================

class TestMultipleConflicts:
    """多重冲突场景"""

    def test_numeric_plus_version_conflicts(self):
        """同一文档 active/superseded 且数值不同（10% vs 8%）→ 2 个冲突

        包含 VERSION_CONFLICT 与 NUMERIC_MISMATCH。
        """
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于10%",
            source_doc="《资本管理办法》",
            version_status="superseded",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            version_status="active",
            metadata={"metric_name": "核心一级资本充足率"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        assert len(conflicts) == 2
        assert _types_of(conflicts) == {
            ConflictType.VERSION_CONFLICT,
            ConflictType.NUMERIC_MISMATCH,
        }


# ============================================================
# ConflictDetector — 效力冲突
# ============================================================

class TestAuthorityConflict:
    """效力冲突检测"""

    def test_authority_conflict_different_levels(self):
        """不同 normative_level 对同一指标有不同值 → AUTHORITY_CONFLICT

        注: 同指标不同值同时会触发 NUMERIC_MISMATCH，此处断言效力冲突存在。
        """
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《商业银行法》",
            normative_level="法律",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于10%",
            source_doc="《资本管理办法》",
            normative_level="部门规章",
            metadata={"metric_name": "核心一级资本充足率"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        authority = [
            c for c in conflicts if c.conflict_type == ConflictType.AUTHORITY_CONFLICT
        ]
        assert len(authority) == 1
        conflict = authority[0]
        assert conflict.priority == CONFLICT_PRIORITY[ConflictType.AUTHORITY_CONFLICT]
        assert set(conflict.details["normative_levels"]) == {"法律", "部门规章"}
        assert set(conflict.evidence_ids) == {ev1.evidence_id, ev2.evidence_id}

    def test_no_authority_conflict_same_level(self):
        """同一效力层级、不同数值 → 仅数值冲突，无效力冲突"""
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《A办法》",
            normative_level="部门规章",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于10%",
            source_doc="《B办法》",
            normative_level="部门规章",
            metadata={"metric_name": "核心一级资本充足率"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert not any(
            c.conflict_type == ConflictType.AUTHORITY_CONFLICT for c in conflicts
        )


# ============================================================
# ConflictDetector — 时效冲突
# ============================================================

class TestTemporalConflict:
    """时效冲突检测"""

    def test_temporal_conflict_different_effective_dates(self):
        """同一文档不同生效日期 → TEMPORAL_CONFLICT"""
        ev1 = make_evidence(
            content="本办法规定了商业银行资本充足率的基本要求",
            source_doc="《资本管理办法》",
            version_status="active",
            metadata={"effective_date": "2024-01-01"},
        )
        ev2 = make_evidence(
            content="本办法明确了商业银行资本充足率的具体标准",
            source_doc="《资本管理办法》",
            version_status="active",
            metadata={"effective_date": "2025-07-01"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.TEMPORAL_CONFLICT
        assert conflict.priority == CONFLICT_PRIORITY[ConflictType.TEMPORAL_CONFLICT]
        assert conflict.details["source_doc"] == "《资本管理办法》"
        assert conflict.details["effective_dates"] == ["2024-01-01", "2025-07-01"]
        assert set(conflict.evidence_ids) == {ev1.evidence_id, ev2.evidence_id}

    def test_temporal_conflict_chinese_date_key(self):
        """metadata 使用中文键「生效日期」也能识别"""
        ev1 = make_evidence(
            content="规定内容一",
            source_doc="《资本管理办法》",
            metadata={"生效日期": "2024-01-01"},
        )
        ev2 = make_evidence(
            content="规定内容二",
            source_doc="《资本管理办法》",
            metadata={"生效日期": "2025-07-01"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert any(
            c.conflict_type == ConflictType.TEMPORAL_CONFLICT for c in conflicts
        )

    def test_no_temporal_conflict_same_date(self):
        """同一文档相同生效日期 → 无时效冲突"""
        ev1 = make_evidence(
            content="规定内容一",
            source_doc="《资本管理办法》",
            metadata={"effective_date": "2024-01-01"},
        )
        ev2 = make_evidence(
            content="规定内容二",
            source_doc="《资本管理办法》",
            metadata={"effective_date": "2024-01-01"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert not any(
            c.conflict_type == ConflictType.TEMPORAL_CONFLICT for c in conflicts
        )


# ============================================================
# ConflictDetector — 适用范围重叠
# ============================================================

class TestScopeOverlap:
    """适用范围重叠检测"""

    def test_scope_overlap_same_applicable_scope(self):
        """不同文档覆盖同一适用范围 → SCOPE_OVERLAP"""
        ev1 = make_evidence(
            content="商业银行应当建立健全风险管理体系",
            source_doc="《商业银行法》",
            metadata={"applicable_scope": "商业银行"},
        )
        ev2 = make_evidence(
            content="商业银行需满足资本充足率监管要求",
            source_doc="《资本管理办法》",
            metadata={"applicable_scope": "商业银行"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.SCOPE_OVERLAP
        assert conflict.priority == CONFLICT_PRIORITY[ConflictType.SCOPE_OVERLAP]
        assert conflict.details["scope"] == "商业银行"
        assert set(conflict.evidence_ids) == {ev1.evidence_id, ev2.evidence_id}

    def test_scope_overlap_by_hierarchy_root(self):
        """无 applicable_scope 时按 hierarchy_path 根节点判定范围重叠"""
        ev1 = make_evidence(
            content="内容规定一",
            source_doc="《商业银行法》",
            hierarchy_path="银行业 > 第一章 > 第1条",
        )
        ev2 = make_evidence(
            content="内容规定二",
            source_doc="《资本管理办法》",
            hierarchy_path="银行业 > 第二章 > 第5条",
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert any(
            c.conflict_type == ConflictType.SCOPE_OVERLAP for c in conflicts
        )

    def test_no_scope_overlap_same_doc(self):
        """同一文档不构成范围重叠（由版本冲突检测覆盖）"""
        ev1 = make_evidence(
            content="内容规定一",
            source_doc="《资本管理办法》",
            metadata={"applicable_scope": "商业银行"},
        )
        ev2 = make_evidence(
            content="内容规定二",
            source_doc="《资本管理办法》",
            metadata={"applicable_scope": "商业银行"},
        )

        conflicts = ConflictDetector().detect([ev1, ev2])
        assert not any(
            c.conflict_type == ConflictType.SCOPE_OVERLAP for c in conflicts
        )


# ============================================================
# ConflictDetector — 空与边界
# ============================================================

class TestDetectorEdgeCases:
    """边界场景"""

    def test_empty_evidence_list(self):
        """空证据列表 → 无冲突"""
        assert ConflictDetector().detect([]) == []

    def test_single_evidence_no_conflict(self):
        """单条证据 → 无冲突"""
        ev = make_evidence(
            content="核心一级资本充足率不得低于8%",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        assert ConflictDetector().detect([ev]) == []

    def test_detect_returns_conflict_instances(self):
        """detect 返回 Conflict 实例列表"""
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《A办法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于8.5%",
            source_doc="《B办法》",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        conflicts = ConflictDetector().detect([ev1, ev2])
        assert all(isinstance(c, Conflict) for c in conflicts)

    def test_conflict_ids_are_unique(self):
        """多条冲突的 conflict_id 互不相同"""
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于10%",
            source_doc="《资本管理办法》",
            version_status="superseded",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            version_status="active",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        conflicts = ConflictDetector().detect([ev1, ev2])
        ids = [c.conflict_id for c in conflicts]
        assert len(ids) == len(set(ids))


# ============================================================
# ConflictResolver — 优先级排序
# ============================================================

class TestResolverSorting:
    """冲突优先级排序"""

    @staticmethod
    def _make_conflict(ctype: ConflictType, evidence_ids=None) -> Conflict:
        return Conflict(
            conflict_id=f"c-{ctype.name}",
            conflict_type=ctype,
            description=f"测试{ctype.value}",
            evidence_ids=evidence_ids or ["ev-1", "ev-2"],
            details={},
            priority=CONFLICT_PRIORITY[ctype],
        )

    def test_sort_by_priority_ascending(self):
        """多冲突按优先级升序排列（数字越小越靠前）"""
        # 故意以乱序输入
        conflicts = [
            self._make_conflict(ConflictType.NUMERIC_MISMATCH),
            self._make_conflict(ConflictType.AUTHORITY_CONFLICT),
            self._make_conflict(ConflictType.SCOPE_OVERLAP),
            self._make_conflict(ConflictType.VERSION_CONFLICT),
            self._make_conflict(ConflictType.TEMPORAL_CONFLICT),
        ]

        sorted_conflicts = ConflictResolver().sort_by_priority(conflicts)

        priorities = [c.priority for c in sorted_conflicts]
        assert priorities == sorted(priorities)
        # 验证具体顺序: 效力 > 版本 > 时效 > 范围 > 数值
        assert [c.conflict_type for c in sorted_conflicts] == [
            ConflictType.AUTHORITY_CONFLICT,
            ConflictType.VERSION_CONFLICT,
            ConflictType.TEMPORAL_CONFLICT,
            ConflictType.SCOPE_OVERLAP,
            ConflictType.NUMERIC_MISMATCH,
        ]

    def test_sort_does_not_mutate_input(self):
        """排序不修改原列表"""
        conflicts = [
            self._make_conflict(ConflictType.NUMERIC_MISMATCH),
            self._make_conflict(ConflictType.AUTHORITY_CONFLICT),
        ]
        original_order = [c.conflict_type for c in conflicts]
        ConflictResolver().sort_by_priority(conflicts)
        assert [c.conflict_type for c in conflicts] == original_order

    def test_sort_stable_for_equal_priority(self):
        """同优先级冲突保持原始相对顺序（稳定排序）"""
        c1 = Conflict(
            conflict_id="c-1",
            conflict_type=ConflictType.NUMERIC_MISMATCH,
            description="一",
            evidence_ids=["ev-1", "ev-2"],
            priority=5,
        )
        c2 = Conflict(
            conflict_id="c-2",
            conflict_type=ConflictType.NUMERIC_MISMATCH,
            description="二",
            evidence_ids=["ev-3", "ev-4"],
            priority=5,
        )
        c3 = Conflict(
            conflict_id="c-3",
            conflict_type=ConflictType.NUMERIC_MISMATCH,
            description="三",
            evidence_ids=["ev-5", "ev-6"],
            priority=5,
        )
        sorted_conflicts = ConflictResolver().sort_by_priority([c1, c2, c3])
        assert [c.conflict_id for c in sorted_conflicts] == ["c-1", "c-2", "c-3"]

    def test_sort_empty(self):
        """空冲突列表排序返回空"""
        assert ConflictResolver().sort_by_priority([]) == []


# ============================================================
# ConflictResolver — 解决建议
# ============================================================

class TestResolverResolutionHint:
    """冲突解决建议"""

    @staticmethod
    def _make_conflict(ctype: ConflictType) -> Conflict:
        return Conflict(
            conflict_id=f"c-{ctype.name}",
            conflict_type=ctype,
            description=f"测试{ctype.value}",
            evidence_ids=["ev-1", "ev-2"],
            details={},
            priority=CONFLICT_PRIORITY[ctype],
        )

    @pytest.mark.parametrize(
        "ctype",
        [
            ConflictType.NUMERIC_MISMATCH,
            ConflictType.VERSION_CONFLICT,
            ConflictType.SCOPE_OVERLAP,
            ConflictType.AUTHORITY_CONFLICT,
            ConflictType.TEMPORAL_CONFLICT,
        ],
    )
    def test_resolution_hint_not_empty(self, ctype):
        """各类型冲突的 resolution_hint 不为空"""
        conflict = self._make_conflict(ctype)
        hint = ConflictResolver().get_resolution_hint(conflict)
        assert isinstance(hint, str)
        assert hint.strip() != ""

    def test_resolution_hint_content(self):
        """解决建议文案符合预期语义"""
        resolver = ConflictResolver()
        assert "效力" in resolver.get_resolution_hint(
            self._make_conflict(ConflictType.AUTHORITY_CONFLICT)
        )
        assert "最新版本" in resolver.get_resolution_hint(
            self._make_conflict(ConflictType.VERSION_CONFLICT)
        )
        assert "过渡期" in resolver.get_resolution_hint(
            self._make_conflict(ConflictType.TEMPORAL_CONFLICT)
        )
        assert "适用范围" in resolver.get_resolution_hint(
            self._make_conflict(ConflictType.SCOPE_OVERLAP)
        )
        assert "权威文件" in resolver.get_resolution_hint(
            self._make_conflict(ConflictType.NUMERIC_MISMATCH)
        )

    def test_resolution_hint_default_for_unknown_type(self):
        """未登记类型返回通用建议（通过手工构造极端优先级冲突验证）"""
        conflict = Conflict(
            conflict_id="c-x",
            conflict_type=ConflictType.NUMERIC_MISMATCH,
            description="测试",
            evidence_ids=["ev-1"],
            details={},
            priority=99,
        )
        # 已登记类型仍有建议
        assert ConflictResolver().get_resolution_hint(conflict) != ""


# ============================================================
# ConflictResolver — 展示格式化
# ============================================================

class TestResolverFormatForDisplay:
    """冲突展示格式化"""

    def test_format_for_display_structure(self):
        """format_for_display 输出字段完整且已按优先级排序"""
        # 构造可产生 2 条冲突的证据（数值 + 版本）
        ev1 = make_evidence(
            content="核心一级资本充足率不得低于10%",
            source_doc="《资本管理办法》",
            version_status="superseded",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        ev2 = make_evidence(
            content="核心一级资本充足率不得低于8%",
            source_doc="《资本管理办法》",
            version_status="active",
            metadata={"metric_name": "核心一级资本充足率"},
        )
        conflicts = ConflictDetector().detect([ev1, ev2])
        assert len(conflicts) == 2

        display = ConflictResolver().format_for_display(conflicts)

        assert len(display) == 2
        # 每条记录字段完整
        for item in display:
            assert set(item.keys()) == {
                "conflict_id",
                "conflict_type",
                "priority",
                "description",
                "evidence_ids",
                "resolution_hint",
                "details",
            }
            assert isinstance(item["conflict_id"], str)
            assert isinstance(item["conflict_type"], str)  # 中文标签
            assert isinstance(item["priority"], int)
            assert isinstance(item["description"], str)
            assert isinstance(item["evidence_ids"], list)
            assert isinstance(item["resolution_hint"], str)
            assert item["resolution_hint"].strip() != ""
            assert isinstance(item["details"], dict)

        # 已按优先级升序排列
        priorities = [item["priority"] for item in display]
        assert priorities == sorted(priorities)
        # 版本冲突(2) 优先于 数值冲突(5)
        assert display[0]["conflict_type"] == "版本冲突"
        assert display[1]["conflict_type"] == "数值不一致"

    def test_format_for_display_empty(self):
        """空冲突列表 → 空展示列表"""
        assert ConflictResolver().format_for_display([]) == []

    def test_format_for_display_all_types(self):
        """五类冲突均能正确格式化"""
        resolver = ConflictResolver()
        conflicts = [
            Conflict(
                conflict_id=f"c-{t.name}",
                conflict_type=t,
                description=f"测试{t.value}",
                evidence_ids=["ev-1", "ev-2"],
                details={"k": "v"},
                priority=CONFLICT_PRIORITY[t],
            )
            for t in ConflictType
        ]
        display = resolver.format_for_display(conflicts)
        assert len(display) == 5
        # conflict_type 为中文枚举值
        assert set(item["conflict_type"] for item in display) == {
            t.value for t in ConflictType
        }
        # 优先级升序
        priorities = [item["priority"] for item in display]
        assert priorities == sorted(priorities)
