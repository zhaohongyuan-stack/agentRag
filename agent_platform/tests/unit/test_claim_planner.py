"""
Claim Planner 单元测试 — M3.1

测试范围:
  - ClaimPlanner: 意图→模板映射、槽位数量、必填/可选判定
  - ClaimPlanner: 未知意图回退到通用模板
  - ClaimPlanner: 自定义模板注册
  - SlotFiller: 证据填充（supported / missing / pending）
  - SlotFiller: 空证据列表处理
  - 槽位 ID 正确性（与模板 slot 字段一致）
"""

import pytest

from agent_platform.evidence.claim_planner import (
    CLAIM_TEMPLATES,
    GENERIC_TEMPLATE,
    ClaimPlanner,
    SlotFiller,
)
from agent_platform.evidence.claim_planner.planner import INTENT_TO_TEMPLATE
from agent_platform.evidence.evidence_assembler.builder import ClaimSlot, EvidenceItem


# ============================================================
# 测试数据工厂
# ============================================================

def make_evidence_item(
    evidence_id: str = "ev-001",
    content: str = "",
    evidence_snippet: str = "",
    score: float = 0.8,
    chunk_id: str = "chunk-001",
    citation: str = "《商业银行资本管理办法》第1条",
    source_doc: str = "《商业银行资本管理办法》",
    hierarchy_path: str = "",
    chunk_type: str = "clause",
) -> EvidenceItem:
    """构造 EvidenceItem 对象"""
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        content=content,
        evidence_snippet=evidence_snippet or content[:200],
        citation=citation,
        score=score,
        source_doc=source_doc,
        hierarchy_path=hierarchy_path,
        chunk_type=chunk_type,
    )


def make_claim_slot(
    claim_id: str = "slot-1",
    description: str = "",
    slot_type: str = "threshold|required",
    status: str = "pending",
    evidence_ids: list = None,
) -> ClaimSlot:
    """构造 ClaimSlot 对象"""
    return ClaimSlot(
        claim_id=claim_id,
        description=description,
        slot_type=slot_type,
        status=status,
        evidence_ids=evidence_ids or [],
    )


def count_required(slots):
    """统计必填槽位数量（slot_type 以 |required 结尾）"""
    return sum(1 for s in slots if s.slot_type.endswith("|required"))


# ============================================================
# ClaimPlanner 模板映射测试
# ============================================================

class TestClaimPlannerTemplates:
    """意图→模板映射测试：验证各模板的槽位数和必填槽位数"""

    @pytest.mark.parametrize(
        "intent,expected_count,expected_required",
        [
            ("threshold", 6, 5),
            ("definition", 3, 3),
            ("table_lookup", 6, 5),
            ("clause_query", 4, 3),
            ("comparison", 5, 5),
        ],
        ids=["阈值", "定义", "表格", "条款", "比较"],
    )
    def test_template_slot_counts(self, intent, expected_count, expected_required):
        """验证各模板的槽位数和必填槽位数"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": intent})
        assert len(slots) == expected_count
        assert count_required(slots) == expected_required

    def test_threshold_query_variant(self):
        """threshold_query 后缀变体映射到 threshold 模板"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "threshold_query"})
        assert len(slots) == 6
        assert count_required(slots) == 5

    def test_definition_query_variant(self):
        """definition_query 后缀变体映射到 definition 模板"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "definition_query"})
        assert len(slots) == 3
        assert count_required(slots) == 3

    @pytest.mark.parametrize(
        "intent",
        ["threshold", "definition", "table_lookup", "clause_query", "comparison"],
    )
    def test_all_slots_initial_status_pending(self, intent):
        """所有槽位初始状态为 pending"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": intent})
        for slot in slots:
            assert slot.status == "pending"

    @pytest.mark.parametrize(
        "intent",
        ["threshold", "definition", "table_lookup", "clause_query", "comparison"],
    )
    def test_all_slots_empty_evidence_ids(self, intent):
        """所有槽位初始无证据绑定"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": intent})
        for slot in slots:
            assert slot.evidence_ids == []

    def test_slot_type_encodes_template_key(self):
        """slot_type 编码了模板来源（格式 {template_key}|required|optional）"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "threshold"})
        for slot in slots:
            assert slot.slot_type.startswith("threshold|")

    def test_explicit_template_key_overrides_intent(self):
        """query_spec 中显式 template_key 优先于 intent 映射"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "threshold", "template_key": "definition"})
        # 应使用 definition 模板（3 个槽位），而非 threshold（6 个）
        assert len(slots) == 3


# ============================================================
# 未知意图回退测试
# ============================================================

class TestClaimPlannerFallback:
    """未知意图回退到通用模板"""

    def test_unknown_intent_falls_back_to_generic(self):
        """未知意图使用通用模板（2 个槽位）"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "totally_unknown_intent"})
        assert len(slots) == len(GENERIC_TEMPLATE)
        assert len(slots) == 2

    def test_missing_intent_falls_back_to_generic(self):
        """query_spec 缺少 intent 字段时使用通用模板"""
        planner = ClaimPlanner()
        slots = planner.plan({})
        assert len(slots) == 2

    def test_generic_template_slots_are_required(self):
        """通用模板槽位均为必填"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "totally_unknown"})
        for slot in slots:
            assert slot.slot_type.endswith("|required")

    def test_generic_slot_ids(self):
        """通用模板的槽位 ID 为 main_answer 和 source"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": "unknown"})
        slot_ids = [s.claim_id for s in slots]
        assert slot_ids == ["main_answer", "source"]


# ============================================================
# 自定义模板注册测试
# ============================================================

class TestClaimPlannerCustomTemplate:
    """自定义模板注册"""

    def test_register_and_use_template(self):
        """注册自定义模板后可通过 template_key 使用"""
        planner = ClaimPlanner()
        custom_template = [
            {"slot": "custom_a", "description": "自定义槽位A", "required": True},
            {"slot": "custom_b", "description": "自定义槽位B", "required": False},
        ]
        planner.register_template("custom", custom_template)

        slots = planner.plan({"template_key": "custom"})
        assert len(slots) == 2
        assert count_required(slots) == 1

    def test_registered_template_slot_ids(self):
        """注册的自定义模板槽位 ID 正确"""
        planner = ClaimPlanner()
        custom_template = [
            {"slot": "alpha", "description": "阿尔法", "required": True},
            {"slot": "beta", "description": "贝塔", "required": True},
        ]
        planner.register_template("my_template", custom_template)

        slots = planner.plan({"template_key": "my_template"})
        assert [s.claim_id for s in slots] == ["alpha", "beta"]

    def test_register_template_via_get(self):
        """get_template 返回注册的模板副本"""
        planner = ClaimPlanner()
        custom_template = [
            {"slot": "x", "description": "X槽位", "required": True},
        ]
        planner.register_template("get_test", custom_template)

        retrieved = planner.get_template("get_test")
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0]["slot"] == "x"

    def test_get_template_returns_none_for_missing(self):
        """获取不存在的模板返回 None"""
        planner = ClaimPlanner()
        assert planner.get_template("nonexistent_key") is None

    def test_register_template_overrides_default(self):
        """注册同名模板覆盖默认模板"""
        planner = ClaimPlanner()
        new_template = [
            {"slot": "only_slot", "description": "唯一槽位", "required": True},
        ]
        planner.register_template("threshold", new_template)

        slots = planner.plan({"intent": "threshold"})
        assert len(slots) == 1
        assert slots[0].claim_id == "only_slot"

    def test_init_with_custom_templates(self):
        """构造时传入自定义模板并与默认模板合并"""
        custom = {
            "regulation": [
                {"slot": "reg_no", "description": "法规编号", "required": True},
                {"slot": "reg_content", "description": "法规内容", "required": True},
            ]
        }
        planner = ClaimPlanner(templates=custom)
        # 自定义模板可用
        slots = planner.plan({"template_key": "regulation"})
        assert len(slots) == 2
        # 默认模板仍可用
        default_slots = planner.plan({"intent": "threshold"})
        assert len(default_slots) == 6


# ============================================================
# 槽位 ID 正确性测试
# ============================================================

class TestSlotIdCorrectness:
    """槽位 ID 与模板 slot 字段一致性"""

    @pytest.mark.parametrize(
        "intent",
        ["threshold", "definition", "table_lookup", "clause_query", "comparison"],
    )
    def test_slot_ids_match_template(self, intent):
        """生成的槽位 ID 与模板定义的 slot 字段一致"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": intent})
        template_key = INTENT_TO_TEMPLATE[intent]
        expected_ids = [s["slot"] for s in CLAIM_TEMPLATES[template_key]]
        actual_ids = [s.claim_id for s in slots]
        assert actual_ids == expected_ids

    @pytest.mark.parametrize(
        "intent",
        ["threshold", "definition", "table_lookup", "clause_query", "comparison"],
    )
    def test_slot_descriptions_match_template(self, intent):
        """生成的槽位描述与模板定义一致"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": intent})
        template_key = INTENT_TO_TEMPLATE[intent]
        expected_descs = [s["description"] for s in CLAIM_TEMPLATES[template_key]]
        actual_descs = [s.description for s in slots]
        assert actual_descs == expected_descs

    @pytest.mark.parametrize(
        "intent",
        ["threshold", "definition", "table_lookup", "clause_query", "comparison"],
    )
    def test_required_flags_match_template(self, intent):
        """必填/可选属性与模板定义一致"""
        planner = ClaimPlanner()
        slots = planner.plan({"intent": intent})
        template_key = INTENT_TO_TEMPLATE[intent]
        expected_required = [s["required"] for s in CLAIM_TEMPLATES[template_key]]
        actual_required = [s.slot_type.endswith("|required") for s in slots]
        assert actual_required == expected_required


# ============================================================
# SlotFiller 测试
# ============================================================

class TestSlotFiller:
    """槽位填充器测试"""

    def test_fill_with_matching_evidence_supported(self):
        """有匹配证据 → supported"""
        filler = SlotFiller()
        slot = make_claim_slot(
            claim_id="metric",
            description="资本充足率",
            slot_type="threshold|required",
        )
        evidence = [
            make_evidence_item(
                evidence_id="ev-1",
                content="核心一级资本充足率不得低于百分之五",
                score=0.9,
            ),
        ]
        result = filler.fill([slot], evidence)
        assert result[0].status == "supported"
        assert "ev-1" in result[0].evidence_ids

    def test_fill_no_match_required_missing(self):
        """无匹配证据 → 必填槽位 missing"""
        filler = SlotFiller()
        slot = make_claim_slot(
            claim_id="metric",
            description="资本充足率指标",
            slot_type="threshold|required",
        )
        evidence = [
            make_evidence_item(
                evidence_id="ev-1",
                content="完全无关的内容关于天气预报",
                score=0.5,
            ),
        ]
        result = filler.fill([slot], evidence)
        assert result[0].status == "missing"
        assert result[0].evidence_ids == []

    def test_fill_no_match_optional_pending(self):
        """无匹配证据 → 可选槽位 pending"""
        filler = SlotFiller()
        slot = make_claim_slot(
            claim_id="exceptions",
            description="例外情况",
            slot_type="threshold|optional",
        )
        evidence = [
            make_evidence_item(
                evidence_id="ev-1",
                content="完全无关的内容关于天气预报",
                score=0.5,
            ),
        ]
        result = filler.fill([slot], evidence)
        assert result[0].status == "pending"
        assert result[0].evidence_ids == []

    def test_fill_empty_evidence_required_missing(self):
        """空证据列表 → 必填槽位 missing"""
        filler = SlotFiller()
        required_slot = make_claim_slot(
            claim_id="metric",
            description="指标",
            slot_type="threshold|required",
        )
        result = filler.fill([required_slot], [])
        assert result[0].status == "missing"
        assert result[0].evidence_ids == []

    def test_fill_empty_evidence_optional_pending(self):
        """空证据列表 → 可选槽位 pending"""
        filler = SlotFiller()
        optional_slot = make_claim_slot(
            claim_id="exceptions",
            description="例外",
            slot_type="threshold|optional",
        )
        result = filler.fill([optional_slot], [])
        assert result[0].status == "pending"
        assert result[0].evidence_ids == []

    def test_fill_empty_evidence_mixed_slots(self):
        """空证据列表 → 必填 missing、可选 pending"""
        filler = SlotFiller()
        slots = [
            make_claim_slot(
                claim_id="req_slot", description="必填项", slot_type="t|required"
            ),
            make_claim_slot(
                claim_id="opt_slot", description="可选项", slot_type="t|optional"
            ),
        ]
        result = filler.fill(slots, [])
        assert result[0].status == "missing"
        assert result[1].status == "pending"

    def test_fill_empty_slots(self):
        """空槽位列表 → 返回空列表"""
        filler = SlotFiller()
        evidence = [make_evidence_item()]
        result = filler.fill([], evidence)
        assert result == []

    def test_fill_max_evidence_per_slot(self):
        """每个槽位最多绑定 max_evidence_per_slot 条证据"""
        filler = SlotFiller(max_evidence_per_slot=2)
        slot = make_claim_slot(
            claim_id="metric",
            description="资本充足率",
            slot_type="threshold|required",
        )
        evidence = [
            make_evidence_item(
                evidence_id=f"ev-{i}",
                content=f"资本充足率相关内容第{i}条",
                score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]
        result = filler.fill([slot], evidence)
        assert result[0].status == "supported"
        assert len(result[0].evidence_ids) == 2

    def test_fill_evidence_sorted_by_score(self):
        """匹配的证据按得分降序绑定"""
        filler = SlotFiller(max_evidence_per_slot=3)
        slot = make_claim_slot(
            claim_id="metric",
            description="资本充足率",
            slot_type="threshold|required",
        )
        evidence = [
            make_evidence_item(
                evidence_id="ev-low", content="资本充足率低分", score=0.3
            ),
            make_evidence_item(
                evidence_id="ev-high", content="资本充足率高分", score=0.95
            ),
            make_evidence_item(
                evidence_id="ev-mid", content="资本充足率中分", score=0.6
            ),
        ]
        result = filler.fill([slot], evidence)
        assert result[0].evidence_ids[0] == "ev-high"

    def test_fill_accepts_dict_evidence(self):
        """SlotFiller 兼容 dict 格式的证据项"""
        filler = SlotFiller()
        slot = make_claim_slot(
            claim_id="metric",
            description="资本充足率",
            slot_type="threshold|required",
        )
        evidence = [
            {
                "evidence_id": "ev-dict",
                "content": "资本充足率不得低于百分之五",
                "evidence_snippet": "资本充足率",
                "score": 0.8,
            }
        ]
        result = filler.fill([slot], evidence)
        assert result[0].status == "supported"
        assert "ev-dict" in result[0].evidence_ids

    def test_fill_multiple_slots_mixed(self):
        """多槽位混合填充: 部分匹配、部分不匹配"""
        filler = SlotFiller()
        slots = [
            make_claim_slot(
                claim_id="matched",
                description="资本充足率",
                slot_type="threshold|required",
            ),
            make_claim_slot(
                claim_id="unmatched",
                description="天气预报",
                slot_type="threshold|required",
            ),
        ]
        evidence = [
            make_evidence_item(
                evidence_id="ev-1",
                content="核心一级资本充足率不得低于百分之五",
                score=0.9,
            ),
        ]
        result = filler.fill(slots, evidence)
        assert result[0].status == "supported"
        assert result[1].status == "missing"

    def test_fill_default_max_evidence_is_three(self):
        """默认每个槽位最多绑定 3 条证据"""
        filler = SlotFiller()
        slot = make_claim_slot(
            claim_id="metric",
            description="资本充足率",
            slot_type="threshold|required",
        )
        evidence = [
            make_evidence_item(
                evidence_id=f"ev-{i}",
                content=f"资本充足率内容{i}",
                score=0.9 - i * 0.05,
            )
            for i in range(5)
        ]
        result = filler.fill([slot], evidence)
        assert len(result[0].evidence_ids) == 3


# ============================================================
# ClaimPlanner + SlotFiller 集成测试
# ============================================================

class TestPlannerFillerIntegration:
    """规划器与填充器集成测试"""

    def test_plan_then_fill_threshold(self):
        """规划 threshold 槽位后填充证据"""
        planner = ClaimPlanner()
        filler = SlotFiller()

        slots = planner.plan({"intent": "threshold"})
        evidence = [
            make_evidence_item(
                evidence_id="ev-1",
                content="适用主体为所有商业银行",
                score=0.9,
            ),
        ]
        result = filler.fill(slots, evidence)

        # 至少有一个槽位被填充为 supported
        statuses = [s.status for s in result]
        assert "supported" in statuses

    def test_plan_then_fill_empty_evidence(self):
        """规划后无证据填充: 必填 missing、可选 pending"""
        planner = ClaimPlanner()
        filler = SlotFiller()

        slots = planner.plan({"intent": "threshold"})
        result = filler.fill(slots, [])

        required_statuses = [
            s.status for s in result if s.slot_type.endswith("|required")
        ]
        optional_statuses = [
            s.status for s in result if s.slot_type.endswith("|optional")
        ]
        assert all(st == "missing" for st in required_statuses)
        assert all(st == "pending" for st in optional_statuses)

    def test_plan_then_fill_all_supported(self):
        """规划后所有槽位都有匹配证据 → 全部 supported"""
        planner = ClaimPlanner()
        filler = SlotFiller()

        slots = planner.plan({"intent": "definition"})
        # 为每个槽位提供匹配的证据
        evidence = [
            make_evidence_item(
                evidence_id="ev-term",
                content="被定义术语为资本充足率",
                score=0.9,
            ),
            make_evidence_item(
                evidence_id="ev-def",
                content="定义内容为商业银行资本占风险加权资产的比例",
                score=0.85,
            ),
            make_evidence_item(
                evidence_id="ev-src",
                content="定义来源为《商业银行资本管理办法》",
                score=0.8,
            ),
        ]
        result = filler.fill(slots, evidence)
        for slot in result:
            assert slot.status == "supported"

    def test_full_pipeline_unknown_intent(self):
        """未知意图 → 通用模板 → 填充证据"""
        planner = ClaimPlanner()
        filler = SlotFiller()

        slots = planner.plan({"intent": "unknown_intent"})
        assert len(slots) == 2

        evidence = [
            make_evidence_item(
                evidence_id="ev-1",
                content="主要回答内容关于资本充足率的要求",
                score=0.9,
            ),
        ]
        result = filler.fill(slots, evidence)
        assert result[0].status == "supported"
