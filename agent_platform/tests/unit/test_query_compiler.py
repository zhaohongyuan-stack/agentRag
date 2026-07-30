"""
Query Compiler 单元测试 (M3.4)

测试范围:
  - IRBuilder: 意图传递、算子生成、答案形状、停止条件、依赖关系
  - LogicalPlanner: 阶段编排、依赖关系
  - PhysicalPlanner: 通道、top_k、rerank 配置、停止条件升级
  - PlanValidator: 合法计划通过、非法计划失败

测试用例表（开发计划）:
  | 测试用例 | 输入意图     | 预期通道                      | 预期停止条件            |
  | 条款     | clause_query | [exact, metadata]            | sufficiency >= 0.85    |
  | 阈值     | threshold    | [lexical, dense, metadata]   | sufficiency >= 0.85    |
  | 表格     | table_lookup | [table, metadata]            | sufficiency >= 0.85    |
  | 非法计划 | 缺必填槽位   | 校验失败                      | —                      |

模式参考: agent_platform/tests/unit/test_evidence_assembler.py
  - 使用 pytest
  - 定义测试数据工厂函数
"""

import pytest

from agent_platform.query_compiler import (
    IRBuilder,
    LogicalPlanner,
    PhysicalPlanner,
    PlanValidator,
)
from agent_platform.query_compiler.query_ir.ir_builder import (
    AnswerShape,
    Operator,
    QueryIR,
    StopCondition,
)
from agent_platform.query_compiler.physical_planner.planner import (
    PhysicalPlan,
    PlanStage,
)
from agent_platform.query_compiler.plan_validator.validator import (
    ValidationResult,
)
from agent_platform.evidence.evidence_assembler.builder import ClaimSlot


# ============================================================
# 测试数据工厂
# ============================================================

def make_query_spec(
    intent: str = "clause_query",
    risk_level: str = "medium",
    entities: list = None,
    constraints: dict = None,
    top_k: int = 10,
    **kwargs,
) -> dict:
    """构造 query_spec 字典

    query_spec 需要有 "intent" 和 "risk_level" 字段。
    其余字段（entities / constraints / top_k 等）为可选项。
    """
    spec = {
        "intent": intent,
        "risk_level": risk_level,
        "entities": entities or [],
        "constraints": constraints or {},
        "top_k": top_k,
    }
    spec.update(kwargs)
    return spec


def make_claims(
    count: int = 2,
    slot_type: str = "metric|required",
) -> list:
    """构造声明槽位列表（dict 格式）

    slot_type 编码 "{template_key}|required" 或 "{template_key}|optional"，
    默认必填（required），用于 PlanValidator 的覆盖校验。
    """
    return [
        {
            "claim_id": f"c{i}",
            "description": f"声明槽位 {i}",
            "slot_type": slot_type,
            "status": "pending",
            "evidence_ids": [],
        }
        for i in range(count)
    ]


def make_claim_objects(count: int = 2, slot_type: str = "metric|required") -> list:
    """构造 ClaimSlot 对象列表（验证 IRBuilder 兼容对象输入）"""
    return [
        ClaimSlot(
            claim_id=f"c{i}",
            description=f"声明槽位 {i}",
            slot_type=slot_type,
            status="pending",
            evidence_ids=[],
        )
        for i in range(count)
    ]


def build_full_pipeline(
    intent: str = "clause_query",
    risk_level: str = "medium",
    claims: list = None,
    slot_type: str = "metric|required",
    query_spec_extra: dict = None,
):
    """端到端构建：IR → 逻辑计划 → 物理计划

    返回 (query_ir, logical_plan, physical_plan, validation_result) 四元组。
    """
    spec = make_query_spec(intent=intent, risk_level=risk_level)
    if query_spec_extra:
        spec.update(query_spec_extra)
    if claims is None:
        claims = make_claims(count=2, slot_type=slot_type)

    ir = IRBuilder().build(spec, claims)
    logical_plan = LogicalPlanner().plan(ir)
    physical_plan = PhysicalPlanner().plan(logical_plan, ir)
    result = PlanValidator().validate(physical_plan, ir)
    return ir, logical_plan, physical_plan, result


def conditions_str(stop_conditions) -> list:
    """提取停止条件的 condition 字符串列表"""
    return [sc.condition for sc in stop_conditions]


# ============================================================
# IRBuilder 测试
# ============================================================

class TestIRBuilder:
    """查询 IR 构建器测试"""

    def test_intent_passed_through(self):
        """意图正确传递到 QueryIR"""
        spec = make_query_spec(intent="threshold", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        assert isinstance(ir, QueryIR)
        assert ir.intent == "threshold"

    def test_risk_level_passed_through(self):
        """风险级别正确传递"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(2))
        assert ir.risk_level == "high"

    def test_operators_generated_correctly(self):
        """算子正确生成：每通道一个 Operator，名称与通道对应"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))

        assert len(ir.retrieval_operators) == 2
        for op in ir.retrieval_operators:
            assert isinstance(op, Operator)
            assert op.name == f"retrieve_{op.channel}"
            assert "claim_ids" in op.params
            assert "top_k" in op.params
            assert op.params["claim_ids"] == ["c0", "c1"]

    def test_clause_query_channels(self):
        """条款意图: channels = [exact, metadata]"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["exact", "metadata"]

    def test_threshold_channels(self):
        """阈值意图: channels = [lexical, dense, metadata]"""
        spec = make_query_spec(intent="threshold", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["lexical", "dense", "metadata"]

    def test_table_lookup_channels(self):
        """表格意图: channels = [table, metadata]"""
        spec = make_query_spec(intent="table_lookup", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["table", "metadata"]

    def test_definition_channels(self):
        """定义意图: channels = [lexical, dense]"""
        spec = make_query_spec(intent="definition", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["lexical", "dense"]

    def test_comparison_channels(self):
        """比较意图: channels = [lexical, dense, metadata]"""
        spec = make_query_spec(intent="comparison", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["lexical", "dense", "metadata"]

    def test_answer_shape_differs_by_intent(self):
        """不同意图生成不同的 AnswerShape"""
        shapes = {}
        for intent in [
            "clause_query",
            "threshold",
            "table_lookup",
            "definition",
            "comparison",
        ]:
            spec = make_query_spec(intent=intent, risk_level="medium")
            ir = IRBuilder().build(spec, make_claims(2))
            assert isinstance(ir.expected_answer, AnswerShape)
            shapes[intent] = ir.expected_answer

        # 条款 → textual
        assert shapes["clause_query"].answer_type == "textual"
        assert "条款" in shapes["clause_query"].format_hint
        # 阈值 → numeric
        assert shapes["threshold"].answer_type == "numeric"
        # 表格 → tabular
        assert shapes["table_lookup"].answer_type == "tabular"
        # 定义 → textual
        assert shapes["definition"].answer_type == "textual"
        assert "定义" in shapes["definition"].format_hint
        # 比较 → comparative
        assert shapes["comparison"].answer_type == "comparative"

    def test_answer_shapes_are_distinct(self):
        """各意图的 AnswerShape 互不相同（answer_type 或 format_hint 不同）"""
        spec_tmpl = make_query_spec(risk_level="medium")
        combos = set()
        for intent in [
            "clause_query",
            "threshold",
            "table_lookup",
            "definition",
            "comparison",
        ]:
            spec = make_query_spec(intent=intent, risk_level="medium")
            ir = IRBuilder().build(spec, make_claims(2))
            combos.add(
                (ir.expected_answer.answer_type, ir.expected_answer.format_hint)
            )
        assert len(combos) == 5

    def test_stop_conditions_medium(self):
        """中风险停止条件包含 sufficiency_score >= 0.85 和 max_retries == 2"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        conds = conditions_str(ir.stop_conditions)
        assert "sufficiency_score >= 0.85" in conds
        assert "max_retries == 2" in conds

    def test_stop_conditions_high_stricter(self):
        """高风险停止条件更严格（sufficiency >= 0.90, max_retries == 3）"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(2))
        conds = conditions_str(ir.stop_conditions)
        assert "sufficiency_score >= 0.90" in conds
        assert "max_retries == 3" in conds

    def test_claims_normalized_from_dict(self):
        """dict 格式声明槽位归一化为 ClaimSlot"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(2)
        ir = IRBuilder().build(spec, claims)
        assert len(ir.claims) == 2
        for c in ir.claims:
            assert isinstance(c, ClaimSlot)
        assert ir.claims[0].claim_id == "c0"
        assert ir.claims[1].claim_id == "c1"

    def test_claims_accept_claimslot_objects(self):
        """IRBuilder 兼容 ClaimSlot 对象输入"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claim_objects(2)
        ir = IRBuilder().build(spec, claims)
        assert len(ir.claims) == 2
        assert ir.claims[0].claim_id == "c0"
        assert ir.claims[1].claim_id == "c1"

    def test_dependencies_parallel_for_medium_risk(self):
        """中风险多声明 → 并行依赖（parallel）"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        assert len(ir.dependencies) == 1
        assert ir.dependencies[0].from_claim == "c0"
        assert ir.dependencies[0].to_claim == "c1"
        assert ir.dependencies[0].type == "parallel"

    def test_dependencies_sequential_for_high_risk(self):
        """高风险多声明 → 串行依赖（sequential）"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(2))
        assert len(ir.dependencies) == 1
        assert ir.dependencies[0].type == "sequential"

    def test_no_dependencies_single_claim(self):
        """单声明无依赖"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(1))
        assert ir.dependencies == []

    def test_empty_claims(self):
        """空声明列表：算子仍按意图生成，但 claim_ids 为空"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, [])
        assert ir.claims == []
        assert len(ir.retrieval_operators) == 2
        for op in ir.retrieval_operators:
            assert op.params["claim_ids"] == []

    def test_explicit_answer_shape_overrides_intent(self):
        """query_spec 显式 answer_shape 优先于意图映射"""
        spec = make_query_spec(
            intent="clause_query",
            risk_level="medium",
            answer_shape={"answer_type": "boolean", "format_hint": "是/否"},
        )
        ir = IRBuilder().build(spec, make_claims(2))
        assert ir.expected_answer.answer_type == "boolean"
        assert ir.expected_answer.format_hint == "是/否"

    def test_explicit_answer_shape_string(self):
        """query_spec answer_shape 为字符串时映射为 answer_type"""
        spec = make_query_spec(
            intent="clause_query",
            risk_level="medium",
            answer_shape="single_value",
        )
        ir = IRBuilder().build(spec, make_claims(2))
        assert ir.expected_answer.answer_type == "numeric"

    def test_explicit_stop_conditions_override(self):
        """query_spec 显式 stop_conditions 优先于风险映射"""
        spec = make_query_spec(
            intent="clause_query",
            risk_level="high",
            stop_conditions=[
                {"condition": "sufficiency_score >= 0.99", "description": "自定义"},
            ],
        )
        ir = IRBuilder().build(spec, make_claims(2))
        conds = conditions_str(ir.stop_conditions)
        assert conds == ["sufficiency_score >= 0.99"]

    def test_top_k_propagated_to_operators(self):
        """query_spec.top_k 传递到算子参数"""
        spec = make_query_spec(intent="threshold", risk_level="medium", top_k=30)
        ir = IRBuilder().build(spec, make_claims(2))
        for op in ir.retrieval_operators:
            assert op.params["top_k"] == 30

    def test_to_dict_serializable(self):
        """QueryIR 可序列化为字典"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        data = ir.to_dict()
        assert data["intent"] == "clause_query"
        assert data["risk_level"] == "medium"
        assert len(data["retrieval_operators"]) == 2
        assert "expected_answer" in data
        assert "stop_conditions" in data


# ============================================================
# LogicalPlanner 测试
# ============================================================

class TestLogicalPlanner:
    """逻辑计划生成器测试"""

    def test_single_stage_parallel(self):
        """无串行依赖 → 单阶段并行执行所有算子"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)

        assert len(plan.stages) == 1
        assert plan.stages[0].name == "retrieve"
        assert plan.stages[0].can_parallel is True
        assert plan.stages[0].dependencies == []
        # 2 个算子全部放入 retrieve 阶段
        assert len(plan.stages[0].operators) == 2

    def test_multiple_stages_for_high_risk(self):
        """高风险串行依赖 → 检索阶段 + 逐声明校验阶段"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)

        # retrieve + verify_c1
        assert len(plan.stages) == 2
        assert plan.stages[0].name == "retrieve"
        assert plan.stages[1].name == "verify_c1"

    def test_stage_count_three_claims_high_risk(self):
        """3 声明高风险 → 1 检索 + 2 校验 = 3 阶段"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(3))
        plan = LogicalPlanner().plan(ir)

        assert len(plan.stages) == 3
        names = [s.name for s in plan.stages]
        assert names == ["retrieve", "verify_c1", "verify_c2"]

    def test_stage_dependencies_sequential(self):
        """串行计划阶段依赖关系正确（链式）"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(3))
        plan = LogicalPlanner().plan(ir)

        # retrieve 无依赖
        assert plan.stages[0].dependencies == []
        # verify_c1 依赖 retrieve
        assert plan.stages[1].dependencies == ["retrieve"]
        # verify_c2 依赖 verify_c1
        assert plan.stages[2].dependencies == ["verify_c1"]

    def test_verify_stages_no_operators(self):
        """校验阶段无检索算子"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)

        verify_stage = plan.stages[1]
        assert verify_stage.operators == []
        assert verify_stage.can_parallel is False

    def test_intent_passed_to_plan(self):
        """意图传递到逻辑计划"""
        spec = make_query_spec(intent="threshold", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)
        assert plan.intent == "threshold"

    def test_risk_level_passed_to_plan(self):
        """风险级别传递到逻辑计划"""
        spec = make_query_spec(intent="clause_query", risk_level="high")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)
        assert plan.risk_level == "high"

    def test_plan_id_generated(self):
        """逻辑计划生成唯一 plan_id"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)
        assert plan.plan_id.startswith("lp-")

    def test_completion_condition_set(self):
        """阶段完成条件已设置"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, make_claims(2))
        plan = LogicalPlanner().plan(ir)
        assert plan.stages[0].completion_condition == "all_operators_returned"


# ============================================================
# PhysicalPlanner 测试
# ============================================================

class TestPhysicalPlanner:
    """物理计划生成器测试"""

    def test_clause_query_channels(self):
        """条款意图: channels = [exact, metadata]"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query")
        retrieve = plan.stages[0]
        assert retrieve.channels == ["exact", "metadata"]

    def test_threshold_config(self):
        """阈值意图: channels=[lexical,dense,metadata], top_k=20, rerank=True"""
        ir, _, plan, _ = build_full_pipeline(intent="threshold")
        retrieve = plan.stages[0]
        assert retrieve.channels == ["lexical", "dense", "metadata"]
        assert retrieve.top_k == 20
        assert retrieve.rerank is True

    def test_table_lookup_config(self):
        """表格意图: channels=[table,metadata], top_k=5, rerank=False"""
        ir, _, plan, _ = build_full_pipeline(intent="table_lookup")
        retrieve = plan.stages[0]
        assert retrieve.channels == ["table", "metadata"]
        assert retrieve.top_k == 5
        assert retrieve.rerank is False

    def test_definition_channels(self):
        """定义意图: channels = [lexical, dense]"""
        ir, _, plan, _ = build_full_pipeline(intent="definition")
        retrieve = plan.stages[0]
        assert retrieve.channels == ["lexical", "dense"]

    def test_comparison_channels(self):
        """比较意图: channels = [lexical, dense, metadata]"""
        ir, _, plan, _ = build_full_pipeline(intent="comparison")
        retrieve = plan.stages[0]
        assert retrieve.channels == ["lexical", "dense", "metadata"]

    def test_stop_conditions_default(self):
        """停止条件包含 sufficiency_score >= 0.85 和 max_retries == 2"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query", risk_level="medium")
        conds = conditions_str(plan.stop_conditions)
        assert "sufficiency_score >= 0.85" in conds
        assert "max_retries == 2" in conds

    def test_stop_conditions_upgrade_for_high_risk(self):
        """高风险 IR 使物理计划停止条件升级到 0.90 / 3"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query", risk_level="high")
        conds = conditions_str(plan.stop_conditions)
        # 高风险 IR 的 sufficiency=0.90 > 0.85 → 升级
        assert "sufficiency_score >= 0.9" in conds
        # 高风险 IR 的 max_retries=3 > 2 → 升级
        assert "max_retries == 3" in conds

    def test_stop_conditions_not_downgraded(self):
        """低风险 IR 不应降低基础停止条件（保持 0.85 / 2）"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query", risk_level="low")
        conds = conditions_str(plan.stop_conditions)
        # 低风险 IR sufficiency=0.75 < 0.85，不升级，保持 0.85
        assert "sufficiency_score >= 0.85" in conds
        assert "max_retries == 2" in conds

    def test_operations_built_with_rerank(self):
        """操作序列: 多通道+rerank → retrieve:* + rerank + fuse"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query")
        ops = plan.stages[0].operations
        # clause_query: channels=[exact, metadata], rerank=True
        assert "retrieve:exact" in ops
        assert "retrieve:metadata" in ops
        assert "rerank" in ops
        assert "fuse" in ops

    def test_operations_built_without_rerank(self):
        """操作序列: 无 rerank → 不含 rerank，但多通道仍 fuse"""
        ir, _, plan, _ = build_full_pipeline(intent="table_lookup")
        ops = plan.stages[0].operations
        # table_lookup: channels=[table, metadata], rerank=False
        assert "retrieve:table" in ops
        assert "retrieve:metadata" in ops
        assert "rerank" not in ops
        assert "fuse" in ops

    def test_claim_ids_collected(self):
        """物理阶段 claim_ids 覆盖所有声明"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query")
        retrieve = plan.stages[0]
        assert "c0" in retrieve.claim_ids
        assert "c1" in retrieve.claim_ids

    def test_timeout_positive(self):
        """每个阶段超时时间 > 0"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query")
        for stage in plan.stages:
            assert stage.timeout_ms > 0

    def test_budget_ms_set(self):
        """物理计划预算时间已设置"""
        ir, _, plan, _ = build_full_pipeline(intent="threshold")
        assert plan.budget_ms == 8000

    def test_plan_id_generated(self):
        """物理计划生成唯一 plan_id"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query")
        assert plan.plan_id.startswith("pp-")

    def test_intent_passed_to_physical_plan(self):
        """意图传递到物理计划"""
        ir, _, plan, _ = build_full_pipeline(intent="threshold")
        assert plan.intent == "threshold"

    def test_verify_stage_has_no_channels(self):
        """高风险串行计划：校验阶段无通道、top_k=0"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query", risk_level="high")
        # retrieve + verify_c1
        assert len(plan.stages) == 2
        verify_stage = plan.stages[1]
        assert verify_stage.channels == []
        assert verify_stage.top_k == 0
        assert verify_stage.rerank is False

    def test_to_dict_serializable(self):
        """物理计划可序列化为字典"""
        ir, _, plan, _ = build_full_pipeline(intent="clause_query")
        data = plan.to_dict()
        assert data["intent"] == "clause_query"
        assert "stages" in data
        assert "stop_conditions" in data
        assert "budget_ms" in data


# ============================================================
# PlanValidator 测试
# ============================================================

class TestPlanValidator:
    """计划合法性校验器测试"""

    def test_valid_plan_passes(self):
        """合法计划通过校验"""
        ir, _, plan, result = build_full_pipeline(intent="clause_query")
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.errors == []

    def test_valid_plan_threshold(self):
        """阈值意图合法计划通过校验"""
        ir, _, plan, result = build_full_pipeline(intent="threshold")
        assert result.is_valid is True

    def test_valid_plan_table_lookup(self):
        """表格意图合法计划通过校验"""
        ir, _, plan, result = build_full_pipeline(intent="table_lookup")
        assert result.is_valid is True

    def test_invalid_missing_required_claim(self):
        """非法计划：缺少必填槽位覆盖 → is_valid=False"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(count=2, slot_type="metric|required")
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 构造不覆盖任何必填声明的物理计划（清空 claim_ids）
        broken_stages = [
            PlanStage(
                name=stage.name,
                channels=list(stage.channels),
                top_k=stage.top_k,
                rerank=stage.rerank,
                timeout_ms=stage.timeout_ms,
                operations=list(stage.operations),
                condition=stage.condition,
                claim_ids=[],  # 清空，模拟缺失覆盖
            )
            for stage in physical_plan.stages
        ]
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=broken_stages,
            stop_conditions=list(physical_plan.stop_conditions),
            budget_ms=physical_plan.budget_ms,
        )

        result = PlanValidator().validate(broken_plan, ir)
        assert result.is_valid is False
        assert len(result.errors) > 0
        # 错误信息应提及未覆盖的声明 ID
        assert any("c0" in e for e in result.errors)
        assert any("c1" in e for e in result.errors)

    def test_invalid_partial_missing_claim(self):
        """非法计划：部分必填槽位缺失覆盖 → is_valid=False"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(count=2, slot_type="metric|required")
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 仅覆盖 c0，不覆盖 c1
        broken_stages = [
            PlanStage(
                name=stage.name,
                channels=list(stage.channels),
                top_k=stage.top_k,
                rerank=stage.rerank,
                timeout_ms=stage.timeout_ms,
                operations=list(stage.operations),
                condition=stage.condition,
                claim_ids=["c0"],  # 只覆盖 c0
            )
            for stage in physical_plan.stages
        ]
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=broken_stages,
            stop_conditions=list(physical_plan.stop_conditions),
            budget_ms=physical_plan.budget_ms,
        )

        result = PlanValidator().validate(broken_plan, ir)
        assert result.is_valid is False
        assert any("c1" in e for e in result.errors)
        # c0 已覆盖，不应出现在错误中
        assert all("c0" not in e for e in result.errors)

    def test_optional_claim_not_required(self):
        """optional 声明未被覆盖不报错"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(count=2, slot_type="metric|optional")
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 清空 claim_ids，但声明是 optional，不应报错
        broken_stages = [
            PlanStage(
                name=stage.name,
                channels=list(stage.channels),
                top_k=stage.top_k,
                rerank=stage.rerank,
                timeout_ms=stage.timeout_ms,
                operations=list(stage.operations),
                condition=stage.condition,
                claim_ids=[],
            )
            for stage in physical_plan.stages
        ]
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=broken_stages,
            stop_conditions=list(physical_plan.stop_conditions),
            budget_ms=physical_plan.budget_ms,
        )

        result = PlanValidator().validate(broken_plan, ir)
        # optional 声明未覆盖不算错误
        assert all("c0" not in e for e in result.errors)
        assert all("c1" not in e for e in result.errors)

    def test_empty_stop_conditions_invalid(self):
        """空停止条件 → is_valid=False"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(2)
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 清空停止条件
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=list(physical_plan.stages),
            stop_conditions=[],
            budget_ms=physical_plan.budget_ms,
        )
        result = PlanValidator().validate(broken_plan, ir)
        assert result.is_valid is False
        assert any("停止条件" in e for e in result.errors)

    def test_zero_top_k_invalid(self):
        """有通道阶段 top_k=0 → is_valid=False"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(2)
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 将 retrieve 阶段 top_k 设为 0（但保留通道）
        broken_stages = []
        for stage in physical_plan.stages:
            if stage.channels:
                broken_stages.append(
                    PlanStage(
                        name=stage.name,
                        channels=list(stage.channels),
                        top_k=0,
                        rerank=stage.rerank,
                        timeout_ms=stage.timeout_ms,
                        operations=list(stage.operations),
                        condition=stage.condition,
                        claim_ids=list(stage.claim_ids),
                    )
                )
            else:
                broken_stages.append(stage)
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=broken_stages,
            stop_conditions=list(physical_plan.stop_conditions),
            budget_ms=physical_plan.budget_ms,
        )
        result = PlanValidator().validate(broken_plan, ir)
        assert result.is_valid is False
        assert any("top_k" in e for e in result.errors)

    def test_zero_timeout_invalid(self):
        """阶段超时时间=0 → is_valid=False"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(2)
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        broken_stages = [
            PlanStage(
                name=stage.name,
                channels=list(stage.channels),
                top_k=stage.top_k,
                rerank=stage.rerank,
                timeout_ms=0,
                operations=list(stage.operations),
                condition=stage.condition,
                claim_ids=list(stage.claim_ids),
            )
            for stage in physical_plan.stages
        ]
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=broken_stages,
            stop_conditions=list(physical_plan.stop_conditions),
            budget_ms=physical_plan.budget_ms,
        )
        result = PlanValidator().validate(broken_plan, ir)
        assert result.is_valid is False
        assert any("timeout_ms" in e for e in result.errors)

    def test_empty_claims_warning(self):
        """QueryIR 无声明槽位时产生警告（但不阻断）"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        ir = IRBuilder().build(spec, [])
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)
        result = PlanValidator().validate(physical_plan, ir)
        # 无声明时无 claim 覆盖错误，但应有警告
        assert any("声明槽位" in w for w in result.warnings)

    def test_validation_result_to_dict(self):
        """ValidationResult 可序列化"""
        ir, _, plan, result = build_full_pipeline(intent="clause_query")
        data = result.to_dict()
        assert "is_valid" in data
        assert "errors" in data
        assert "warnings" in data


# ============================================================
# 端到端流水线测试（开发计划测试用例表）
# ============================================================

class TestEndToEndPipeline:
    """端到端流水线测试：IR → 逻辑计划 → 物理计划 → 校验"""

    def test_clause_query_pipeline(self):
        """条款意图完整流水线：channels=[exact,metadata], sufficiency>=0.85"""
        ir, logical_plan, physical_plan, result = build_full_pipeline(
            intent="clause_query", risk_level="medium"
        )
        # IR 层：意图与算子
        assert ir.intent == "clause_query"
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["exact", "metadata"]

        # 物理计划层：通道与停止条件
        assert physical_plan.stages[0].channels == ["exact", "metadata"]
        conds = conditions_str(physical_plan.stop_conditions)
        assert "sufficiency_score >= 0.85" in conds

        # 校验通过
        assert result.is_valid is True

    def test_threshold_pipeline(self):
        """阈值意图完整流水线：channels=[lexical,dense,metadata], sufficiency>=0.85"""
        ir, logical_plan, physical_plan, result = build_full_pipeline(
            intent="threshold", risk_level="medium"
        )
        assert ir.intent == "threshold"
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["lexical", "dense", "metadata"]

        assert physical_plan.stages[0].channels == ["lexical", "dense", "metadata"]
        conds = conditions_str(physical_plan.stop_conditions)
        assert "sufficiency_score >= 0.85" in conds

        assert result.is_valid is True

    def test_table_lookup_pipeline(self):
        """表格意图完整流水线：channels=[table,metadata], sufficiency>=0.85"""
        ir, logical_plan, physical_plan, result = build_full_pipeline(
            intent="table_lookup", risk_level="medium"
        )
        assert ir.intent == "table_lookup"
        channels = [op.channel for op in ir.retrieval_operators]
        assert channels == ["table", "metadata"]

        assert physical_plan.stages[0].channels == ["table", "metadata"]
        conds = conditions_str(physical_plan.stop_conditions)
        assert "sufficiency_score >= 0.85" in conds

        assert result.is_valid is True

    def test_invalid_plan_missing_required_claim_pipeline(self):
        """非法计划：缺少必填槽位覆盖 → 校验失败"""
        spec = make_query_spec(intent="clause_query", risk_level="medium")
        claims = make_claims(count=2, slot_type="metric|required")
        ir = IRBuilder().build(spec, claims)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 构造缺少必填槽位覆盖的计划
        broken_stages = [
            PlanStage(
                name=stage.name,
                channels=list(stage.channels),
                top_k=stage.top_k,
                rerank=stage.rerank,
                timeout_ms=stage.timeout_ms,
                operations=list(stage.operations),
                condition=stage.condition,
                claim_ids=[],
            )
            for stage in physical_plan.stages
        ]
        broken_plan = PhysicalPlan(
            plan_id=physical_plan.plan_id,
            intent=physical_plan.intent,
            stages=broken_stages,
            stop_conditions=list(physical_plan.stop_conditions),
            budget_ms=physical_plan.budget_ms,
        )

        result = PlanValidator().validate(broken_plan, ir)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_definition_pipeline(self):
        """定义意图完整流水线：channels=[lexical,dense]"""
        ir, _, physical_plan, result = build_full_pipeline(
            intent="definition", risk_level="medium"
        )
        assert physical_plan.stages[0].channels == ["lexical", "dense"]
        assert result.is_valid is True

    def test_comparison_pipeline(self):
        """比较意图完整流水线：channels=[lexical,dense,metadata]"""
        ir, _, physical_plan, result = build_full_pipeline(
            intent="comparison", risk_level="medium"
        )
        assert physical_plan.stages[0].channels == ["lexical", "dense", "metadata"]
        assert result.is_valid is True

    def test_full_pipeline_max_retries(self):
        """完整流水线停止条件包含 max_retries == 2"""
        ir, _, physical_plan, _ = build_full_pipeline(
            intent="clause_query", risk_level="medium"
        )
        conds = conditions_str(physical_plan.stop_conditions)
        assert "max_retries == 2" in conds

    def test_high_risk_pipeline_multi_stage(self):
        """高风险完整流水线：多阶段 + 升级停止条件"""
        ir, logical_plan, physical_plan, result = build_full_pipeline(
            intent="clause_query", risk_level="high"
        )
        # 串行依赖 → 多阶段
        assert len(logical_plan.stages) >= 2
        # 停止条件升级
        conds = conditions_str(physical_plan.stop_conditions)
        assert "sufficiency_score >= 0.9" in conds
        assert "max_retries == 3" in conds
        # 高风险计划仍应通过校验
        assert result.is_valid is True
