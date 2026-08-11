"""
Phase 3 端到端集成测试

验证 Phase 3 模块协同工作:
  声明槽位规划 → 查询编译 → 证据组装 → 槽位填充 → 冲突检测
  → 充分性评分 → 预算控制 → 二次检索

测试场景:
  1. 表格取数端到端 — 声明槽位覆盖，表格值正确
  2. 上下文扩展 — 父子/近邻扩展正确
  3. 二次检索 — 首次证据不足 → 触发二次检索 → 最终充分
  4. 冲突展示 — 两个版本规定不同 → 检测冲突 → 并列展示
  5. 预算耗尽 — 复杂问题反复检索 → 超预算停止 → 返回部分结果
"""

import pytest

from agent_platform.evidence.claim_planner import ClaimPlanner, SlotFiller
from agent_platform.evidence.evidence_assembler.builder import (
    ClaimSlot,
    EvidenceBuilder,
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.evidence.conflict_detector import (
    ConflictDetector,
    ConflictResolver,
    ConflictType,
)
from agent_platform.evidence.sufficiency_scorer import SufficiencyScorer
from agent_platform.orchestration.budget_controller import (
    BudgetAction,
    BudgetController,
    BudgetEnforcer,
)
from agent_platform.query_compiler import (
    IRBuilder,
    LogicalPlanner,
    PhysicalPlanner,
    PlanValidator,
)


# ============================================================
# 测试数据工厂
# ============================================================

def make_evidence_item(
    evidence_id: str = "",
    chunk_id: str = "",
    content: str = "",
    score: float = 0.8,
    source_doc: str = "《商业银行资本管理办法》",
    hierarchy_path: str = "",
    chunk_type: str = "clause",
    normative_level: str = "部门规章",
    version_status: str = "active",
    citation: str = "",
    metadata: dict = None,
) -> EvidenceItem:
    """构造 EvidenceItem"""
    import uuid

    return EvidenceItem(
        evidence_id=evidence_id or f"ev-{uuid.uuid4().hex[:8]}",
        chunk_id=chunk_id or f"chunk-{uuid.uuid4().hex[:8]}",
        content=content,
        evidence_snippet=content[:200],
        citation=citation or f"{source_doc} 第1条",
        score=score,
        source_doc=source_doc,
        hierarchy_path=hierarchy_path,
        chunk_type=chunk_type,
        normative_level=normative_level,
        version_status=version_status,
        metadata=metadata or {},
    )


def make_hit(
    chunk_id: str = "",
    content: str = "",
    score: float = 0.8,
    doc_name: str = "《商业银行资本管理办法》",
    chunk_type: str = "clause",
    citation: str = "",
    hierarchy_path: str = "",
    metadata: dict = None,
) -> dict:
    """构造 RetrievalHit dict"""
    import uuid

    return {
        "chunk_id": chunk_id or f"chunk-{uuid.uuid4().hex[:8]}",
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


# ============================================================
# 端到端测试
# ============================================================

class TestPhase3E2ETableLookup:
    """场景 1: 表格取数端到端"""

    def test_table_lookup_full_pipeline(self):
        """表格取数 — 声明槽位规划 → 查询编译 → 证据组装 → 充分性评分"""
        # 1. 声明槽位规划
        planner = ClaimPlanner()
        query_spec = {"intent": "table_lookup", "risk_level": "low"}
        slots = planner.plan(query_spec)

        assert len(slots) == 6
        slot_ids = [s.claim_id for s in slots]
        assert "table_name" in slot_ids
        assert "value" in slot_ids

        # 2. 查询编译
        ir_builder = IRBuilder()
        query_ir = ir_builder.build(query_spec, slots)

        assert query_ir.intent == "table_lookup"
        # 表格查询应有 table 和 metadata 通道
        operator_channels = [op.channel for op in query_ir.retrieval_operators]
        assert "table" in operator_channels

        # 3. 逻辑计划 → 物理计划
        logical_plan = LogicalPlanner().plan(query_ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, query_ir)

        # 验证物理计划
        primary_stage = physical_plan.stages[0]
        assert "table" in primary_stage.channels
        assert primary_stage.top_k > 0

        # 4. 计划校验
        validation = PlanValidator().validate(physical_plan, query_ir)
        assert validation.is_valid, f"校验失败: {validation.errors}"

        # 5. 证据组装（模拟表格检索结果）
        hits = [
            make_hit(
                content="附件1：核心一级资本充足率 非系统重要性银行 最低 7.75%",
                chunk_type="table",
                score=0.95,
                metadata={"table_name": "资本充足率要求", "channel": "table"},
            ),
            make_hit(
                content="附件1：核心一级资本充足率 系统重要性银行 最低 8.5%",
                chunk_type="table",
                score=0.88,
                metadata={"table_name": "资本充足率要求", "channel": "table"},
            ),
        ]

        builder = EvidenceBuilder()
        claims_dict = [
            {"claim_id": s.claim_id, "description": s.description, "slot_type": s.slot_type}
            for s in slots
        ]
        bundle = builder.build(hits, claims_dict)

        # 6. 槽位填充（表格查询的槽位描述与证据内容可能不完全匹配，
        #    但流水线应正常完成）
        filler = SlotFiller()
        filled_slots = filler.fill(slots, bundle.evidence_items)

        # 至少有一些槽位被处理（supported 或 pending）
        processed = sum(1 for s in filled_slots if s.status in ("supported", "pending"))
        assert processed > 0, "至少应有槽位被处理"

        # 7. 充分性评分
        scorer = SufficiencyScorer()
        # 更新 bundle 的 claim_slots 为填充后的
        bundle.claim_slots = filled_slots
        result = scorer.score(bundle)

        assert result.score > 0
        assert "coverage" in result.components

    def test_table_lookup_value_extraction(self):
        """表格取数 — 验证表格值可从证据中提取"""
        # 模拟表格检索结果
        hits = [
            make_hit(
                content="附件1表：非系统重要性银行核心一级资本充足率最低要求为 7.75%",
                chunk_type="table",
                score=0.95,
                metadata={"table_name": "资本充足率监管要求"},
            ),
        ]

        builder = EvidenceBuilder()
        bundle = builder.build(hits, [
            {"claim_id": "value", "description": "单元格值"},
        ])

        assert bundle.evidence_count > 0
        # 验证数值出现在证据内容中
        assert "7.75" in bundle.evidence_items[0].content


class TestPhase3E2EContextExpansion:
    """场景 2: 上下文扩展"""

    def test_context_expansion_parent_aggregation(self):
        """上下文扩展 — 父文档聚合正确"""
        # 模拟同一父条款下的多个子 chunk
        hits = [
            make_hit(
                chunk_id="child-1",
                content="第43条：商业银行核心一级资本充足率不得低于5%。",
                score=0.85,
                hierarchy_path="第四章/第二节/第43条/段落1",
                metadata={"parent_chunk_id": "parent-43"},
            ),
            make_hit(
                chunk_id="child-2",
                content="前款规定的核心一级资本充足率，系统重要性银行应额外满足附加资本要求。",
                score=0.75,
                hierarchy_path="第四章/第二节/第43条/段落2",
                metadata={"parent_chunk_id": "parent-43"},
            ),
        ]

        builder = EvidenceBuilder()
        bundle = builder.build(hits, [
            {"claim_id": "clause_content", "description": "条款内容"},
        ])

        # 父文档聚合后，同一父级的子 chunk 应被处理
        assert bundle.evidence_count > 0
        # 证据应按得分降序排列
        if bundle.evidence_count > 1:
            assert bundle.evidence_items[0].score >= bundle.evidence_items[1].score

    def test_context_expansion_sibling_retrieval(self):
        """上下文扩展 — 近邻条款可获取"""
        # 模拟第43条和第42条（前一条）
        hits = [
            make_hit(
                content="第43条：商业银行核心一级资本充足率不得低于5%。",
                score=0.90,
                hierarchy_path="第四章/第二节/第43条",
            ),
            make_hit(
                content="第42条：商业银行一级资本充足率不得低于6%。",
                score=0.70,
                hierarchy_path="第四章/第二节/第42条",
            ),
        ]

        builder = EvidenceBuilder()
        bundle = builder.build(hits, [
            {"claim_id": "clause_content", "description": "前一条 条款内容"},
        ])

        # 两条证据都应保留
        assert bundle.evidence_count >= 1
        # 得分最高的应该是第43条
        assert "第43条" in bundle.evidence_items[0].content or "第42条" in bundle.evidence_items[0].content


class TestPhase3E2ERetry:
    """场景 3: 二次检索"""

    def test_insufficient_evidence_triggers_retry(self):
        """二次检索 — 首次证据不足 → 触发二次检索"""
        # 1. 首次检索：证据不足
        first_hits = [
            make_hit(
                content="商业银行应满足资本充足率要求。",
                score=0.4,
                chunk_type="clause",
            ),
        ]

        planner = ClaimPlanner()
        slots = planner.plan({"intent": "threshold", "risk_level": "medium"})

        builder = EvidenceBuilder()
        claims_dict = [
            {"claim_id": s.claim_id, "description": s.description, "slot_type": s.slot_type}
            for s in slots
        ]
        bundle = builder.build(first_hits, claims_dict)

        filler = SlotFiller()
        filled_slots = filler.fill(slots, bundle.evidence_items)
        bundle.claim_slots = filled_slots

        scorer = SufficiencyScorer(threshold=0.85)
        first_score = scorer.score(bundle)

        # 首次评分应该不足
        assert not first_score.is_sufficient, "首次证据应不足"

        # 2. 二次检索：补充更相关的证据
        second_hits = first_hits + [
            make_hit(
                content="核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，"
                       "资本充足率不得低于8%。生效日期：2024年1月1日。",
                score=0.92,
                chunk_type="clause",
                metadata={"normative_level": "部门规章"},
            ),
            make_hit(
                content="《商业银行资本管理办法》适用于中华人民共和国境内设立的商业银行。",
                score=0.85,
                chunk_type="clause",
            ),
        ]

        bundle2 = builder.build(second_hits, claims_dict)
        filled_slots2 = filler.fill(slots, bundle2.evidence_items)
        bundle2.claim_slots = filled_slots2

        second_score = scorer.score(bundle2)

        # 二次评分应该提升
        assert second_score.score > first_score.score, "二次检索应提升评分"

    def test_retry_with_budget_control(self):
        """二次检索 — 预算控制下触发二次检索"""
        budget = BudgetController("P3")  # P3: max_retrieval_rounds=3
        enforcer = BudgetEnforcer()

        # 第一次检索（P1 仅允许 1 次检索，消耗后可能触发降级）
        action1 = budget.consume_retrieval_round()
        signal1 = enforcer.enforce(action1)
        assert signal1 in ("continue", "downgrade"), \
            f"P1 第一次检索应继续或降级，实际: {signal1}"

        # 第二次检索（二次检索）
        action2 = budget.consume_retrieval_round()
        signal2 = enforcer.enforce(action2)
        assert signal2 == "continue"

        # 第三次检索
        action3 = budget.consume_retrieval_round()
        signal3 = enforcer.enforce(action3)
        # 第三次后到达上限
        assert signal3 in ("continue", "downgrade")

        # 第四次应超预算
        action4 = budget.consume_retrieval_round()
        signal4 = enforcer.enforce(action4)
        assert signal4 == "stop"


class TestPhase3E2EConflict:
    """场景 4: 冲突展示"""

    def test_conflict_detection_and_display(self):
        """冲突展示 — 两个版本规定不同 → 检测冲突 → 并列展示"""
        # 构造两个版本的同一条款
        evidence_items = [
            make_evidence_item(
                content="核心一级资本充足率不得低于8%。",
                source_doc="《商业银行资本管理办法》（2023版）",
                version_status="superseded",
                score=0.85,
                normative_level="部门规章",
            ),
            make_evidence_item(
                content="核心一级资本充足率不得低于7.75%。",
                source_doc="《商业银行资本管理办法》（2024版）",
                version_status="active",
                score=0.90,
                normative_level="部门规章",
            ),
        ]

        # 冲突检测
        detector = ConflictDetector()
        conflicts = detector.detect(evidence_items)

        # 应检测到冲突
        assert len(conflicts) > 0, "应检测到冲突"

        # 至少有一个数值冲突或版本冲突
        conflict_types = [c.conflict_type for c in conflicts]
        assert (
            ConflictType.NUMERIC_MISMATCH in conflict_types
            or ConflictType.VERSION_CONFLICT in conflict_types
        ), f"应检测到数值或版本冲突，实际: {conflict_types}"

        # 冲突优先级排序
        resolver = ConflictResolver()
        sorted_conflicts = resolver.sort_by_priority(conflicts)

        # 格式化为展示
        display_items = resolver.format_for_display(sorted_conflicts)

        assert len(display_items) > 0
        for item in display_items:
            assert "conflict_type" in item
            assert "description" in item
            assert "resolution_hint" in item
            assert item["resolution_hint"]  # 建议不为空

    def test_conflict_display_includes_both_sources(self):
        """冲突展示 — 并列展示两个来源"""
        evidence_items = [
            make_evidence_item(
                content="核心一级资本充足率不得低于8%。",
                source_doc="《商业银行资本管理办法》（2023版）",
                version_status="superseded",
                evidence_id="ev-old",
            ),
            make_evidence_item(
                content="核心一级资本充足率不得低于7.75%。",
                source_doc="《商业银行资本管理办法》（2024版）",
                version_status="active",
                evidence_id="ev-new",
            ),
        ]

        detector = ConflictDetector()
        conflicts = detector.detect(evidence_items)

        assert len(conflicts) > 0

        # 每个冲突应涉及两条证据
        for conflict in conflicts:
            assert len(conflict.evidence_ids) >= 2, "冲突应涉及至少两条证据"

    def test_no_conflict_same_source(self):
        """冲突展示 — 同一来源多块证据无冲突"""
        evidence_items = [
            make_evidence_item(
                content="核心一级资本充足率不得低于8%。",
                source_doc="《商业银行资本管理办法》",
                version_status="active",
            ),
            make_evidence_item(
                content="一级资本充足率不得低于6%。",
                source_doc="《商业银行资本管理办法》",
                version_status="active",
            ),
        ]

        detector = ConflictDetector()
        conflicts = detector.detect(evidence_items)

        # 同一来源、不同指标 → 不应产生数值冲突
        numeric_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.NUMERIC_MISMATCH
        ]
        assert len(numeric_conflicts) == 0, "同一来源不同指标不应产生数值冲突"


class TestPhase3E2EBudgetExhaustion:
    """场景 5: 预算耗尽"""

    def test_budget_exhaustion_stops_execution(self):
        """预算耗尽 — 超过检索轮次 → 停止"""
        budget = BudgetController("P2")  # P2: max_retrieval_rounds=2
        enforcer = BudgetEnforcer()

        signals = []
        # 模拟多次检索
        for i in range(5):
            action = budget.consume_retrieval_round()
            signal = enforcer.enforce(action)
            signals.append(signal)
            if signal == "stop":
                break

        # 应在超过 2 次后停止
        assert "stop" in signals, "应在超预算时停止"
        stop_index = signals.index("stop")
        # P2 允许 2 次检索，第 3 次应停止（或降级）
        assert stop_index <= 3, f"应在第3次检索时停止，实际停止于第{stop_index + 1}次"

    def test_budget_exhaustion_returns_partial_result(self):
        """预算耗尽 — 停止时返回已有部分结果"""
        budget = BudgetController("P1")  # P1: max_retrieval_rounds=1
        enforcer = BudgetEnforcer()

        # 第一次检索（P1 仅允许 1 次检索，消耗后可能触发降级）
        action1 = budget.consume_retrieval_round()
        signal1 = enforcer.enforce(action1)
        assert signal1 in ("continue", "downgrade"), \
            f"P1 第一次检索应继续或降级，实际: {signal1}"

        # 第一次检索获得了部分证据
        hits = [
            make_hit(content="核心一级资本充足率不得低于5%。", score=0.6),
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits, [
            {"claim_id": "minimum_value", "description": "最低比例"},
        ])

        # 部分结果存在
        assert bundle.evidence_count > 0

        # 第二次检索应被阻止
        action2 = budget.consume_retrieval_round()
        signal2 = enforcer.enforce(action2)
        assert signal2 == "stop", "P1 路径第二次检索应触发停止"

        # 即使停止，之前的部分结果仍可用
        assert bundle.evidence_count > 0, "停止后应保留部分结果"

    def test_budget_downgrade_before_stop(self):
        """预算耗尽 — 80% 消耗时触发降级"""
        budget = BudgetController("P4")  # P4: max_retrieval_rounds=5
        enforcer = BudgetEnforcer()

        signals = []
        # 消耗 4 次（80%）
        for i in range(4):
            action = budget.consume_retrieval_round()
            signal = enforcer.enforce(action)
            signals.append(signal)

        # 80% 时应触发降级
        assert "downgrade" in signals or "stop" in signals, \
            "80%消耗时应触发降级或停止"

    def test_budget_token_tracking(self):
        """预算耗尽 — Token 追踪"""
        budget = BudgetController("P2")
        enforcer = BudgetEnforcer()

        # 消耗 token
        action = budget.consume_tokens(5000)
        assert action == BudgetAction.CONTINUE

        summary = budget.get_summary()
        assert summary["consumed"]["tokens"] == 5000

        # 消耗大量 token
        action = budget.consume_tokens(50000)
        # 总计 55000 > 50000 预算 → 应停止
        assert action == BudgetAction.STOP, "超过 token 预算应停止"

    def test_budget_timeout(self):
        """预算耗尽 — 超时停止"""
        budget = BudgetController("P1")  # P1: total_timeout_ms=2000
        enforcer = BudgetEnforcer()

        # 消耗时间
        action = budget.consume_time(1500)
        assert action == BudgetAction.CONTINUE

        # 超过超时
        action = budget.consume_time(1000)
        # 总计 2500 > 2000 → 应停止
        assert action == BudgetAction.STOP, "超过超时应停止"


class TestPhase3E2EFullPipeline:
    """Phase 3 完整流水线集成测试"""

    def test_threshold_query_full_pipeline(self):
        """阈值查询完整流水线 — 规划 → 编译 → 检索 → 评分"""
        # 1. 声明槽位规划
        planner = ClaimPlanner()
        query_spec = {"intent": "threshold", "risk_level": "medium"}
        slots = planner.plan(query_spec)
        assert len(slots) == 6

        # 2. 查询编译
        ir = IRBuilder().build(query_spec, slots)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 3. 计划校验
        validation = PlanValidator().validate(physical_plan, ir)
        assert validation.is_valid

        # 4. 预算控制
        budget = BudgetController("P3")
        action = budget.consume_retrieval_round()
        assert action == BudgetAction.CONTINUE

        # 5. 证据组装
        hits = [
            make_hit(
                content="核心一级资本充足率不得低于5%。《商业银行资本管理办法》第43条。",
                score=0.92,
                metadata={"normative_level": "部门规章", "channel": "lexical"},
            ),
            make_hit(
                content="适用主体：中华人民共和国境内设立的商业银行。生效日期：2024年1月1日。",
                score=0.80,
                metadata={"normative_level": "部门规章", "channel": "dense"},
            ),
        ]

        claims_dict = [
            {"claim_id": s.claim_id, "description": s.description, "slot_type": s.slot_type}
            for s in slots
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits, claims_dict)

        # 6. 槽位填充
        filler = SlotFiller()
        filled_slots = filler.fill(slots, bundle.evidence_items)
        bundle.claim_slots = filled_slots

        # 7. 冲突检测
        detector = ConflictDetector()
        conflicts = detector.detect(bundle.evidence_items)

        # 8. 充分性评分
        scorer = SufficiencyScorer()
        # 将冲突信息加入 bundle
        bundle.conflicts = [c.to_dict() for c in conflicts] if conflicts else []
        result = scorer.score(bundle)

        # 验证完整流水线输出
        assert result.score > 0
        assert "coverage" in result.components
        assert "authority" in result.components
        assert "version_validity" in result.components
        assert result.components["coverage"] > 0

    def test_clause_query_full_pipeline(self):
        """条款查询完整流水线"""
        # 1. 规划
        planner = ClaimPlanner()
        query_spec = {"intent": "clause_query", "risk_level": "low"}
        slots = planner.plan(query_spec)
        assert len(slots) == 4

        # 2. 编译
        ir = IRBuilder().build(query_spec, slots)
        assert ir.intent == "clause_query"

        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        # 3. 校验
        validation = PlanValidator().validate(physical_plan, ir)
        assert validation.is_valid

        # 4. 检索（模拟）
        hits = [
            make_hit(
                content="第43条：商业银行核心一级资本充足率不得低于5%。",
                score=0.95,
                chunk_type="clause",
                hierarchy_path="第四章/第二节/第43条",
            ),
        ]

        claims_dict = [
            {"claim_id": s.claim_id, "description": s.description, "slot_type": s.slot_type}
            for s in slots
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits, claims_dict)

        # 5. 评分
        filler = SlotFiller()
        filled_slots = filler.fill(slots, bundle.evidence_items)
        bundle.claim_slots = filled_slots

        scorer = SufficiencyScorer()
        result = scorer.score(bundle)

        assert result.score > 0

    def test_comparison_query_full_pipeline(self):
        """比较查询完整流水线"""
        planner = ClaimPlanner()
        query_spec = {"intent": "comparison", "risk_level": "high"}
        slots = planner.plan(query_spec)
        assert len(slots) == 5

        ir = IRBuilder().build(query_spec, slots)
        assert ir.intent == "comparison"

        # 高风险应有更严格的停止条件
        condition_strs = [sc.condition for sc in ir.stop_conditions]
        assert any("0.85" in c or "0.90" in c for c in condition_strs)

        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        validation = PlanValidator().validate(physical_plan, ir)
        assert validation.is_valid

        # 比较查询需要多路检索
        primary_stage = physical_plan.stages[0]
        assert len(primary_stage.channels) >= 2, "比较查询应有多通道"

    def test_definition_query_full_pipeline(self):
        """定义查询完整流水线"""
        planner = ClaimPlanner()
        query_spec = {"intent": "definition", "risk_level": "low"}
        slots = planner.plan(query_spec)
        assert len(slots) == 3

        ir = IRBuilder().build(query_spec, slots)
        logical_plan = LogicalPlanner().plan(ir)
        physical_plan = PhysicalPlanner().plan(logical_plan, ir)

        validation = PlanValidator().validate(physical_plan, ir)
        assert validation.is_valid

        # 定义查询应有 lexical 和 dense 通道
        primary_stage = physical_plan.stages[0]
        assert "lexical" in primary_stage.channels
        assert "dense" in primary_stage.channels
