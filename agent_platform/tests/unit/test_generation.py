"""
回答生成模块单元测试 — Phase 2 (M2.4)

测试覆盖:
  - TemplateGenerator 向后兼容: 条款查询带证据 → 回答含引用
  - TemplateGenerator: 无证据 → 拒答
  - GroundedGenerator Mock 模式: 条款查询带证据 → 生成回答
  - GroundedGenerator Mock 模式: 无证据 → 拒答
  - CitationFormatter: 条款引用格式正确
  - CitationFormatter: 表格引用格式正确（附件X 表Y 第Z行）
  - CitationFormatter: 编号列表正确
  - AnswerPlanner: clause_query 结构有 3 个章节
  - AnswerPlanner: threshold 结构有 5 个章节
  - AnswerPlanner: 证据按评分排序
"""

import pytest

from agent_platform.evidence.evidence_assembler import (
    ClaimSlot,
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.generation.answer_planner import AnswerPlan, AnswerPlanner
from agent_platform.generation.citation_formatter import CitationFormatter
from agent_platform.generation.grounded_generator import (
    GeneratedAnswer,
    GroundedGenerator,
    TemplateGenerator,
)
from agent_platform.runtime.llm_client import LLMClient


# ============================================================
# 测试辅助工厂
# ============================================================

def make_evidence_item(
    evidence_id="ev-1",
    chunk_id="chunk-1",
    content="商业银行核心一级资本充足率不得低于5%。",
    evidence_snippet="核心一级资本充足率不得低于5%",
    citation="《商业银行资本管理办法》第43条",
    score=0.9,
    source_doc="商业银行资本管理办法",
    hierarchy_path="第三章/第二节",
    chunk_type="clause",
    normative_level="neutral",
    metadata=None,
) -> EvidenceItem:
    """构造测试用证据项"""
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        content=content,
        evidence_snippet=evidence_snippet,
        citation=citation,
        score=score,
        source_doc=source_doc,
        hierarchy_path=hierarchy_path,
        chunk_type=chunk_type,
        normative_level=normative_level,
        version_status="active",
        metadata=metadata or {},
    )


def make_evidence_bundle(
    evidence_items=None,
    is_sufficient=True,
    sufficiency_score=0.9,
    claim_slots=None,
) -> EvidenceBundle:
    """构造测试用证据包"""
    return EvidenceBundle(
        bundle_id="eb-test",
        claim_slots=claim_slots or [],
        evidence_items=evidence_items if evidence_items is not None else [],
        sufficiency_score=sufficiency_score,
        sufficiency_threshold=0.85,
        is_sufficient=is_sufficient,
        conflicts=[],
        missing_conditions=[],
    )


def make_grounded_generator() -> GroundedGenerator:
    """构造 Mock 模式的 GroundedGenerator（不依赖环境变量/API Key）"""
    client = LLMClient(mock=True)
    return GroundedGenerator(llm_client=client)


# ============================================================
# TemplateGenerator 向后兼容测试
# ============================================================

class TestTemplateGenerator:
    """模板生成器测试（向后兼容）"""

    def test_clause_query_with_evidence_contains_citation(self):
        """条款查询带证据 → 回答中包含引用"""
        gen = TemplateGenerator()
        ev = make_evidence_item()
        bundle = make_evidence_bundle([ev])

        result = gen.generate("clause_query", bundle, "《商业银行资本管理办法》第43条")

        assert isinstance(result, GeneratedAnswer)
        assert not result.is_refusal
        assert result.answer_text  # 非空
        # 回答文本中应包含引用标记
        assert "第43条" in result.answer_text
        # 引用列表非空
        assert len(result.citations) >= 1
        assert any("第43条" in c["citation"] for c in result.citations)

    def test_no_evidence_returns_refusal(self):
        """无证据 → 拒答"""
        gen = TemplateGenerator()
        bundle = make_evidence_bundle(
            [], is_sufficient=False, sufficiency_score=0.0
        )

        result = gen.generate("clause_query", bundle, "测试问题")

        assert result.is_refusal is True
        assert result.refusal_reason == "证据不足"
        assert result.confidence == 0.0


# ============================================================
# GroundedGenerator Mock 模式测试
# ============================================================

class TestGroundedGeneratorMock:
    """基于 LLM 的生成器测试（Mock 模式回退到模板）"""

    def test_mock_mode_clause_query_generates_answer(self):
        """Mock 模式下条款查询带证据 → 生成回答"""
        gen = make_grounded_generator()
        assert gen.is_mock is True

        ev = make_evidence_item()
        bundle = make_evidence_bundle([ev])

        result = gen.generate(
            "clause_query", bundle, "《商业银行资本管理办法》第43条"
        )

        assert isinstance(result, GeneratedAnswer)
        assert not result.is_refusal
        assert result.answer_text  # 非空
        # 回退到模板，引用应存在
        assert len(result.citations) >= 1

    def test_mock_mode_no_evidence_returns_refusal(self):
        """Mock 模式下无证据 → 拒答"""
        gen = make_grounded_generator()

        bundle = make_evidence_bundle(
            [], is_sufficient=False, sufficiency_score=0.0
        )

        result = gen.generate("clause_query", bundle, "测试问题")

        assert result.is_refusal is True
        assert result.refusal_reason == "证据不足"

    def test_grounded_generator_falls_back_on_llm_failure(self):
        """LLM 调用失败 → 回退到模板生成器"""
        class FailingClient:
            is_mock = False

            def chat_json(self, messages, temperature=0.1):
                raise RuntimeError("模拟 LLM 调用失败")

        gen = GroundedGenerator(llm_client=FailingClient())
        ev = make_evidence_item()
        bundle = make_evidence_bundle([ev])

        result = gen.generate("clause_query", bundle, "第43条")

        # 应回退到模板并成功生成
        assert not result.is_refusal
        assert result.answer_text

    def test_generate_clarification(self):
        """生成澄清请求"""
        gen = make_grounded_generator()
        ambiguities = [
            {"description": "未明确适用主体", "resolution": "请指明银行类型"}
        ]

        result = gen.generate_clarification(ambiguities)

        assert isinstance(result, GeneratedAnswer)
        assert "澄清" in result.answer_text or "澄清" in result.answer_text


# ============================================================
# CitationFormatter 测试
# ============================================================

class TestCitationFormatter:
    """引用格式化器测试"""

    def test_clause_citation_format(self):
        """条款引用格式: 《文件名》第X条"""
        fmt = CitationFormatter()
        ev = make_evidence_item(
            chunk_type="clause",
            source_doc="商业银行资本管理办法",
            citation="《商业银行资本管理办法》第43条",
            metadata={"clause_number": "43"},
        )

        citation = fmt.format_citation(ev)
        assert citation == "《商业银行资本管理办法》第43条"

    def test_table_citation_format(self):
        """表格引用格式: 附件X 表Y 第Z行"""
        fmt = CitationFormatter()
        ev = EvidenceItem(
            evidence_id="ev-tbl",
            chunk_id="chunk-tbl",
            content="表格内容",
            evidence_snippet="风险权重表",
            citation="附件1 表2 第3行",
            score=0.88,
            source_doc="资本监管附件",
            hierarchy_path="附件1/表2",
            chunk_type="table",
            normative_level="",
            version_status="active",
            metadata={
                "attachment_number": "1",
                "table_number": "2",
                "row_number": "3",
            },
        )

        citation = fmt.format_citation(ev)
        assert citation == "附件1 表2 第3行"

    def test_definition_citation_format(self):
        """定义引用格式: 《文件名》第X条"""
        fmt = CitationFormatter()
        ev = make_evidence_item(
            chunk_type="definition",
            source_doc="商业银行资本管理办法",
            citation="《商业银行资本管理办法》第5条",
            metadata={"clause_number": "5"},
        )

        citation = fmt.format_citation(ev)
        assert citation == "《商业银行资本管理办法》第5条"

    def test_numbered_citation_list(self):
        """编号引用列表正确"""
        fmt = CitationFormatter()
        ev1 = make_evidence_item(
            evidence_id="ev-1",
            chunk_id="chunk-1",
            source_doc="商业银行资本管理办法",
            citation="《商业银行资本管理办法》第43条",
            metadata={"clause_number": "43"},
        )
        ev2 = make_evidence_item(
            evidence_id="ev-2",
            chunk_id="chunk-2",
            source_doc="商业银行杠杆率管理办法",
            citation="《商业银行杠杆率管理办法》第8条",
            metadata={"clause_number": "8"},
        )

        result = fmt.format_citation_list([ev1, ev2])

        assert len(result) == 2
        assert result[0]["index"] == "1"
        assert result[1]["index"] == "2"
        assert result[0]["citation"] == "《商业银行资本管理办法》第43条"
        assert result[1]["citation"] == "《商业银行杠杆率管理办法》第8条"
        assert result[0]["source_doc"] == "商业银行资本管理办法"
        assert result[0]["chunk_id"] == "chunk-1"

    def test_numbered_list_dedup(self):
        """编号引用列表去重"""
        fmt = CitationFormatter()
        ev1 = make_evidence_item(
            evidence_id="ev-1",
            chunk_id="chunk-1",
            metadata={"clause_number": "43"},
        )
        ev2 = make_evidence_item(
            evidence_id="ev-2",
            chunk_id="chunk-2",
            metadata={"clause_number": "43"},
        )

        result = fmt.format_citation_list([ev1, ev2])

        # 相同引用去重，只保留一条
        assert len(result) == 1
        assert result[0]["index"] == "1"

    def test_inline_citation(self):
        """行内引用标记"""
        fmt = CitationFormatter()
        assert fmt.format_inline_citation(1) == "[1]"
        assert fmt.format_inline_citation(3) == "[3]"


# ============================================================
# AnswerPlanner 测试
# ============================================================

class TestAnswerPlanner:
    """回答规划器测试"""

    def test_clause_query_structure_has_3_sections(self):
        """clause_query 结构有 3 个章节"""
        planner = AnswerPlanner()
        bundle = make_evidence_bundle([make_evidence_item()])

        plan = planner.plan("clause_query", bundle)

        assert isinstance(plan, AnswerPlan)
        assert len(plan.structure) == 3
        assert plan.structure == ["法条原文", "适用范围", "规范强度"]
        assert plan.answer_shape == "text_excerpt"

    def test_threshold_structure_has_5_sections(self):
        """threshold 结构有 5 个章节"""
        planner = AnswerPlanner()
        bundle = make_evidence_bundle([make_evidence_item()])

        plan = planner.plan("threshold", bundle)

        assert len(plan.structure) == 5
        assert plan.structure == [
            "指标名称",
            "最低/最高要求",
            "适用主体",
            "生效时间",
            "法规依据",
        ]
        assert plan.answer_shape == "text_excerpt"

    def test_definition_structure(self):
        """definition 结构正确"""
        planner = AnswerPlanner()
        bundle = make_evidence_bundle([make_evidence_item()])

        plan = planner.plan("definition", bundle)

        assert plan.structure == ["定义内容", "适用范围", "来源"]
        assert plan.answer_shape == "text_excerpt"

    def test_table_lookup_structure_and_shape(self):
        """table_lookup 结构与回答形态"""
        planner = AnswerPlanner()
        bundle = make_evidence_bundle([make_evidence_item()])

        plan = planner.plan("table_lookup", bundle)

        assert plan.structure == ["查询结果", "数据来源"]
        assert plan.answer_shape == "table_data"

    def test_comparison_answer_shape(self):
        """comparison 回答形态"""
        planner = AnswerPlanner()
        bundle = make_evidence_bundle([make_evidence_item()])

        plan = planner.plan("comparison", bundle)

        assert plan.answer_shape == "comparison"

    def test_compliance_answer_shape(self):
        """compliance 回答形态"""
        planner = AnswerPlanner()
        bundle = make_evidence_bundle([make_evidence_item()])

        plan = planner.plan("compliance", bundle)

        assert plan.answer_shape == "compliance_judgment"

    def test_evidence_ordered_by_score(self):
        """证据按评分降序排列"""
        planner = AnswerPlanner()
        ev_low = make_evidence_item(evidence_id="ev-low", score=0.3)
        ev_high = make_evidence_item(evidence_id="ev-high", score=0.95)
        ev_mid = make_evidence_item(evidence_id="ev-mid", score=0.6)
        bundle = make_evidence_bundle([ev_low, ev_high, ev_mid])

        plan = planner.plan("clause_query", bundle)

        # 评分高的排前面
        assert plan.evidence_order == ["ev-high", "ev-mid", "ev-low"]
        assert plan.required_citations == 3

    def test_required_citations_capped_at_5(self):
        """所需引用数量上限为 5"""
        planner = AnswerPlanner()
        items = [
            make_evidence_item(evidence_id=f"ev-{i}", score=0.9 - i * 0.01)
            for i in range(8)
        ]
        bundle = make_evidence_bundle(items)

        plan = planner.plan("clause_query", bundle)

        assert len(plan.evidence_order) == 8
        assert plan.required_citations == 5  # 上限 5

    def test_plan_with_no_evidence(self):
        """无证据时规划仍可执行"""
        planner = AnswerPlanner()
        plan = planner.plan("clause_query", None)

        assert plan.structure == ["法条原文", "适用范围", "规范强度"]
        assert plan.evidence_order == []
        assert plan.required_citations == 0

    def test_answer_plan_to_dict(self):
        """AnswerPlan 序列化为字典"""
        plan = AnswerPlan(
            intent="clause_query",
            structure=["法条原文", "适用范围", "规范强度"],
            evidence_order=["ev-1", "ev-2"],
            required_citations=2,
            answer_shape="text_excerpt",
        )

        d = plan.to_dict()

        assert d["intent"] == "clause_query"
        assert d["structure"] == ["法条原文", "适用范围", "规范强度"]
        assert d["evidence_order"] == ["ev-1", "ev-2"]
        assert d["required_citations"] == 2
        assert d["answer_shape"] == "text_excerpt"
