"""
M5.1 声明级验证模块单元测试

测试覆盖:
  - NumericValidator: 数值一致/不一致、单位一致性
  - VersionValidator: 版本有效/失效/废止、时间点校验
  - CitationValidator: 引用存在性、来源匹配
  - ScopeValidator: 范围匹配/不匹配、范围包含
  - AnswerValidator: 综合验证、无证据声明拦截
"""

import pytest

from agent_platform.evidence.evidence_assembler.builder import (
    ClaimSlot,
    EvidenceBuilder,
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.verification import (
    AnswerValidation,
    AnswerValidator,
    CitationValidator,
    ClaimValidation,
    NumericValidator,
    ScopeValidator,
    ValidationResult,
    VersionValidator,
)


# ============================================================
# 测试数据工厂
# ============================================================

def make_evidence_item(
    evidence_id: str = "",
    chunk_id: str = "",
    content: str = "",
    score: float = 0.85,
    source_doc: str = "《商业银行资本管理办法》",
    hierarchy_path: str = "",
    chunk_type: str = "clause",
    normative_level: str = "部门规章",
    version_status: str = "active",
    citation: str = "",
    metadata: dict = None,
) -> EvidenceItem:
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


def make_bundle(evidence_items=None) -> EvidenceBundle:
    """构造 EvidenceBundle"""
    return EvidenceBundle(
        bundle_id="test-bundle",
        evidence_items=evidence_items or [],
    )


# ============================================================
# NumericValidator 测试
# ============================================================

class TestNumericValidator:
    """数值验证器测试"""

    def setup_method(self):
        self.validator = NumericValidator()

    def test_numeric_consistent(self):
        """数值一致 — 声明 '最低8%' 与证据 '不得低于8%'"""
        claim = "核心一级资本充足率最低8%"
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于8%。"),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, f"数值一致应通过: {result.errors}"
        assert len(result.details.get("matched", [])) > 0

    def test_numeric_inconsistent(self):
        """数值不一致 — 声明 '最低8.5%' 与证据 '不得低于8%'"""
        claim = "核心一级资本充足率最低8.5%"
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于8%。"),
        ]
        result = self.validator.validate(claim, evidence)

        assert not result.valid, "数值不一致应失败"
        assert len(result.errors) > 0

    def test_unit_mismatch(self):
        """单位不一致 — 声明 '8bps' 与证据 '8%'"""
        claim = "核心一级资本充足率最低8bps"
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于8%。"),
        ]
        result = self.validator.validate(claim, evidence)

        # bps 和 % 是不同单位
        assert not result.valid or result.details.get("unit_mismatch")

    def test_no_numbers_in_claim(self):
        """声明中无数值 — 跳过校验"""
        claim = "商业银行应满足资本充足率要求"
        evidence = [
            make_evidence_item(content="资本充足率不得低于8%。"),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, "声明中无数值应跳过校验"

    def test_multiple_numbers_all_match(self):
        """多数值全部匹配"""
        claim = "核心一级资本充足率5%，一级资本充足率6%，资本充足率8%"
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%，一级资本充足率不得低于6%，"
                       "资本充足率不得低于8%。"
            ),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, f"多数值匹配应通过: {result.errors}"

    def test_multiple_numbers_partial_mismatch(self):
        """多数值部分不匹配"""
        claim = "核心一级资本充足率5%，一级资本充足率7%"
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%，一级资本充足率不得低于6%。"
            ),
        ]
        result = self.validator.validate(claim, evidence)

        assert not result.valid, "部分数值不匹配应失败"

    def test_tolerance(self):
        """数值容差 — 8.0 与 8.01 在容差范围内"""
        claim = "资本充足率最低8.0%"
        evidence = [
            make_evidence_item(content="资本充足率不得低于8.01%。"),
        ]
        result = self.validator.validate(claim, evidence, tolerance=0.02)

        assert result.valid, "容差范围内应通过"

    def test_empty_evidence(self):
        """空证据列表"""
        claim = "资本充足率最低8%"
        result = self.validator.validate(claim, [])

        assert not result.valid, "有数值声明但无证据应失败"


# ============================================================
# VersionValidator 测试
# ============================================================

class TestVersionValidator:
    """版本验证器测试"""

    def setup_method(self):
        self.validator = VersionValidator()

    def test_active_version(self):
        """有效版本 — version_status=active"""
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。生效日期：2024年1月1日。",
                version_status="active",
            ),
        ]
        result = self.validator.validate("测试声明", evidence)

        assert result.valid, "active 版本应通过"

    def test_superseded_version_warning(self):
        """已替代版本 — 产生警告但通过"""
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于8%。",
                source_doc="《商业银行资本管理办法》（2023版）",
                version_status="superseded",
            ),
        ]
        result = self.validator.validate("测试声明", evidence)

        # superseded 产生 warning 但 valid=True（仍可参考）
        assert len(result.warnings) > 0, "已替代版本应产生警告"

    def test_repealed_version_invalid(self):
        """已废止版本 — 验证失败"""
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于8%。",
                source_doc="旧版资本管理办法",
                version_status="repealed",
            ),
        ]
        result = self.validator.validate("测试声明", evidence)

        assert not result.valid, "已废止版本应验证失败"
        assert len(result.errors) > 0

    def test_not_yet_effective(self):
        """未生效 — 生效日期晚于查询时间点"""
        evidence = [
            make_evidence_item(
                content="新规生效日期：2026年12月1日。",
                version_status="active",
                metadata={"effective_date": "2026-12-01"},
            ),
        ]
        result = self.validator.validate("测试声明", evidence, query_date="2026-01-01")

        assert not result.valid, "未生效版本应验证失败"

    def test_effective_within_query_date(self):
        """已生效 — 生效日期早于查询时间点"""
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。2024年1月1日生效。",
                version_status="active",
                metadata={"effective_date": "2024-01-01"},
            ),
        ]
        result = self.validator.validate("测试声明", evidence, query_date="2026-01-01")

        assert result.valid, "已生效版本应通过"

    def test_empty_evidence(self):
        """空证据 — 跳过校验"""
        result = self.validator.validate("测试声明", [])
        assert result.valid

    def test_unknown_version_status(self):
        """未知版本状态 — 产生警告"""
        evidence = [
            make_evidence_item(
                content="测试内容",
                version_status="unknown_status",
            ),
        ]
        result = self.validator.validate("测试声明", evidence)

        assert len(result.warnings) > 0, "未知版本状态应产生警告"


# ============================================================
# CitationValidator 测试
# ============================================================

class TestCitationValidator:
    """引用验证器测试"""

    def setup_method(self):
        self.validator = CitationValidator()

    def test_valid_numeric_citation(self):
        """有效数字引用 — [1] 对应第 1 条证据"""
        answer = "核心一级资本充足率不得低于5%[1]。"
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于5%。"),
        ]
        bundle = make_bundle(evidence)
        result = self.validator.validate(answer, bundle)

        assert result.valid, f"有效引用应通过: {result.errors}"

    def test_citation_out_of_range(self):
        """引用超出范围 — [3] 但只有 2 条证据"""
        answer = "核心一级资本充足率不得低于5%[3]。"
        evidence = [
            make_evidence_item(content="证据1"),
            make_evidence_item(content="证据2"),
        ]
        bundle = make_bundle(evidence)
        result = self.validator.validate(answer, bundle)

        assert not result.valid, "引用超出范围应失败"

    def test_source_citation_match(self):
        """来源引用匹配"""
        answer = "核心一级资本充足率不得低于5%（来源：《商业银行资本管理办法》）。"
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。",
                source_doc="《商业银行资本管理办法》",
            ),
        ]
        bundle = make_bundle(evidence)
        result = self.validator.validate(answer, bundle)

        assert result.valid

    def test_no_citation_with_evidence_warning(self):
        """有证据但无引用 — 产生警告"""
        answer = "核心一级资本充足率不得低于5%。"
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于5%。"),
        ]
        bundle = make_bundle(evidence)
        result = self.validator.validate(answer, bundle)

        assert len(result.warnings) > 0, "有证据但无引用应产生警告"

    def test_citation_with_no_evidence_error(self):
        """有引用但无证据 — 报错"""
        answer = "核心一级资本充足率不得低于5%[1]。"
        bundle = make_bundle([])
        result = self.validator.validate(answer, bundle)

        assert not result.valid, "有引用但无证据应失败"

    def test_multiple_citations(self):
        """多引用 [1,2]"""
        answer = "资本充足率要求包括核心一级5%[1]和总资本8%[2]。"
        evidence = [
            make_evidence_item(content="核心一级资本充足率5%。"),
            make_evidence_item(content="资本充足率8%。"),
        ]
        bundle = make_bundle(evidence)
        result = self.validator.validate(answer, bundle)

        assert result.valid

    def test_range_citation(self):
        """范围引用 [1-3]"""
        answer = "参见相关规定[1-3]。"
        evidence = [
            make_evidence_item(content="证据1"),
            make_evidence_item(content="证据2"),
            make_evidence_item(content="证据3"),
        ]
        bundle = make_bundle(evidence)
        result = self.validator.validate(answer, bundle)

        assert result.valid, f"范围引用应通过: {result.errors}"

    def test_no_citation_no_evidence(self):
        """无引用无证据 — 跳过"""
        answer = "这是一个简单回答。"
        bundle = make_bundle([])
        result = self.validator.validate(answer, bundle)

        assert result.valid


# ============================================================
# ScopeValidator 测试
# ============================================================

class TestScopeValidator:
    """范围验证器测试"""

    def setup_method(self):
        self.validator = ScopeValidator()

    def test_scope_match(self):
        """范围匹配 — 声明和证据都适用于所有银行"""
        claim = "该规定适用于所有银行"
        evidence = [
            make_evidence_item(
                content="本规定适用于所有商业银行。",
            ),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, "范围匹配应通过"

    def test_scope_mismatch(self):
        """范围不匹配 — 声明适用于所有银行，证据仅适用于系统重要性银行"""
        claim = "该规定适用于所有银行"
        evidence = [
            make_evidence_item(
                content="本规定适用于系统重要性银行。",
            ),
        ]
        result = self.validator.validate(claim, evidence)

        assert not result.valid, "范围不匹配应失败"
        assert len(result.errors) > 0

    def test_scope_covered_by_all_banks(self):
        """范围被 all_banks 覆盖 — 证据适用于所有银行，声明适用于系统重要性"""
        claim = "该规定适用于系统重要性银行"
        evidence = [
            make_evidence_item(
                content="本规定适用于所有商业银行。",
            ),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, "all_banks 应覆盖系统重要性银行"

    def test_no_scope_in_claim(self):
        """声明中无范围 — 跳过"""
        claim = "资本充足率不得低于8%"
        evidence = [
            make_evidence_item(content="资本充足率不得低于8%。"),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, "无范围声明应跳过"

    def test_no_scope_in_evidence_warning(self):
        """证据中无范围 — 警告但通过"""
        claim = "该规定适用于所有银行"
        evidence = [
            make_evidence_item(content="资本充足率不得低于8%。"),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid, "证据无范围应通过但有警告"
        assert len(result.warnings) > 0

    def test_non_systemically_important_scope(self):
        """非系统重要性银行范围"""
        claim = "该规定适用于非系统重要性银行"
        evidence = [
            make_evidence_item(
                content="本规定适用于非系统重要性银行。",
            ),
        ]
        result = self.validator.validate(claim, evidence)

        assert result.valid


# ============================================================
# AnswerValidator 测试
# ============================================================

class TestAnswerValidator:
    """综合验证器测试"""

    def setup_method(self):
        self.validator = AnswerValidator()

    def test_all_claims_valid(self):
        """全部声明通过 — accept"""
        answer = (
            "核心一级资本充足率不得低于5%[1]。"
            "该规定适用于所有商业银行[2]。"
        )
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。适用于所有商业银行。",
                version_status="active",
            ),
            make_evidence_item(
                content="本规定适用于中华人民共和国境内设立的商业银行。",
                version_status="active",
            ),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert result.action == "accept", f"全部通过应 accept: {result.errors}"
        assert result.valid

    def test_unsupported_claim_triggers_retry(self):
        """无证据声明 — retry"""
        answer = (
            "核心一级资本充足率不得低于5%[1]。"
            "另外某银行2024年利润为1000亿元。"
        )
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于5%。"),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert result.action == "retry", "有无证据声明应 retry"
        assert len(result.unsupported_claims) > 0

    def test_mostly_unsupported_triggers_refuse(self):
        """超过一半声明无证据 — refuse"""
        answer = (
            "银行A利润100亿。"
            "银行B资产5000亿。"
            "银行C资本充足率10%。"
            "银行D不良率2%。"
            "核心一级资本充足率不得低于5%[1]。"
        )
        evidence = [
            make_evidence_item(content="核心一级资本充足率不得低于5%。"),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert result.action == "refuse", "超过一半无证据应 refuse"

    def test_numeric_mismatch_triggers_retry(self):
        """数值不匹配 — retry"""
        answer = "核心一级资本充足率不得低于8.5%[1]。"
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。",
                version_status="active",
            ),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert result.action == "retry", "数值不匹配应 retry"
        assert not result.valid

    def test_repealed_version_triggers_retry(self):
        """引用已废止版本 — retry"""
        answer = "资本充足率不得低于8%[1]。"
        evidence = [
            make_evidence_item(
                content="资本充足率不得低于8%。",
                source_doc="旧版管理办法",
                version_status="repealed",
            ),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert not result.valid, "引用已废止版本应失败"
        assert result.action == "retry"

    def test_citation_out_of_range(self):
        """引用超出范围 — retry"""
        answer = "资本充足率不得低于8%[5]。"
        evidence = [
            make_evidence_item(content="资本充足率不得低于8%。"),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert not result.valid, "引用超出范围应失败"

    def test_scope_mismatch_triggers_retry(self):
        """范围不匹配 — retry"""
        answer = "该规定适用于所有银行[1]。"
        evidence = [
            make_evidence_item(
                content="本规定适用于系统重要性银行。",
                version_status="active",
            ),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert not result.valid, "范围不匹配应失败"

    def test_empty_answer(self):
        """空回答 — accept（无声明可验证）"""
        result = self.validator.validate_answer("", make_bundle([]))

        assert result.valid
        assert result.action == "accept"

    def test_claim_results_structure(self):
        """声明验证结果结构完整"""
        answer = "核心一级资本充足率不得低于5%[1]。"
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。",
                version_status="active",
            ),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert len(result.claim_results) > 0
        for cr in result.claim_results:
            assert isinstance(cr, ClaimValidation)
            assert cr.status in ("valid", "invalid", "unsupported")
            assert hasattr(cr, "errors")
            assert hasattr(cr, "warnings")
            assert hasattr(cr, "details")

    def test_citation_result_included(self):
        """引用验证结果包含在整体结果中"""
        answer = "资本充足率8%[1]。"
        evidence = [
            make_evidence_item(content="资本充足率不得低于8%。"),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)

        assert result.citation_result is not None
        assert isinstance(result.citation_result, ValidationResult)

    def test_to_dict_serialization(self):
        """结果可序列化为字典"""
        answer = "核心一级资本充足率不得低于5%[1]。"
        evidence = [
            make_evidence_item(
                content="核心一级资本充足率不得低于5%。",
                version_status="active",
            ),
        ]
        bundle = make_bundle(evidence)

        result = self.validator.validate_answer(answer, bundle)
        d = result.to_dict()

        assert "valid" in d
        assert "action" in d
        assert "claim_results" in d
        assert isinstance(d["claim_results"], list)


# ============================================================
# ValidationResult 辅助测试
# ============================================================

class TestValidationResult:
    """验证结果数据结构测试"""

    def test_merge(self):
        """合并两个验证结果"""
        r1 = ValidationResult(valid=True, errors=["e1"], warnings=["w1"])
        r2 = ValidationResult(valid=False, errors=["e2"], warnings=["w2"])

        merged = r1.merge(r2)

        assert not merged.valid  # r2.valid=False → 合并为 False
        assert "e1" in merged.errors
        assert "e2" in merged.errors
        assert "w1" in merged.warnings
        assert "w2" in merged.warnings

    def test_to_dict(self):
        """序列化为字典"""
        r = ValidationResult(
            valid=True,
            errors=["error1"],
            warnings=["warning1"],
            details={"key": "value"},
        )
        d = r.to_dict()

        assert d["valid"] is True
        assert d["errors"] == ["error1"]
        assert d["warnings"] == ["warning1"]
        assert d["details"] == {"key": "value"}
