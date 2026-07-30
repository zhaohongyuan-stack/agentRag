"""
综合验证器 — 声明-证据对齐，整合所有验证器

职责:
  1. 将回答拆分为声明列表（基于句子切分）
  2. 为每个声明查找支持证据
  3. 对每个声明执行数值/版本/引用/范围校验
  4. 汇总结果，决定 retry / refuse / accept

验证流程:
  validate_answer(answer, bundle)
    → 拆分回答为声明句子
    → 逐声明查找支持证据（关键词匹配）
    → 执行四项验证
    → 汇总 → AnswerValidation

模式参考: evidence/sufficiency_scorer/ 的组合验证风格
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...evidence.evidence_assembler.builder import EvidenceBundle, EvidenceItem
from ..citation_validator.validator import CitationValidator
from ..numeric_validator.validator import NumericValidator, ValidationResult
from ..scope_validator.validator import ScopeValidator
from ..version_validator.validator import VersionValidator


logger = logging.getLogger(__name__)


@dataclass
class ClaimValidation:
    """
    单个声明的验证结果

    Attributes:
        claim_text: 声明文本
        status: 验证状态（valid / invalid / unsupported）
        supporting_evidence_ids: 支持该声明的证据 ID 列表
        errors: 验证错误列表
        warnings: 验证警告列表
        details: 各验证器的详细信息
    """

    claim_text: str
    status: str = "valid"  # valid / invalid / unsupported
    supporting_evidence_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "status": self.status,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }


@dataclass
class AnswerValidation:
    """
    回答验证结果

    Attributes:
        valid: 整体是否通过验证
        action: 建议动作（accept / retry / refuse）
        claim_results: 各声明的验证结果
        unsupported_claims: 无证据支持的声明列表
        citation_result: 引用验证结果
        errors: 汇总错误
        warnings: 汇总警告
    """

    valid: bool
    action: str = "accept"  # accept / retry / refuse
    claim_results: List[ClaimValidation] = field(default_factory=list)
    unsupported_claims: List[ClaimValidation] = field(default_factory=list)
    citation_result: Optional[ValidationResult] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "action": self.action,
            "claim_results": [c.to_dict() for c in self.claim_results],
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "citation_result": self.citation_result.to_dict() if self.citation_result else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class AnswerValidator:
    """
    综合验证器

    将回答拆分为声明，逐声明验证数值、版本、引用、范围，
    并汇总结果给出 accept / retry / refuse 建议。

    用法:
        validator = AnswerValidator()
        result = validator.validate_answer(answer_text, bundle)
        if result.action == "refuse":
            ...
    """

    # 句子切分正则（按句号、问号、感叹号、分号切分）
    SENTENCE_PATTERN = re.compile(r"[。！？；;!\?\n]+")

    # 最小声明长度（太短的不作为声明）
    MIN_CLAIM_LENGTH = 5

    def __init__(
        self,
        numeric_validator: Optional[NumericValidator] = None,
        version_validator: Optional[VersionValidator] = None,
        citation_validator: Optional[CitationValidator] = None,
        scope_validator: Optional[ScopeValidator] = None,
    ):
        """
        Args:
            numeric_validator: 数值验证器，为 None 时使用默认
            version_validator: 版本验证器
            citation_validator: 引用验证器
            scope_validator: 范围验证器
        """
        self._numeric = numeric_validator or NumericValidator()
        self._version = version_validator or VersionValidator()
        self._citation = citation_validator or CitationValidator()
        self._scope = scope_validator or ScopeValidator()

    def validate_answer(
        self,
        answer: str,
        bundle: EvidenceBundle,
        query_date: str = "",
    ) -> AnswerValidation:
        """
        验证回答

        流程:
          1. 拆分回答为声明句子
          2. 逐声明查找支持证据
          3. 对每个声明执行四项验证
          4. 汇总结果

        Args:
            answer: 回答文本
            bundle: 证据包
            query_date: 查询时间点（用于版本校验）

        Returns:
            AnswerValidation
        """
        # 1. 拆分回答为声明
        claims = self._split_claims(answer)
        evidence_items = bundle.evidence_items

        logger.info(
            "回答验证开始: claims=%d, evidence=%d",
            len(claims),
            len(evidence_items),
        )

        # 2. 引用验证（整体）
        citation_result = self._citation.validate(answer, bundle)

        # 3. 逐声明验证
        claim_results: List[ClaimValidation] = []
        all_errors: List[str] = []
        all_warnings: List[str] = []
        has_unsupported = False

        for claim_text in claims:
            # 查找支持证据
            supporting = self._find_supporting_evidence(claim_text, evidence_items)

            if not supporting:
                # 无证据支持
                cv = ClaimValidation(
                    claim_text=claim_text,
                    status="unsupported",
                    errors=["无证据支持此声明"],
                )
                claim_results.append(cv)
                has_unsupported = True
                continue

            # 执行四项验证
            errors: List[str] = []
            warnings: List[str] = []
            details: Dict[str, Any] = {}

            # 数值验证
            numeric_result = self._numeric.validate(claim_text, supporting)
            details["numeric"] = numeric_result.to_dict()
            errors.extend(numeric_result.errors)
            warnings.extend(numeric_result.warnings)

            # 版本验证
            version_result = self._version.validate(
                claim_text, supporting, query_date=query_date
            )
            details["version"] = version_result.to_dict()
            errors.extend(version_result.errors)
            warnings.extend(version_result.warnings)

            # 范围验证
            scope_result = self._scope.validate(claim_text, supporting)
            details["scope"] = scope_result.to_dict()
            errors.extend(scope_result.errors)
            warnings.extend(scope_result.warnings)

            status = "invalid" if errors else "valid"

            cv = ClaimValidation(
                claim_text=claim_text,
                status=status,
                supporting_evidence_ids=[ev.evidence_id for ev in supporting],
                errors=errors,
                warnings=warnings,
                details=details,
            )
            claim_results.append(cv)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

        # 4. 决定动作
        unsupported_claims = [
            c for c in claim_results if c.status == "unsupported"
        ]
        invalid_claims = [
            c for c in claim_results if c.status == "invalid"
        ]

        # 引用错误也计入
        citation_errors = citation_result.errors
        all_errors.extend(citation_errors)

        if has_unsupported and len(unsupported_claims) > len(claims) // 2:
            # 超过一半声明无证据 → 拒答
            action = "refuse"
            valid = False
        elif unsupported_claims or invalid_claims or citation_errors:
            # 有不支持的声明或验证失败 → 重试
            action = "retry"
            valid = False
        else:
            action = "accept"
            valid = True

        result = AnswerValidation(
            valid=valid,
            action=action,
            claim_results=claim_results,
            unsupported_claims=unsupported_claims,
            citation_result=citation_result,
            errors=all_errors,
            warnings=all_warnings,
        )

        logger.info(
            "回答验证完成: valid=%s, action=%s, claims=%d, unsupported=%d",
            valid,
            action,
            len(claim_results),
            len(unsupported_claims),
        )
        return result

    # ============================================================
    # 内部方法
    # ============================================================

    def _split_claims(self, answer: str) -> List[str]:
        """
        将回答拆分为声明句子

        按句号、分号等切分，过滤过短的片段。
        """
        if not answer:
            return []

        # 去除引用标记后切分
        clean = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", answer)
        sentences = self.SENTENCE_PATTERN.split(clean)

        claims = [
            s.strip()
            for s in sentences
            if s.strip() and len(s.strip()) >= self.MIN_CLAIM_LENGTH
        ]

        logger.debug("拆分回答为声明: %d 条", len(claims))
        return claims

    def _find_supporting_evidence(
        self,
        claim_text: str,
        evidence: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        """
        为声明查找支持证据

        匹配策略:
          1. 提取声明中的关键词
          2. 检查每条证据内容是否包含这些关键词
          3. 匹配度超过阈值的证据作为支持证据

        Args:
            claim_text: 声明文本
            evidence: 证据列表

        Returns:
            支持该声明的证据列表
        """
        if not evidence:
            return []

        # 提取关键词（简化版：取长度 >= 2 的中文词组）
        keywords = self._extract_keywords(claim_text)
        if not keywords:
            # 无法提取关键词 → 返回所有证据（保守策略）
            return list(evidence)

        supporting: List[tuple] = []  # (evidence, match_score)

        for ev in evidence:
            # 计算关键词匹配度
            matched = sum(1 for kw in keywords if kw in ev.content)
            match_score = matched / len(keywords) if keywords else 0

            if match_score >= 0.3 or matched >= 2:
                supporting.append((ev, match_score))

        # 按匹配度降序排列
        supporting.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in supporting]

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """
        从文本中提取关键词

        简化版：提取 2-6 字的连续中文字符串作为关键词。
        """
        # 匹配连续中文字符
        pattern = re.compile(r"[\u4e00-\u9fa5]{2,6}")
        keywords = pattern.findall(text)

        # 去重并过滤常见无意义词
        stop_words = {"的", "了", "在", "是", "和", "与", "或", "及", "等", "为", "对", "按"}
        keywords = [
            kw for kw in keywords
            if kw not in stop_words and len(kw) >= 2
        ]

        # 去重保序
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:10]  # 最多取 10 个关键词
