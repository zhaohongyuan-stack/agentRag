"""
范围验证器 — 校验声明的适用主体和范围与证据一致

职责:
  1. 从声明中提取适用主体（如 "所有银行" "系统重要性银行"）
  2. 从证据中提取适用范围
  3. 检查声明范围是否被证据范围覆盖

模式参考: evidence/conflict_detector/ 的 dataclass + to_dict 风格
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...evidence.evidence_assembler.builder import EvidenceItem
from ..numeric_validator.validator import ValidationResult


logger = logging.getLogger(__name__)


# 适用主体关键词映射
SCOPE_KEYWORDS: Dict[str, List[str]] = {
    "all_banks": ["所有银行", "全部银行", "各类银行", "商业银行"],
    "systemically_important": ["系统重要性银行", "系统重要性机构", "D-SIB"],
    "non_systemically_important": ["非系统重要性银行", "非系统重要性机构"],
    "state_owned": ["国有大型银行", "国有银行"],
    "joint_stock": ["股份制银行", "股份制商业银行"],
    "city_commercial": ["城商行", "城市商业银行"],
    "rural": ["农商行", "农村商业银行", "农村信用社"],
    "foreign": ["外资银行", "外国银行"],
}

# 范围包含关系（前者的范围包含后者）
SCOPE_CONTAINMENT: Dict[str, List[str]] = {
    "all_banks": [
        "systemically_important",
        "non_systemically_important",
        "state_owned",
        "joint_stock",
        "city_commercial",
        "rural",
        "foreign",
    ],
    "state_owned": [],  # 不包含其他子类
    "systemically_important": [],
    "non_systemically_important": [],
}


class ScopeValidator:
    """
    范围验证器

    校验声明的适用范围是否与证据中的适用范围一致。

    用法:
        validator = ScopeValidator()
        result = validator.validate(claim_text, evidence)
    """

    def validate(
        self,
        claim_text: str,
        evidence: List[EvidenceItem],
    ) -> ValidationResult:
        """
        验证声明范围与证据范围是否一致

        Args:
            claim_text: 声明文本
            evidence: 支持该声明的证据列表

        Returns:
            ValidationResult
        """
        claim_scopes = self._extract_scopes(claim_text)
        evidence_scopes = self._extract_scopes_from_evidence(evidence)

        if not claim_scopes:
            # 声明中无明确范围 → 跳过
            return ValidationResult(
                valid=True,
                details={"reason": "声明中无明确适用范围，跳过范围校验"},
            )

        errors: List[str] = []
        warnings: List[str] = []

        if not evidence_scopes:
            warnings.append("证据中无明确适用范围信息")
            return ValidationResult(
                valid=True,
                errors=errors,
                warnings=warnings,
                details={
                    "claim_scopes": claim_scopes,
                    "evidence_scopes": [],
                },
            )

        # 检查声明范围是否被证据范围覆盖
        for claim_scope in claim_scopes:
            covered = False
            for ev_scope in evidence_scopes:
                if self._is_covered(claim_scope, ev_scope):
                    covered = True
                    break

            if not covered:
                # 检查是否范围不匹配
                claim_label = self._scope_label(claim_scope)
                ev_labels = [self._scope_label(s) for s in evidence_scopes]
                errors.append(
                    f"范围不匹配: 声明适用于 '{claim_label}'，"
                    f"但证据适用于 {ev_labels}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={
                "claim_scopes": claim_scopes,
                "evidence_scopes": evidence_scopes,
            },
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _extract_scopes(self, text: str) -> List[str]:
        """
        从文本中提取适用范围

        返回范围 key 列表（如 ["all_banks", "systemically_important"]）
        """
        if not text:
            return []

        scopes: List[str] = []
        text_lower = text.lower()

        for scope_key, keywords in SCOPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    if scope_key not in scopes:
                        scopes.append(scope_key)
                    break

        # 特殊处理："非系统重要性" 需要在 "系统重要性" 之前检查
        if "非系统重要性" in text and "non_systemically_important" not in scopes:
            # 如果同时有 "系统重要性银行" 但没有 "非" 前缀，需要区分
            # 移除 systemically_important 如果实际是 non_systemically_important
            if "non_systemically_important" in scopes:
                if "systemically_important" in scopes:
                    scopes.remove("systemically_important")

        logger.debug("提取范围: '%s' → %s", text[:50], scopes)
        return scopes

    def _extract_scopes_from_evidence(
        self, evidence: List[EvidenceItem]
    ) -> List[str]:
        """从证据列表中提取所有适用范围"""
        all_scopes: List[str] = []
        for ev in evidence:
            scopes = self._extract_scopes(ev.content)
            # 也检查 metadata 中的 scope 信息
            meta = ev.metadata or {}
            if "scope" in meta:
                meta_scope = meta["scope"]
                if isinstance(meta_scope, str):
                    meta_scopes = self._extract_scopes(meta_scope)
                    for s in meta_scopes:
                        if s not in all_scopes:
                            all_scopes.append(s)
            for s in scopes:
                if s not in all_scopes:
                    all_scopes.append(s)
        return all_scopes

    @staticmethod
    def _is_covered(claim_scope: str, evidence_scope: str) -> bool:
        """
        检查声明范围是否被证据范围覆盖

        规则:
          1. 范围相同 → 覆盖
          2. 证据范围包含声明范围 → 覆盖
          3. 声明范围更宽 → 不覆盖
        """
        if claim_scope == evidence_scope:
            return True

        # 检查证据范围是否包含声明范围
        contained = SCOPE_CONTAINMENT.get(evidence_scope, [])
        if claim_scope in contained:
            return True

        # all_banks 覆盖一切
        if evidence_scope == "all_banks":
            return True

        return False

    @staticmethod
    def _scope_label(scope_key: str) -> str:
        """获取范围的可读标签"""
        for key, keywords in SCOPE_KEYWORDS.items():
            if key == scope_key:
                return keywords[0]
        return scope_key
