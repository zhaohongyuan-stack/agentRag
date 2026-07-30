"""
数值验证器 — 校验声明中的数字、单位与证据一致

职责:
  1. 从声明文本中提取数字（百分比、金额、日期等）
  2. 从证据中提取对应数字
  3. 逐一比对，确保声明中每个数字都能在证据中找到对应
  4. 检查单位一致性（% / bps / 万元等）

模式参考: evidence/conflict_detector/ 的 dataclass + to_dict 风格
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...evidence.evidence_assembler.builder import EvidenceItem


logger = logging.getLogger(__name__)


# ============================================================
# 常见单位及其别名
# ============================================================
UNIT_ALIASES: Dict[str, List[str]] = {
    "%": ["%", "百分号", "百分比"],
    "bps": ["bps", "基点", "BP", "BPs"],
    "万元": ["万元", "万人民币"],
    "亿元": ["亿元", "亿人民币"],
    "元": ["元", "人民币", "RMB", "CNY"],
}

# 单位归一化映射（别名 → 标准单位）
_UNIT_NORMALIZE: Dict[str, str] = {}
for std, aliases in UNIT_ALIASES.items():
    for alias in aliases:
        _UNIT_NORMALIZE[alias.lower()] = std


@dataclass
class ValidationResult:
    """
    验证结果

    Attributes:
        valid: 是否通过验证
        errors: 错误信息列表
        warnings: 警告信息列表（不影响 valid 判定）
        details: 详细信息（提取的数字、单位等）
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """合并两个验证结果"""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            details={**self.details, **other.details},
        )


class NumericValidator:
    """
    数值验证器

    校验声明文本中的数字与证据中的数字是否一致，并检查单位。

    用法:
        validator = NumericValidator()
        result = validator.validate("最低8%", evidence_items)
        if not result.valid:
            print(result.errors)
    """

    # 数字提取正则：匹配百分比、小数、整数、金额
    # 不使用 \w lookbehind/lookahead（中文字符会被 \w 匹配导致失败）
    NUMBER_PATTERN = re.compile(
        r"(\d+\.?\d*)"  # 数字部分（整数或小数）
        r"\s*"
        r"(%|bps|BP|基点|万元|亿元|元|万亿)?",  # 可选单位
        re.IGNORECASE,
    )

    def validate(
        self,
        claim_text: str,
        evidence: List[EvidenceItem],
        tolerance: float = 0.01,
    ) -> ValidationResult:
        """
        验证声明中的数值与证据是否一致

        Args:
            claim_text: 声明文本（如 "核心一级资本充足率不得低于8%"）
            evidence: 支持该声明的证据列表
            tolerance: 数值容差（默认 0.01，即 1%）

        Returns:
            ValidationResult
        """
        claim_numbers = self._extract_numbers(claim_text)
        evidence_numbers = self._extract_numbers_from_evidence(evidence)

        if not claim_numbers:
            # 声明中没有数字 → 无需数值校验
            return ValidationResult(
                valid=True,
                details={"claim_numbers": [], "evidence_numbers": evidence_numbers},
            )

        errors: List[str] = []
        warnings: List[str] = []
        matched: List[Dict[str, Any]] = []
        unmatched: List[Dict[str, Any]] = []

        for cn in claim_numbers:
            match = self._find_match(cn, evidence_numbers, tolerance)
            if match:
                matched.append({"claim": cn, "evidence": match})
            else:
                unmatched.append(cn)
                errors.append(
                    f"声明中的数值 {cn['value']}{cn.get('unit', '')} "
                    f"在证据中找不到对应"
                )

        # 单位一致性检查
        claim_units = {cn.get("unit", "") for cn in claim_numbers if cn.get("unit")}
        evidence_units = {
            en.get("unit", "") for en in evidence_numbers if en.get("unit")
        }

        unit_mismatch = False
        for cu in claim_units:
            normalized_cu = self._normalize_unit(cu)
            if not normalized_cu:
                continue
            has_match = any(
                self._normalize_unit(eu) == normalized_cu for eu in evidence_units
            )
            if evidence_units and not has_match:
                unit_mismatch = True
                errors.append(
                    f"单位不一致: 声明使用 '{cu}'，证据使用 {list(evidence_units)}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={
                "claim_numbers": claim_numbers,
                "evidence_numbers": evidence_numbers,
                "matched": matched,
                "unmatched": unmatched,
                "unit_mismatch": unit_mismatch,
            },
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取数字及单位

        返回 [{"value": 8.0, "unit": "%", "raw": "8%"}, ...]
        """
        results: List[Dict[str, Any]] = []
        if not text:
            return results

        for match in self.NUMBER_PATTERN.finditer(text):
            value_str = match.group(1)
            unit = match.group(2) or ""
            try:
                value = float(value_str)
            except ValueError:
                continue
            results.append(
                {
                    "value": value,
                    "unit": unit,
                    "raw": match.group(0).strip(),
                }
            )

        logger.debug("从文本提取数字: %s → %s", text[:50], results)
        return results

    def _extract_numbers_from_evidence(
        self, evidence: List[EvidenceItem]
    ) -> List[Dict[str, Any]]:
        """从证据列表中提取所有数字"""
        all_numbers: List[Dict[str, Any]] = []
        for ev in evidence:
            ev_numbers = self._extract_numbers(ev.content)
            for n in ev_numbers:
                n["evidence_id"] = ev.evidence_id
            all_numbers.extend(ev_numbers)
        return all_numbers

    @staticmethod
    def _find_match(
        claim_num: Dict[str, Any],
        evidence_numbers: List[Dict[str, Any]],
        tolerance: float,
    ) -> Optional[Dict[str, Any]]:
        """
        在证据数字中查找与声明数字匹配的项

        匹配规则:
          1. 数值相等（容差范围内）
          2. 单位相同或可归一化为同一标准单位
        """
        claim_value = claim_num["value"]
        claim_unit = claim_num.get("unit", "")
        claim_unit_norm = NumericValidator._normalize_unit(claim_unit)

        for ev_num in evidence_numbers:
            ev_value = ev_num["value"]
            ev_unit = ev_num.get("unit", "")
            ev_unit_norm = NumericValidator._normalize_unit(ev_unit)

            # 数值匹配（容差）
            if abs(claim_value - ev_value) > tolerance:
                continue

            # 单位匹配
            if claim_unit and ev_unit:
                if claim_unit_norm and ev_unit_norm:
                    if claim_unit_norm != ev_unit_norm:
                        continue
                elif claim_unit.lower() != ev_unit.lower():
                    continue

            return ev_num

        return None

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        """将单位别名归一化为标准单位"""
        if not unit:
            return ""
        return _UNIT_NORMALIZE.get(unit.lower(), unit.lower())
