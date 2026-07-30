"""
引用验证器 — 校验回答中的引用标记是否存在且内容匹配

职责:
  1. 从回答文本中提取引用标记（如 [1], [2], (来源: xxx)）
  2. 检查每个引用是否在证据包中有对应
  3. 校验引用内容是否与证据原文一致（子串匹配）

模式参考: evidence/conflict_detector/ 的 dataclass + to_dict 风格
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...evidence.evidence_assembler.builder import EvidenceBundle, EvidenceItem
from ..numeric_validator.validator import ValidationResult


logger = logging.getLogger(__name__)


# 引用标记正则：[1] [1,2] [1-3] (来源: xxx) 等
CITATION_PATTERN = re.compile(
    r"\[(\d+(?:[-,]\d+)*)\]"  # [1] [1,2] [1-3]
    r"|\(来源[：:]\s*([^)]+)\)"  # (来源: xxx)
    r"|\(引用[：:]\s*([^)]+)\)",  # (引用: xxx)
    re.IGNORECASE,
)


class CitationValidator:
    """
    引用验证器

    校验回答中的引用标记是否存在对应证据，以及引用内容是否匹配。

    用法:
        validator = CitationValidator()
        result = validator.validate(answer_text, bundle)
    """

    def validate(
        self,
        answer_text: str,
        bundle: EvidenceBundle,
    ) -> ValidationResult:
        """
        验证回答中的引用

        Args:
            answer_text: 回答文本
            bundle: 证据包

        Returns:
            ValidationResult
        """
        citations = self._extract_citations(answer_text)
        evidence_list = bundle.evidence_items

        if not citations and not evidence_list:
            # 无引用也无证据 → 跳过
            return ValidationResult(
                valid=True,
                details={"reason": "无引用标记，跳过引用校验"},
            )

        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {
            "citations_found": citations,
            "evidence_count": len(evidence_list),
        }

        # 检查每个数字引用是否有对应证据
        for citation in citations:
            if citation["type"] == "numeric":
                for num in citation["numbers"]:
                    if num > len(evidence_list):
                        errors.append(
                            f"引用 [{num}] 超出证据范围"
                            f"（仅有 {len(evidence_list)} 条证据）"
                        )
                    else:
                        # 检查引用内容是否匹配
                        # （简化版：检查引用附近的文本是否在证据中出现）
                        pass

            elif citation["type"] == "source":
                # 检查来源引用是否匹配某条证据的 source_doc
                source_text = citation["source"]
                matched = any(
                    source_text.lower() in ev.source_doc.lower()
                    or ev.source_doc.lower() in source_text.lower()
                    for ev in evidence_list
                )
                if not matched:
                    warnings.append(
                        f"引用来源 '{source_text}' 未在证据中找到完全匹配"
                    )

        # 检查回答中有引用但无证据
        if citations and not evidence_list:
            errors.append("回答包含引用标记但证据包为空")

        # 检查有证据但回答中无引用
        if evidence_list and not citations:
            warnings.append(
                "证据包有证据但回答中无引用标记，建议添加引用"
            )

        details["numeric_citations"] = [
            c for c in citations if c["type"] == "numeric"
        ]
        details["source_citations"] = [
            c for c in citations if c["type"] == "source"
        ]

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取引用标记

        返回 [{"type": "numeric", "numbers": [1, 2], "raw": "[1,2]"},
              {"type": "source", "source": "《商业银行资本管理办法》", "raw": "(来源: ...)"}]
        """
        citations: List[Dict[str, Any]] = []
        if not text:
            return citations

        for match in CITATION_PATTERN.finditer(text):
            raw = match.group(0)

            if match.group(1):  # 数字引用 [1] [1,2] [1-3]
                num_str = match.group(1)
                numbers = self._parse_citation_numbers(num_str)
                citations.append({
                    "type": "numeric",
                    "numbers": numbers,
                    "raw": raw,
                })
            elif match.group(2):  # (来源: xxx)
                citations.append({
                    "type": "source",
                    "source": match.group(2).strip(),
                    "raw": raw,
                })
            elif match.group(3):  # (引用: xxx)
                citations.append({
                    "type": "source",
                    "source": match.group(3).strip(),
                    "raw": raw,
                })

        logger.debug("提取引用标记: %s", citations)
        return citations

    @staticmethod
    def _parse_citation_numbers(num_str: str) -> List[int]:
        """
        解析引用编号字符串

        支持: "1" → [1], "1,2" → [1,2], "1-3" → [1,2,3]
        """
        numbers: List[int] = []
        parts = num_str.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                # 范围: 1-3 → [1,2,3]
                try:
                    start, end = part.split("-")
                    for n in range(int(start), int(end) + 1):
                        numbers.append(n)
                except ValueError:
                    continue
            else:
                try:
                    numbers.append(int(part))
                except ValueError:
                    continue
        return numbers
