"""
版本验证器 — 校验引用法规版本在查询时间点是否有效

职责:
  1. 检查证据的 version_status（active / superseded / repealed）
  2. 根据查询时间点判断版本是否适用
  3. 如果引用了已失效版本，标记为 invalid

模式参考: evidence/conflict_detector/ 的 dataclass + to_dict 风格
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...evidence.evidence_assembler.builder import EvidenceItem
from ..numeric_validator.validator import ValidationResult


logger = logging.getLogger(__name__)


# 版本状态值
ACTIVE_STATUSES = {"active", "current", "in_force", "有效"}
SUPERSEDED_STATUSES = {"superseded", "replaced", "已被替代"}
REPEALED_STATUSES = {"repealed", "abolished", "废止", "已废止"}


@dataclass
class VersionInfo:
    """
    版本信息

    Attributes:
        version_status: 版本状态（active / superseded / repealed）
        effective_date: 生效日期（ISO 字符串）
        superseded_date: 被替代日期
        source_doc: 来源文档
    """

    version_status: str = "active"
    effective_date: str = ""
    superseded_date: str = ""
    source_doc: str = ""


class VersionValidator:
    """
    版本验证器

    校验证据在指定查询时间点是否为有效版本。

    用法:
        validator = VersionValidator()
        result = validator.validate(claim_text, evidence, query_date="2026-01-01")
    """

    # 日期提取正则（支持 YYYY-MM-DD / YYYY年MM月DD日 / YYYY.MM.DD）
    DATE_PATTERN = re.compile(
        r"(\d{4})[-年./](\d{1,2})[-月./](\d{1,2})日?"
    )

    def validate(
        self,
        claim_text: str,
        evidence: List[EvidenceItem],
        query_date: str = "",
    ) -> ValidationResult:
        """
        验证证据版本有效性

        Args:
            claim_text: 声明文本
            evidence: 支持该声明的证据列表
            query_date: 查询时间点（ISO 字符串，如 "2026-01-01"），
                        为空时不做时间点校验，仅检查版本状态

        Returns:
            ValidationResult
        """
        if not evidence:
            return ValidationResult(
                valid=True,
                details={"reason": "无证据，跳过版本校验"},
            )

        errors: List[str] = []
        warnings: List[str] = []
        version_details: List[Dict[str, Any]] = []

        query_normalized = self._normalize_date(query_date) if query_date else ""

        for ev in evidence:
            info = self._extract_version_info(ev)
            detail: Dict[str, Any] = {
                "evidence_id": ev.evidence_id,
                "source_doc": info.source_doc,
                "version_status": info.version_status,
                "effective_date": info.effective_date,
            }

            # 检查版本状态
            status_lower = info.version_status.lower()
            if status_lower in REPEALED_STATUSES:
                errors.append(
                    f"证据 {ev.evidence_id}（{info.source_doc}）已废止，"
                    f"不能作为有效依据"
                )
                detail["issue"] = "repealed"
            elif status_lower in SUPERSEDED_STATUSES:
                warnings.append(
                    f"证据 {ev.evidence_id}（{info.source_doc}）已被替代，"
                    f"建议使用最新版本"
                )
                detail["issue"] = "superseded"
                # 被替代版本仍可参考，但不作为主要依据 → valid=True 但有 warning
            elif status_lower not in ACTIVE_STATUSES and status_lower:
                # 未知状态 → 警告
                warnings.append(
                    f"证据 {ev.evidence_id} 版本状态未知: {info.version_status}"
                )
                detail["issue"] = "unknown_status"

            # 检查时间点有效性
            if query_normalized and info.effective_date:
                ev_date = self._normalize_date(info.effective_date)
                if ev_date and ev_date > query_normalized:
                    errors.append(
                        f"证据 {ev.evidence_id} 生效日期 {info.effective_date} "
                        f"晚于查询时间点 {query_date}，在该时间点尚未生效"
                    )
                    detail["issue"] = "not_yet_effective"

            # 检查被替代日期
            if query_normalized and info.superseded_date:
                sup_date = self._normalize_date(info.superseded_date)
                if sup_date and sup_date <= query_normalized:
                    errors.append(
                        f"证据 {ev.evidence_id}（{info.source_doc}）"
                        f"在查询时间点 {query_date} 已被替代"
                    )
                    detail["issue"] = "superseded_before_query"

            version_details.append(detail)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={"versions": version_details},
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _extract_version_info(self, evidence: EvidenceItem) -> VersionInfo:
        """
        从证据中提取版本信息

        优先使用 metadata 中的版本字段，其次从 content 中解析日期。
        """
        meta = evidence.metadata or {}
        info = VersionInfo(
            version_status=getattr(evidence, "version_status", "active")
            or meta.get("version_status", "active"),
            effective_date=meta.get("effective_date", ""),
            superseded_date=meta.get("superseded_date", ""),
            source_doc=evidence.source_doc,
        )

        # 如果没有生效日期，尝试从内容中提取
        if not info.effective_date:
            dates = self.DATE_PATTERN.findall(evidence.content)
            if dates:
                # 取第一个日期作为可能的生效日期
                y, m, d = dates[0]
                info.effective_date = f"{y}-{int(m):02d}-{int(d):02d}"

        return info

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """
        归一化日期字符串为 YYYY-MM-DD 格式

        支持 YYYY-MM-DD / YYYY年MM月DD日 / YYYY.MM.DD
        """
        if not date_str:
            return ""
        match = VersionValidator.DATE_PATTERN.search(date_str)
        if match:
            y, m, d = match.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"
        return date_str
