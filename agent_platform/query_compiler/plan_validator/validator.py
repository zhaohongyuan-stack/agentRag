"""
计划合法性校验 — 校验物理计划与 QueryIR 的一致性

职责:
  1. 校验必填声明槽位是否有对应的检索算子（claim_ids 覆盖）
  2. 校验每个通道的 top_k > 0
  3. 校验超时时间 > 0
  4. 校验停止条件不为空

校验结果包含错误（errors，阻断执行）和警告（warnings，不阻断）。

模式参考: routing/route_policy/ 的 dataclass + to_dict 风格
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List

from ..physical_planner.planner import PhysicalPlan
from ..query_ir.ir_builder import QueryIR


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    校验结果

    Attributes:
        is_valid: 是否通过校验（无错误即为 True）
        errors: 错误列表（阻断执行的问题）
        warnings: 警告列表（不阻断执行，但需关注）
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class PlanValidator:
    """
    计划合法性校验器

    校验物理计划是否满足执行约束，以及与 QueryIR 的一致性。

    校验规则:
      1. 必填声明槽位必须有对应的检索算子（物理阶段的 claim_ids 覆盖）
      2. 每个有通道的阶段 top_k > 0
      3. 每个阶段的超时时间 > 0
      4. 停止条件不为空

    用法:
        validator = PlanValidator()
        result = validator.validate(physical_plan, query_ir)
        if not result.is_valid:
            print(result.errors)
    """

    def validate(
        self, physical_plan: PhysicalPlan, query_ir: QueryIR
    ) -> ValidationResult:
        """
        校验物理计划

        Args:
            physical_plan: 物理执行计划
            query_ir: 查询中间表示

        Returns:
            ValidationResult 对象
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 规则1：必填声明槽位必须有对应的检索算子
        self._check_claim_coverage(physical_plan, query_ir, errors, warnings)

        # 规则2：每个通道的 top_k > 0
        self._check_top_k(physical_plan, errors, warnings)

        # 规则3：超时时间 > 0
        self._check_timeout(physical_plan, errors, warnings)

        # 规则4：停止条件不为空
        self._check_stop_conditions(physical_plan, errors, warnings)

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings
        )
        logger.info(
            "计划校验完成: is_valid=%s, errors=%d, warnings=%d",
            is_valid,
            len(errors),
            len(warnings),
        )
        return result

    # ============================================================
    # 内部方法 — 各校验规则
    # ============================================================

    def _check_claim_coverage(
        self,
        physical_plan: PhysicalPlan,
        query_ir: QueryIR,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """
        规则1：校验必填声明槽位的检索覆盖

        收集物理计划所有阶段的 claim_ids，检查每个必填声明是否被覆盖。
        """
        # 收集物理计划中所有被覆盖的声明 ID
        covered_ids = set()
        for stage in physical_plan.stages:
            for cid in stage.claim_ids:
                covered_ids.add(cid)

        # 检查每个必填声明是否被覆盖
        for claim in query_ir.claims:
            if not self._is_required(claim):
                continue
            if claim.claim_id not in covered_ids:
                errors.append(
                    f"必填声明槽位 '{claim.claim_id}'"
                    f"（{claim.description}）无对应的检索算子"
                )

        if not query_ir.claims:
            warnings.append("QueryIR 无声明槽位，跳过声明覆盖校验")

    def _check_top_k(
        self,
        physical_plan: PhysicalPlan,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """
        规则2：校验每个有通道阶段的 top_k > 0

        无通道的阶段（如校验阶段）跳过此检查。
        """
        for stage in physical_plan.stages:
            if not stage.channels:
                continue
            if stage.top_k <= 0:
                errors.append(
                    f"阶段 '{stage.name}' 的 top_k={stage.top_k}，必须大于 0"
                )

    def _check_timeout(
        self,
        physical_plan: PhysicalPlan,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """规则3：校验每个阶段的超时时间 > 0"""
        for stage in physical_plan.stages:
            if stage.timeout_ms <= 0:
                errors.append(
                    f"阶段 '{stage.name}' 的 timeout_ms={stage.timeout_ms}，"
                    f"必须大于 0"
                )

    def _check_stop_conditions(
        self,
        physical_plan: PhysicalPlan,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """规则4：校验停止条件不为空"""
        if not physical_plan.stop_conditions:
            errors.append("物理计划缺少停止条件（stop_conditions 为空）")

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _is_required(claim: Any) -> bool:
        """
        判断声明槽位是否为必填

        通过 slot_type 编码判断:
          - 格式 "{template_key}|required" 或 "{template_key}|optional"
          - 默认 True（无法解析时视为必填）

        兼容 ClaimSlot 对象和 dict。
        """
        if isinstance(claim, dict):
            slot_type = claim.get("slot_type", "")
        else:
            slot_type = getattr(claim, "slot_type", "")

        if not slot_type:
            return True
        if "|" in slot_type:
            parts = slot_type.split("|", 1)
            return parts[1].strip().lower() != "optional"
        return True
