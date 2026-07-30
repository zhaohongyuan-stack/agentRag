"""
物理计划生成 — 将逻辑计划 + QueryIR 转换为物理执行计划

职责:
  1. 根据意图选择物理计划模板（通道 / top_k / 重排 / 预算）
  2. 将逻辑阶段实例化为物理阶段（含超时、操作序列）
  3. 设置停止条件（充分性阈值 + 最大重试次数）

第一版采用固定模板策略，后续可扩展为自适应调度。

模式参考: routing/route_policy/ 的 dataclass + to_dict 风格
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..logical_planner.planner import LogicalPlan
from ..query_ir.ir_builder import QueryIR, StopCondition


logger = logging.getLogger(__name__)


# ============================================================
# 物理计划模板（第一版固定模板）
# ============================================================
PLAN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "clause_query": {
        "channels": ["exact", "metadata"],
        "top_k": 10,
        "rerank": True,
        "budget_ms": 5000,
    },
    "threshold": {
        "channels": ["lexical", "dense", "metadata"],
        "top_k": 20,
        "rerank": True,
        "budget_ms": 8000,
    },
    "table_lookup": {
        "channels": ["table", "metadata"],
        "top_k": 5,
        "rerank": False,
        "budget_ms": 3000,
    },
    "definition": {
        "channels": ["lexical", "dense"],
        "top_k": 10,
        "rerank": True,
        "budget_ms": 5000,
    },
    "comparison": {
        "channels": ["lexical", "dense", "metadata"],
        "top_k": 20,
        "rerank": True,
        "budget_ms": 10000,
    },
}

# 默认模板（未知意图使用）
DEFAULT_TEMPLATE: Dict[str, Any] = {
    "channels": ["lexical", "dense", "metadata"],
    "top_k": 20,
    "rerank": True,
    "budget_ms": 8000,
}

# 默认停止条件（第一版固定：sufficiency_score >= 0.85, max_retries == 2）
DEFAULT_STOP_CONDITIONS: List[StopCondition] = [
    StopCondition("sufficiency_score >= 0.85", "证据充分性达到0.85即停止"),
    StopCondition("max_retries == 2", "最多重试2次"),
]


@dataclass
class PlanStage:
    """
    物理计划阶段

    描述一个物理执行阶段的具体参数：通道、top_k、重排、超时、操作序列。

    Attributes:
        name: 阶段名称
        channels: 检索通道列表
        top_k: 每通道召回数量
        rerank: 是否重排
        timeout_ms: 超时时间（毫秒）
        operations: 操作序列（如 retrieve:exact / rerank / fuse）
        condition: 阶段完成条件
        claim_ids: 该阶段覆盖的声明槽位 ID
    """

    name: str
    channels: List[str] = field(default_factory=list)
    top_k: int = 10
    rerank: bool = False
    timeout_ms: int = 5000
    operations: List[str] = field(default_factory=list)
    condition: str = ""
    claim_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "channels": list(self.channels),
            "top_k": self.top_k,
            "rerank": self.rerank,
            "timeout_ms": self.timeout_ms,
            "operations": list(self.operations),
            "condition": self.condition,
            "claim_ids": list(self.claim_ids),
        }


@dataclass
class PhysicalPlan:
    """
    物理执行计划

    由 PhysicalPlanner 生成，包含若干物理阶段和停止条件，
    是 DAG 执行器的直接输入。
    """

    plan_id: str
    intent: str
    stages: List[PlanStage] = field(default_factory=list)
    stop_conditions: List[StopCondition] = field(default_factory=list)
    budget_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "intent": self.intent,
            "stages": [s.to_dict() for s in self.stages],
            "stop_conditions": [s.to_dict() for s in self.stop_conditions],
            "budget_ms": self.budget_ms,
        }


class PhysicalPlanner:
    """
    物理计划生成器

    将逻辑计划与 QueryIR 结合，生成可执行的物理计划。

    生成流程:
      1. 根据 query_ir.intent 查找物理计划模板
      2. 将逻辑阶段映射为物理阶段（填充通道/top_k/超时等参数）
      3. 生成操作序列（retrieve + rerank + fuse）
      4. 设置停止条件（基础值 sufficiency >= 0.85 / max_retries == 2，
         若 IR 中存在更严格阈值则取更严格值）

    用法:
        planner = PhysicalPlanner()
        plan = planner.plan(logical_plan, query_ir)
        print(plan.stages[0].channels)
    """

    def __init__(self, templates: Dict[str, Dict[str, Any]] = None):
        """
        Args:
            templates: 自定义计划模板，为 None 时使用默认 PLAN_TEMPLATES。
                       传入时会与默认模板合并（自定义覆盖同名 key）。
        """
        self._templates: Dict[str, Dict[str, Any]] = {
            k: dict(v) for k, v in PLAN_TEMPLATES.items()
        }
        if templates:
            for key, tmpl in templates.items():
                self._templates[key] = dict(tmpl)

    def plan(
        self, logical_plan: LogicalPlan, query_ir: QueryIR
    ) -> PhysicalPlan:
        """
        生成物理计划

        Args:
            logical_plan: 逻辑计划
            query_ir: 查询中间表示

        Returns:
            PhysicalPlan 对象
        """
        plan_id = f"pp-{uuid.uuid4().hex[:8]}"
        intent = query_ir.intent

        # 选择模板
        template = self._templates.get(intent)
        if template is None:
            template = dict(DEFAULT_TEMPLATE)
            logger.debug("意图 '%s' 无专用模板，使用默认模板", intent)

        channels = list(template["channels"])
        top_k = template["top_k"]
        rerank = template["rerank"]
        budget_ms = template["budget_ms"]

        # 收集所有声明 ID（从逻辑计划算子的 params 中提取）
        all_claim_ids = self._collect_claim_ids(logical_plan)

        # 将逻辑阶段映射为物理阶段
        stages: List[PlanStage] = []
        for logical_stage in logical_plan.stages:
            has_operators = len(logical_stage.operators) > 0
            physical_stage = PlanStage(
                name=logical_stage.name,
                channels=list(channels) if has_operators else [],
                top_k=top_k if has_operators else 0,
                rerank=rerank if has_operators else False,
                timeout_ms=budget_ms,
                operations=self._build_operations(
                    channels if has_operators else [], rerank if has_operators else False
                ),
                condition=logical_stage.completion_condition
                or "all_operators_returned",
                claim_ids=list(all_claim_ids) if has_operators else [],
            )
            stages.append(physical_stage)

        # 如果没有阶段，创建默认检索阶段
        if not stages:
            stages.append(
                PlanStage(
                    name="retrieve",
                    channels=list(channels),
                    top_k=top_k,
                    rerank=rerank,
                    timeout_ms=budget_ms,
                    operations=self._build_operations(channels, rerank),
                    condition="all_operators_returned",
                    claim_ids=list(all_claim_ids),
                )
            )

        # 停止条件：基础值 + IR 更严格阈值升级
        stop_conditions = self._build_stop_conditions(query_ir)

        plan = PhysicalPlan(
            plan_id=plan_id,
            intent=intent,
            stages=stages,
            stop_conditions=stop_conditions,
            budget_ms=budget_ms,
        )
        logger.info(
            "物理计划生成完成: plan_id=%s, intent=%s, stages=%d",
            plan_id,
            intent,
            len(stages),
        )
        return plan

    # ============================================================
    # 内部方法
    # ============================================================

    def _collect_claim_ids(self, logical_plan: LogicalPlan) -> List[str]:
        """
        从逻辑计划算子中收集所有声明 ID

        遍历所有阶段的算子，提取 params.claim_ids（去重保序）。
        """
        claim_ids: List[str] = []
        seen = set()
        for stage in logical_plan.stages:
            for op in stage.operators:
                for cid in op.params.get("claim_ids", []):
                    if cid not in seen:
                        seen.add(cid)
                        claim_ids.append(cid)
        return claim_ids

    @staticmethod
    def _build_operations(channels: List[str], rerank: bool) -> List[str]:
        """
        构建操作序列

        根据通道和重排标志生成操作列表:
          - 每个通道 → retrieve:<channel>
          - rerank=True → 追加 rerank
          - 多通道 → 追加 fuse（结果融合）
        """
        ops: List[str] = [f"retrieve:{ch}" for ch in channels]
        if rerank:
            ops.append("rerank")
        if len(channels) > 1:
            ops.append("fuse")
        return ops

    def _build_stop_conditions(self, query_ir: QueryIR) -> List[StopCondition]:
        """
        构建停止条件

        第一版基础值: sufficiency_score >= 0.85, max_retries == 2
        若 IR 中存在更严格的充分性阈值或重试次数，则采用更严格值。
        """
        sufficiency = 0.85
        max_retries = 2

        for sc in query_ir.stop_conditions:
            # 解析 sufficiency_score >= X
            if "sufficiency_score" in sc.condition:
                parsed = self._parse_threshold(sc.condition, "sufficiency_score", ">=")
                if parsed is not None and parsed > sufficiency:
                    sufficiency = parsed
            # 解析 max_retries == X
            if "max_retries" in sc.condition:
                parsed = self._parse_threshold(sc.condition, "max_retries", "==")
                if parsed is not None and parsed > max_retries:
                    max_retries = int(parsed)

        return [
            StopCondition(
                f"sufficiency_score >= {sufficiency}",
                f"证据充分性达{sufficiency}即停止",
            ),
            StopCondition(
                f"max_retries == {max_retries}",
                f"最多重试{max_retries}次",
            ),
        ]

    @staticmethod
    def _parse_threshold(condition: str, key: str, op: str) -> float:
        """
        从条件字符串中解析数值

        如 "sufficiency_score >= 0.90" → 0.90
        解析失败时返回 None。
        """
        try:
            marker = f"{key} {op}"
            idx = condition.find(marker)
            if idx < 0:
                return None
            value_part = condition[idx + len(marker):].strip()
            # 取首个数字部分
            num_str = ""
            for ch in value_part:
                if ch.isdigit() or ch == ".":
                    num_str += ch
                elif num_str:
                    break
            return float(num_str) if num_str else None
        except (ValueError, IndexError):
            return None
