"""
查询 IR 构建 — 将 QuerySpec + 声明槽位编译为查询中间表示（Query IR）

职责:
  1. 根据 intent 选择检索算子（通道）
  2. 根据风险级别设置停止条件
  3. 生成答案形状预期（AnswerShape）
  4. 构建声明间依赖关系（Dependency）

Query IR 是逻辑计划与物理计划的输入，是查询编译的核心中间表示。

模式参考: routing/route_policy/ 的 dataclass + to_dict 风格
依赖: evidence_assembler/builder.py 中的 ClaimSlot
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ...evidence.evidence_assembler.builder import ClaimSlot


logger = logging.getLogger(__name__)


# ============================================================
# 数据结构定义
# ============================================================
@dataclass
class Operator:
    """
    检索算子

    描述一次检索操作：算子名称、检索通道、参数。
    通道类型: exact / lexical / dense / metadata / table
    """

    name: str
    channel: str  # exact / lexical / dense / metadata / table
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "channel": self.channel,
            "params": dict(self.params),
        }


@dataclass
class Dependency:
    """
    声明间依赖关系

    描述两个声明槽位之间的执行依赖:
      - sequential: to_claim 依赖 from_claim 的结果
      - parallel: 两者可并行执行
      - conditional: to_claim 仅在特定条件下需要
    """

    from_claim: str
    to_claim: str
    type: str  # sequential / parallel / conditional

    def to_dict(self) -> dict:
        return {
            "from_claim": self.from_claim,
            "to_claim": self.to_claim,
            "type": self.type,
        }


@dataclass
class AnswerShape:
    """
    答案形状预期

    描述期望的回答形态，用于指导生成与校验。
    answer_type: numeric / textual / tabular / comparative / boolean
    """

    answer_type: str  # numeric / textual / tabular / comparative / boolean
    format_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "answer_type": self.answer_type,
            "format_hint": self.format_hint,
        }


@dataclass
class StopCondition:
    """停止条件 — 描述何时终止检索/重试"""

    condition: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "description": self.description,
        }


@dataclass
class QueryIR:
    """
    查询中间表示（Query IR）

    查询编译的核心数据结构，由 IRBuilder 生成，
    作为 LogicalPlanner 与 PhysicalPlanner 的输入。
    """

    intent: str
    claims: List[ClaimSlot] = field(default_factory=list)  # List[ClaimSlot]
    entities: List[Dict[str, Any]] = field(default_factory=list)  # List[Dict]
    constraints: List[Dict[str, Any]] = field(default_factory=list)  # List[Dict]
    retrieval_operators: List[Operator] = field(default_factory=list)  # List[Operator]
    dependencies: List[Dependency] = field(default_factory=list)  # List[Dependency]
    risk_level: str = "medium"
    expected_answer: AnswerShape = field(
        default_factory=lambda: AnswerShape(answer_type="textual")
    )
    stop_conditions: List[StopCondition] = field(default_factory=list)  # List[StopCondition]

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "intent": self.intent,
            "claims": [self._claim_to_dict(c) for c in self.claims],
            "entities": list(self.entities),
            "constraints": list(self.constraints),
            "retrieval_operators": [o.to_dict() for o in self.retrieval_operators],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "risk_level": self.risk_level,
            "expected_answer": self.expected_answer.to_dict(),
            "stop_conditions": [s.to_dict() for s in self.stop_conditions],
        }

    @staticmethod
    def _claim_to_dict(claim: Any) -> dict:
        """将声明槽位序列化为字典（兼容 ClaimSlot 对象和 dict）"""
        if isinstance(claim, ClaimSlot):
            return claim.to_dict()
        if isinstance(claim, dict):
            return dict(claim)
        return {}


# ============================================================
# 意图 → 检索通道映射（与 physical_planner 的 PLAN_TEMPLATES 对齐）
# ============================================================
INTENT_CHANNELS: Dict[str, List[str]] = {
    "clause_query": ["exact", "metadata"],
    "threshold": ["lexical", "dense", "metadata"],
    "table_lookup": ["table", "metadata"],
    "definition": ["lexical", "dense"],
    "comparison": ["lexical", "dense", "metadata"],
}

# 未知意图使用的默认通道
DEFAULT_CHANNELS: List[str] = ["lexical", "dense", "metadata"]


# ============================================================
# 意图 → 答案形状映射
# ============================================================
INTENT_ANSWER_SHAPE: Dict[str, AnswerShape] = {
    "clause_query": AnswerShape(
        answer_type="textual", format_hint="条款原文引用+来源"
    ),
    "threshold": AnswerShape(
        answer_type="numeric", format_hint="数值+单位+法规依据"
    ),
    "table_lookup": AnswerShape(
        answer_type="tabular", format_hint="单元格值+行列标题+来源"
    ),
    "definition": AnswerShape(
        answer_type="textual", format_hint="术语定义+定义来源"
    ),
    "comparison": AnswerShape(
        answer_type="comparative", format_hint="对比表+差异说明+双方来源"
    ),
}

# query_spec.answer_shape 字符串 → answer_type 映射
_ANSWER_SHAPE_STRING_MAP: Dict[str, str] = {
    "single_value": "numeric",
    "text_excerpt": "textual",
    "table_cell": "tabular",
    "comparison_table": "comparative",
    "yes_no": "boolean",
    "summary": "textual",
}


# ============================================================
# 风险级别 → 停止条件映射
# ============================================================
RISK_STOP_CONDITIONS: Dict[str, List[StopCondition]] = {
    "low": [
        StopCondition("sufficiency_score >= 0.75", "低风险：充分性达0.75即停止"),
        StopCondition("max_retries == 1", "低风险：最多重试1次"),
        StopCondition("max_latency_ms <= 5000", "低风险：延迟上限5秒"),
    ],
    "medium": [
        StopCondition("sufficiency_score >= 0.85", "中风险：充分性达0.85即停止"),
        StopCondition("max_retries == 2", "中风险：最多重试2次"),
        StopCondition("max_latency_ms <= 8000", "中风险：延迟上限8秒"),
    ],
    "high": [
        StopCondition("sufficiency_score >= 0.90", "高风险：充分性达0.90即停止"),
        StopCondition("max_retries == 3", "高风险：最多重试3次"),
        StopCondition("max_latency_ms <= 12000", "高风险：延迟上限12秒"),
    ],
    "critical": [
        StopCondition("sufficiency_score >= 0.95", "极高风险：充分性达0.95即停止"),
        StopCondition("max_retries == 3", "极高风险：最多重试3次"),
        StopCondition("max_latency_ms <= 15000", "极高风险：延迟上限15秒"),
    ],
}


# ============================================================
# IRBuilder 主类
# ============================================================
class IRBuilder:
    """
    查询 IR 构建器

    将查询规格（QuerySpec）和声明槽位（ClaimSlot）编译为查询中间表示（QueryIR）。

    构建流程:
      1. 提取 intent / risk_level / entities / constraints
      2. 归一化声明槽位（dict → ClaimSlot）
      3. 根据 intent 选择检索算子（通道）
      4. 根据风险级别设置停止条件
      5. 生成答案形状预期
      6. 构建声明间依赖关系

    用法:
        builder = IRBuilder()
        ir = builder.build(query_spec, claims)
        print(ir.intent, ir.retrieval_operators)
    """

    def __init__(
        self,
        intent_channels: Dict[str, List[str]] = None,
        intent_answer_shape: Dict[str, AnswerShape] = None,
        risk_stop_conditions: Dict[str, List[StopCondition]] = None,
    ):
        """
        Args:
            intent_channels: 自定义意图→通道映射，为 None 时使用默认 INTENT_CHANNELS
            intent_answer_shape: 自定义意图→答案形状映射，为 None 时使用默认
            risk_stop_conditions: 自定义风险→停止条件映射，为 None 时使用默认
        """
        self._intent_channels = dict(INTENT_CHANNELS)
        if intent_channels:
            self._intent_channels.update(intent_channels)

        self._intent_answer_shape = dict(INTENT_ANSWER_SHAPE)
        if intent_answer_shape:
            self._intent_answer_shape.update(intent_answer_shape)

        self._risk_stop_conditions = {
            k: list(v) for k, v in RISK_STOP_CONDITIONS.items()
        }
        if risk_stop_conditions:
            for k, v in risk_stop_conditions.items():
                self._risk_stop_conditions[k] = list(v)

    def build(self, query_spec: Dict[str, Any], claims: List[Any]) -> QueryIR:
        """
        构建 Query IR

        Args:
            query_spec: 查询规格字典，需包含 intent / risk_level 等字段。
                        也兼容带 to_dict()/__dict__ 的 QuerySpec 对象。
            claims: 声明槽位列表（ClaimSlot 对象或 dict）

        Returns:
            QueryIR 对象
        """
        # 兼容非 dict 的 query_spec
        if not isinstance(query_spec, dict):
            query_spec = self._coerce_to_dict(query_spec)

        intent = query_spec.get("intent", "unknown")
        risk_level = query_spec.get("risk_level", "medium")

        # 归一化声明槽位
        normalized_claims = self._normalize_claims(claims)

        # 提取实体与约束
        entities = list(query_spec.get("entities", []))
        constraints = self._normalize_constraints(
            query_spec.get("constraints", {})
        )

        # 选择检索算子
        operators = self._select_operators(intent, normalized_claims, query_spec)

        # 生成答案形状
        expected_answer = self._build_answer_shape(intent, query_spec)

        # 设置停止条件
        stop_conditions = self._build_stop_conditions(risk_level, query_spec)

        # 构建依赖关系
        dependencies = self._build_dependencies(normalized_claims, risk_level)

        ir = QueryIR(
            intent=intent,
            claims=normalized_claims,
            entities=entities,
            constraints=constraints,
            retrieval_operators=operators,
            dependencies=dependencies,
            risk_level=risk_level,
            expected_answer=expected_answer,
            stop_conditions=stop_conditions,
        )
        logger.info(
            "QueryIR 构建完成: intent=%s, risk=%s, operators=%d, claims=%d",
            intent,
            risk_level,
            len(operators),
            len(normalized_claims),
        )
        return ir

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _coerce_to_dict(obj: Any) -> Dict[str, Any]:
        """将对象转换为字典（兼容 QuerySpec 对象）"""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return {"intent": "unknown"}

    def _normalize_claims(self, claims: List[Any]) -> List[ClaimSlot]:
        """
        归一化声明槽位为 ClaimSlot 对象

        兼容 ClaimSlot 对象和 dict 两种输入格式。
        """
        normalized: List[ClaimSlot] = []
        for i, c in enumerate(claims):
            if isinstance(c, ClaimSlot):
                normalized.append(c)
            elif isinstance(c, dict):
                normalized.append(
                    ClaimSlot(
                        claim_id=c.get("claim_id", f"c{i}"),
                        description=c.get("description", ""),
                        slot_type=c.get("slot_type", ""),
                        status=c.get("status", "pending"),
                        evidence_ids=list(c.get("evidence_ids", [])),
                    )
                )
            else:
                logger.warning("忽略无法识别的声明项: %r", c)
        return normalized

    @staticmethod
    def _normalize_constraints(constraints: Any) -> List[Dict[str, Any]]:
        """
        将约束条件归一化为列表

        query_spec.constraints 可能是 dict 或 list，统一转为 List[Dict]。
        """
        if not constraints:
            return []
        if isinstance(constraints, list):
            return list(constraints)
        if isinstance(constraints, dict):
            return [
                {"name": k, "value": v} for k, v in constraints.items() if v
            ]
        return []

    def _select_operators(
        self,
        intent: str,
        claims: List[ClaimSlot],
        query_spec: Dict[str, Any],
    ) -> List[Operator]:
        """
        根据意图选择检索算子

        每个 intent 对应一组检索通道，每个通道生成一个 Operator。
        所有算子覆盖全部声明槽位（claim_ids 记录在 params 中）。
        """
        channels = self._intent_channels.get(intent)
        if channels is None:
            channels = list(DEFAULT_CHANNELS)
            logger.debug(
                "意图 '%s' 无预定义通道，使用默认通道 %s", intent, channels
            )

        claim_ids = [c.claim_id for c in claims]
        top_k = query_spec.get("top_k", 10)

        operators: List[Operator] = []
        for ch in channels:
            op = Operator(
                name=f"retrieve_{ch}",
                channel=ch,
                params={
                    "claim_ids": list(claim_ids),
                    "top_k": top_k,
                },
            )
            operators.append(op)
        return operators

    def _build_answer_shape(
        self, intent: str, query_spec: Dict[str, Any]
    ) -> AnswerShape:
        """
        生成答案形状预期

        优先级:
          1. query_spec 中显式指定的 answer_shape / expected_answer
          2. intent → INTENT_ANSWER_SHAPE 映射
          3. 默认 textual
        """
        # 优先使用 query_spec 中的显式答案形状
        spec_shape = query_spec.get("answer_shape") or query_spec.get(
            "expected_answer"
        )
        if isinstance(spec_shape, dict):
            return AnswerShape(
                answer_type=spec_shape.get("answer_type", "textual"),
                format_hint=spec_shape.get(
                    "format_hint", spec_shape.get("template", "")
                ),
            )
        if isinstance(spec_shape, str) and spec_shape:
            answer_type = _ANSWER_SHAPE_STRING_MAP.get(spec_shape, "textual")
            return AnswerShape(answer_type=answer_type)

        # 使用 intent 映射
        return self._intent_answer_shape.get(
            intent, AnswerShape(answer_type="textual")
        )

    def _build_stop_conditions(
        self, risk_level: str, query_spec: Dict[str, Any]
    ) -> List[StopCondition]:
        """
        根据风险级别构建停止条件

        优先级:
          1. query_spec 中显式指定的停止条件
          2. 风险级别 → RISK_STOP_CONDITIONS 映射
          3. medium 风险默认条件
        """
        # 优先使用 query_spec 中显式指定的停止条件
        spec_stops = query_spec.get("stop_conditions")
        if isinstance(spec_stops, list) and spec_stops:
            return [
                StopCondition(
                    condition=(
                        s.get("condition", str(s))
                        if isinstance(s, dict)
                        else str(s)
                    ),
                    description=(
                        s.get("description", "")
                        if isinstance(s, dict)
                        else ""
                    ),
                )
                for s in spec_stops
            ]
        if isinstance(spec_stops, dict) and spec_stops:
            return [
                StopCondition(condition=f"{k} == {v}", description=str(k))
                for k, v in spec_stops.items()
                if v is not None
            ]

        # 使用风险级别映射
        conditions = self._risk_stop_conditions.get(risk_level)
        if conditions is None:
            conditions = self._risk_stop_conditions.get("medium", [])
            logger.debug(
                "风险级别 '%s' 无预定义停止条件，使用 medium 默认", risk_level
            )
        return list(conditions)

    def _build_dependencies(
        self, claims: List[ClaimSlot], risk_level: str
    ) -> List[Dependency]:
        """
        构建声明间依赖关系

        策略:
          - 声明数 < 2 时无依赖
          - 高/极高风险: 串行依赖（sequential），允许逐声明校验后推进
          - 低/中风险: 并行依赖（parallel），同时检索所有声明
        """
        if len(claims) < 2:
            return []

        dep_type = "sequential" if risk_level in ("high", "critical") else "parallel"
        deps: List[Dependency] = []
        for i in range(len(claims) - 1):
            deps.append(
                Dependency(
                    from_claim=claims[i].claim_id,
                    to_claim=claims[i + 1].claim_id,
                    type=dep_type,
                )
            )
        return deps
