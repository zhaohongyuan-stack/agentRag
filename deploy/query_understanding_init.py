"""
查询理解模块 — QuerySpec 构建器

将用户原始问题转换为结构化的 QuerySpec 对象，包含：
  - 意图分类（intent_classifier）
  - 实体抽取（entity_extractor）
  - 约束提取（constraint_extractor）
  - 歧义检测（ambiguity_detector）
  - 复杂度评级
  - 风险级别评估
  - 检索通道建议
  - 回答形态预测

QuerySpec 是后续所有模块的统一输入，后续模块只消费结构化对象。
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .ambiguity_detector.detector import Ambiguity, AmbiguityDetector
from .constraint_extractor.extractor import ConstraintExtractor, QueryConstraints
from .entity_extractor.extractor import EntityExtractor, ExtractedEntity
from .intent_classifier.classifier import IntentClassifier, IntentResult
from .query_decomposer import QueryDecomposer, SubQuery
from .context_anchor import ContextAnchorExtractor, ContextAnchor


# ============================================================
# QuerySpec 数据结构
# 与 contracts/schemas/query_spec.schema.json 对齐
# ============================================================
@dataclass
class QuerySpec:
    """
    查询规格 — 用户问题的结构化表示

    这是 B组查询理解模块的统一产出，后续所有模块只消费此对象。
    """

    query_id: str
    raw_query: str
    intent: str
    complexity: str  # L0-L4
    risk_level: str  # low/medium/high/critical
    confidence: float

    # 可选字段
    contextualized_query: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    answer_shape: Optional[str] = None
    retrieval_needs: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)
    sub_queries: List[Dict[str, Any]] = field(default_factory=list)
    context_anchors: List[Dict[str, Any]] = field(default_factory=list)
    conversation_refs: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "raw_query": self.raw_query,
            "contextualized_query": self.contextualized_query,
            "intent": self.intent,
            "entities": self.entities,
            "constraints": self.constraints,
            "answer_shape": self.answer_shape,
            "retrieval_needs": self.retrieval_needs,
            "claims": self.claims,
            "complexity": self.complexity,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "ambiguities": self.ambiguities,
            "sub_queries": self.sub_queries,
            "context_anchors": self.context_anchors,
            "conversation_refs": self.conversation_refs,
        }


# ============================================================
# 意图 → 复杂度/风险/通道 映射表
# ============================================================
INTENT_TO_COMPLEXITY: Dict[str, str] = {
    "greeting": "L0",
    "clause_query": "L1",
    "definition": "L2",
    "threshold": "L2",
    "table_lookup": "L2",
    "comparison": "L3",
    "compliance": "L4",
    "overview": "L2",
    "unknown": "L2",
}

INTENT_TO_RISK: Dict[str, str] = {
    "greeting": "low",
    "clause_query": "low",
    "definition": "low",
    "threshold": "medium",
    "table_lookup": "low",
    "comparison": "medium",
    "compliance": "high",
    "overview": "low",
    "unknown": "medium",
}

INTENT_TO_CHANNELS: Dict[str, List[Dict[str, Any]]] = {
    "greeting": [],
    "clause_query": [
        {"channel": "exact", "priority": "must"},
        {"channel": "metadata", "priority": "must"},
    ],
    "definition": [
        {"channel": "lexical", "priority": "must"},
        {"channel": "dense", "priority": "should"},
        {"channel": "metadata", "priority": "should"},
    ],
    "threshold": [
        {"channel": "lexical", "priority": "must"},
        {"channel": "dense", "priority": "must"},
        {"channel": "metadata", "priority": "should"},
    ],
    "table_lookup": [
        {"channel": "table", "priority": "must"},
        {"channel": "metadata", "priority": "must"},
    ],
    "comparison": [
        {"channel": "hybrid", "priority": "must"},
        {"channel": "table", "priority": "should"},
        {"channel": "metadata", "priority": "should"},
    ],
    "compliance": [
        {"channel": "hybrid", "priority": "must"},
        {"channel": "exact", "priority": "should"},
        {"channel": "neighborhood", "priority": "should"},
        {"channel": "relation", "priority": "optional"},
    ],
    "overview": [
        {"channel": "dense", "priority": "must"},
        {"channel": "metadata", "priority": "should"},
    ],
    "unknown": [
        {"channel": "hybrid", "priority": "must"},
        {"channel": "metadata", "priority": "should"},
    ],
}

INTENT_TO_ANSWER_SHAPE: Dict[str, str] = {
    "greeting": "text_excerpt",
    "clause_query": "text_excerpt",
    "definition": "text_excerpt",
    "threshold": "single_value",
    "table_lookup": "table_cell",
    "comparison": "comparison_table",
    "compliance": "yes_no",
    "overview": "summary",
    "unknown": "text_excerpt",
}


# ============================================================
# QuerySpec 构建器
# ============================================================
class QuerySpecBuilder:
    """
    QuerySpec 构建器

    组合意图分类、实体抽取、约束提取、歧义检测，
    产出完整的 QuerySpec 对象。
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        constraint_extractor: Optional[ConstraintExtractor] = None,
        ambiguity_detector: Optional[AmbiguityDetector] = None,
    ):
        self._intent_classifier = intent_classifier or IntentClassifier()
        self._entity_extractor = entity_extractor or EntityExtractor()
        self._constraint_extractor = constraint_extractor or ConstraintExtractor()
        self._ambiguity_detector = ambiguity_detector or AmbiguityDetector()
        self._query_decomposer = QueryDecomposer()
        self._context_anchor_extractor = ContextAnchorExtractor()

    def build(self, query: str, session_id: Optional[str] = None) -> QuerySpec:
        """
        从用户问题构建 QuerySpec

        Args:
            query: 用户原始问题
            session_id: 会话 ID（用于多轮对话引用）

        Returns:
            QuerySpec 对象
        """
        query_id = str(uuid.uuid4())

        # 1. 意图分类
        intent_result: IntentResult = self._intent_classifier.classify(query)

        # 2. 实体抽取
        entities: List[ExtractedEntity] = self._entity_extractor.extract(query)

        # 3. 约束提取
        constraints: QueryConstraints = self._constraint_extractor.extract(query)

        # 4. 歧义检测
        ambiguities: List[Ambiguity] = self._ambiguity_detector.detect(
            query, entities=entities, intent=intent_result.intent
        )

        # 4.5 查询分解（多选题等复合查询拆分）
        sub_queries = self._query_decomposer.decompose(query)
        if sub_queries:
            logger.info(f"[查询理解] 检测到复合查询，分解为 {len(sub_queries)} 个子问题")

        # 4.6 语境锚点提取（用于歧义场景下的强语境优先检索）
        context_anchors = self._context_anchor_extractor.extract(
            query, entities=[e.to_dict() for e in entities]
        )

        # 5. 复杂度评级
        complexity = self._assess_complexity(intent_result, entities, ambiguities)

        # 6. 风险级别
        risk_level = self._assess_risk(intent_result.intent, complexity)

        # 7. 检索通道建议
        retrieval_needs = INTENT_TO_CHANNELS.get(intent_result.intent, [])

        # 8. 回答形态
        answer_shape = INTENT_TO_ANSWER_SHAPE.get(intent_result.intent, "text_excerpt")

        # 9. 声明槽位（第一版预定义模板，Phase 2 再细化）
        claims = self._build_claims(intent_result.intent, entities)

        # 10. 会话引用
        conversation_refs = None
        if session_id:
            conversation_refs = {"session_id": session_id, "turn_number": 0}

        return QuerySpec(
            query_id=query_id,
            raw_query=query,
            contextualized_query=query,  # Phase 1 无指代消解，直接使用原始问题
            intent=intent_result.intent,
            entities=[e.to_dict() for e in entities],
            constraints=constraints.to_dict(),
            answer_shape=answer_shape,
            retrieval_needs=retrieval_needs,
            claims=claims,
            complexity=complexity,
            risk_level=risk_level,
            confidence=intent_result.confidence,
            ambiguities=[a.to_dict() for a in ambiguities],
            sub_queries=[sq.to_dict() for sq in sub_queries],
            context_anchors=[a.to_dict() for a in context_anchors],
            conversation_refs=conversation_refs,
        )

    def _assess_complexity(
        self,
        intent_result: IntentResult,
        entities: List[ExtractedEntity],
        ambiguities: List[Ambiguity],
    ) -> str:
        """
        复杂度评级（纯规则，L0-L4）

        L0: 问候，直接回复
        L1: 精确文号/条款号查询
        L2: 普通事实查询（定义、阈值、表格取数）
        L3: 比较查询，需拆解为多个子问题
        L4: 合规判断，多跳推理
        """
        base_level = INTENT_TO_COMPLEXITY.get(intent_result.intent, "L2")

        # 有歧义时提升一级（最高 L4）
        if ambiguities and base_level != "L0":
            level_num = int(base_level[1:])
            level_num = min(level_num + 1, 4)
            base_level = f"L{level_num}"

        return base_level

    def _assess_risk(self, intent: str, complexity: str) -> str:
        """风险级别评估"""
        base_risk = INTENT_TO_RISK.get(intent, "medium")

        # 复杂度 L3/L4 风险升级
        if complexity in ("L3", "L4") and base_risk == "low":
            base_risk = "medium"
        if complexity == "L4" and base_risk == "medium":
            base_risk = "high"

        return base_risk

    def _build_claims(
        self, intent: str, entities: List[ExtractedEntity]
    ) -> List[Dict[str, Any]]:
        """
        构建声明槽位（第一版简单模板）

        Phase 2 会接入完整的 claim_slots.yaml 模板。
        """
        claims = []

        if intent == "threshold":
            # 阈值查询: 需要找到指标名称、最低值、适用主体、生效时间
            metric_entities = [e for e in entities if e.entity_type == "metric_name"]
            metric_name = metric_entities[0].value if metric_entities else "未指定指标"

            claims = [
                {"claim_id": "c1", "description": "指标名称", "slot_type": "metric", "status": "pending", "evidence_ids": []},
                {"claim_id": "c2", "description": f"{metric_name}的最低要求值", "slot_type": "value", "status": "pending", "evidence_ids": []},
                {"claim_id": "c3", "description": "适用主体", "slot_type": "subject", "status": "pending", "evidence_ids": []},
                {"claim_id": "c4", "description": "法规依据", "slot_type": "source", "status": "pending", "evidence_ids": []},
            ]

        elif intent == "clause_query":
            # 条款查询: 需要找到条款内容、所属文档、适用范围
            claims = [
                {"claim_id": "c1", "description": "条款内容", "slot_type": "text", "status": "pending", "evidence_ids": []},
                {"claim_id": "c2", "description": "所属文档", "slot_type": "source", "status": "pending", "evidence_ids": []},
                {"claim_id": "c3", "description": "适用范围", "slot_type": "scope", "status": "pending", "evidence_ids": []},
            ]

        elif intent == "definition":
            # 定义查询: 需要找到术语定义、定义来源
            term_entities = [e for e in entities if e.entity_type == "term"]
            term_name = term_entities[0].value if term_entities else "未指定术语"

            claims = [
                {"claim_id": "c1", "description": f"{term_name}的定义", "slot_type": "text", "status": "pending", "evidence_ids": []},
                {"claim_id": "c2", "description": "定义来源", "slot_type": "source", "status": "pending", "evidence_ids": []},
            ]

        elif intent == "table_lookup":
            # 表格取数: 需要找到表格名、行/列、具体数值
            claims = [
                {"claim_id": "c1", "description": "表格名称", "slot_type": "source", "status": "pending", "evidence_ids": []},
                {"claim_id": "c2", "description": "行/列标识", "slot_type": "position", "status": "pending", "evidence_ids": []},
                {"claim_id": "c3", "description": "单元格数值", "slot_type": "value", "status": "pending", "evidence_ids": []},
            ]

        return claims


# ============================================================
# 模块导出
# ============================================================
__all__ = [
    "QuerySpec",
    "QuerySpecBuilder",
    "IntentClassifier",
    "IntentResult",
    "EntityExtractor",
    "ExtractedEntity",
    "ConstraintExtractor",
    "QueryConstraints",
    "AmbiguityDetector",
    "Ambiguity",
    "QueryDecomposer",
    "SubQuery",
    "ContextAnchorExtractor",
    "ContextAnchor",
    "INTENT_TO_COMPLEXITY",
    "INTENT_TO_RISK",
    "INTENT_TO_CHANNELS",
]
