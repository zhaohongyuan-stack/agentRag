"""
风险路由器 — 基于规则的风险评级模块

根据查询意图、关键词、数值敏感度评估回答风险级别（low/medium/high）。

风险评估规则:
  - 意图驱动:
      compliance  → high  （合规查询涉及判断，风险最高）
      threshold   → medium（阈值查询涉及数值，风险中等）
      comparison  → medium（比较查询涉及多个对象，风险中等）
      definition  → low   （定义查询，风险低）
      overview    → low   （概览查询，风险低）
      clause_query→ low   （条款查询，精确匹配，风险低）
  - 关键词升级:
      "不得"/"禁止"/"严禁" → 升级为 high（禁止性条款）
      "处罚"/"违规"/"违法" → high（涉及处罚后果）
      "是否符合"/"是否满足"/"是否合规" → high（合规判断类）
  - 数值敏感度:
      查询同时包含百分比/金额 + 阈值关键词 → high

来源: 问题确认.md — 风险规则用户确认
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class RiskLevel(Enum):
    """
    风险级别枚举

    - LOW:    低风险 — 定义、概览、条款查询
    - MEDIUM: 中风险 — 阈值、比较查询
    - HIGH:   高风险 — 合规查询、禁止性条款、处罚相关
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "RiskLevel":
        """从字符串构建 RiskLevel"""
        mapping = {
            "low": cls.LOW,
            "medium": cls.MEDIUM,
            "high": cls.HIGH,
        }
        return mapping.get(value.lower(), cls.MEDIUM)


@dataclass
class RiskAssessment:
    """
    风险评估结果

    包含风险级别、判定原因和触发因子列表。
    """

    level: str  # low / medium / high
    reason: str
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "reason": self.reason,
            "factors": self.factors,
        }


# ============================================================
# 意图 → 基础风险级别 映射表
# ============================================================
INTENT_TO_BASE_RISK: Dict[str, str] = {
    "greeting": "low",
    "clause_query": "low",
    "definition": "low",
    "overview": "low",
    "table_lookup": "low",
    "threshold": "medium",
    "comparison": "medium",
    "unknown": "medium",
    "compliance": "high",
}

# ============================================================
# 高风险关键词（出现即升级为 high）
# ============================================================

# 禁止性条款关键词 — 直接升级为 high
PROHIBITIVE_KEYWORDS: List[str] = [
    "不得", "禁止", "严禁", "不准", "不得低于", "不得超过",
    "不得少于", "不得高于", "不得违反",
]

# 处罚/违规关键词 — 直接升级为 high
PENALTY_KEYWORDS: List[str] = [
    "处罚", "违规", "违法", "罚款", "罚没", "行政处罚",
    "法律责任", "追责", "处分", "惩戒",
]

# 合规判断关键词 — 直接升级为 high
COMPLIANCE_CHECK_KEYWORDS: List[str] = [
    "是否符合", "是否满足", "是否合规", "是否达标",
    "是否违反", "是否可以", "能不能", "可不可以",
]

# 阈值关键词（用于数值敏感度判定）
THRESHOLD_KEYWORDS: List[str] = [
    "最低", "最高", "不少于", "不超过", "至少", "至多",
    "不低于", "不高于", "下限", "上限", "要求", "标准",
]

# 数值正则: 百分比 / 金额
PERCENTAGE_PATTERN = re.compile(r"\d+(?:\.\d+)?%")
AMOUNT_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:万亿|亿|万|千)元")


class RiskRouter:
    """
    风险路由器

    根据查询意图、关键词和数值敏感度评估风险级别。
    纯规则驱动，不依赖 LLM。

    用法:
        router = RiskRouter()
        assessment = router.assess(query_spec)
        # 或直接传参:
        assessment = router.assess(intent="threshold", query_text="不得低于8%")
    """

    def __init__(self):
        """初始化风险路由器"""
        self._prohibitive_keywords = list(PROHIBITIVE_KEYWORDS)
        self._penalty_keywords = list(PENALTY_KEYWORDS)
        self._compliance_check_keywords = list(COMPLIANCE_CHECK_KEYWORDS)
        self._threshold_keywords = list(THRESHOLD_KEYWORDS)

    # ----------------------------------------------------------
    # 主评估入口（支持两种调用方式）
    # ----------------------------------------------------------
    def assess(
        self,
        query_spec: Any = None,
        *,
        intent: Optional[str] = None,
        query_text: Optional[str] = None,
        entities: Optional[List[Any]] = None,
    ) -> RiskAssessment:
        """
        评估查询的风险级别

        支持两种调用方式:
          1. assess(query_spec)         — 传入 QuerySpec 对象
          2. assess(intent=..., query_text=..., entities=...) — 直接传参

        Args:
            query_spec: QuerySpec 对象（或兼容的 dict），优先使用
            intent: 查询意图（query_spec 为 None 时使用）
            query_text: 原始问题文本（query_spec 为 None 时使用）
            entities: 实体列表（query_spec 为 None 时使用）

        Returns:
            RiskAssessment 对象
        """
        # 解析输入参数
        if query_spec is not None:
            resolved = self._resolve_from_spec(query_spec)
            resolved_intent = resolved["intent"]
            resolved_text = resolved["query_text"]
            resolved_entities = resolved["entities"]
        else:
            resolved_intent = intent or "unknown"
            resolved_text = query_text or ""
            resolved_entities = entities or []

        # 1. 意图驱动的基础风险级别
        base_risk = INTENT_TO_BASE_RISK.get(resolved_intent, "medium")
        factors: List[str] = []
        reasons: List[str] = []

        factors.append(f"intent={resolved_intent}")
        reasons.append(f"意图 '{resolved_intent}' 基础风险为 {base_risk}")

        # 2. 关键词升级检测
        keyword_risk = self._check_keywords(resolved_text)
        if keyword_risk:
            for kw_factor in keyword_risk["factors"]:
                factors.append(kw_factor)
            reasons.append(keyword_risk["reason"])
            # 关键词命中 → 升级为 high
            if keyword_risk["level"] == "high":
                base_risk = "high"

        # 3. 数值敏感度检测
        numeric_risk = self._check_numeric_sensitivity(resolved_text, resolved_entities)
        if numeric_risk:
            for num_factor in numeric_risk["factors"]:
                factors.append(num_factor)
            reasons.append(numeric_risk["reason"])
            # 数值 + 阈值 → 升级为 high
            if numeric_risk["level"] == "high" and base_risk != "high":
                base_risk = "high"

        # 4. 合规意图直接为 high（不可降级）
        if resolved_intent == "compliance":
            base_risk = "high"
            if "意图 'compliance' 基础风险为 high" not in reasons:
                reasons.append("合规查询意图，风险级别固定为 high")

        return RiskAssessment(
            level=base_risk,
            reason="; ".join(reasons),
            factors=factors,
        )

    # ----------------------------------------------------------
    # 内部方法: 从 QuerySpec 解析参数
    # ----------------------------------------------------------
    def _resolve_from_spec(self, query_spec: Any) -> Dict[str, Any]:
        """从 QuerySpec 对象或 dict 中解析意图、文本、实体"""
        if hasattr(query_spec, "intent"):
            # QuerySpec 对象
            intent = query_spec.intent
            query_text = (
                getattr(query_spec, "contextualized_query", None)
                or getattr(query_spec, "raw_query", "")
                or ""
            )
            entities = getattr(query_spec, "entities", []) or []
        elif isinstance(query_spec, dict):
            # dict 形式
            intent = query_spec.get("intent", "unknown")
            query_text = (
                query_spec.get("contextualized_query")
                or query_spec.get("raw_query", "")
                or ""
            )
            entities = query_spec.get("entities", []) or []
        else:
            intent = "unknown"
            query_text = str(query_spec)
            entities = []

        return {
            "intent": intent,
            "query_text": query_text,
            "entities": entities,
        }

    # ----------------------------------------------------------
    # 内部方法: 关键词检测
    # ----------------------------------------------------------
    def _check_keywords(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        检测高风险关键词

        Args:
            query_text: 原始问题文本

        Returns:
            命中时返回包含 level/reason/factors 的 dict，未命中返回 None
        """
        if not query_text:
            return None

        matched_factors: List[str] = []
        matched_categories: List[str] = []

        # 禁止性条款关键词
        for kw in self._prohibitive_keywords:
            if kw in query_text:
                matched_factors.append(f"prohibitive_keyword='{kw}'")
                if "禁止性条款" not in matched_categories:
                    matched_categories.append("禁止性条款")

        # 处罚/违规关键词
        for kw in self._penalty_keywords:
            if kw in query_text:
                matched_factors.append(f"penalty_keyword='{kw}'")
                if "处罚/违规" not in matched_categories:
                    matched_categories.append("处罚/违规")

        # 合规判断关键词
        for kw in self._compliance_check_keywords:
            if kw in query_text:
                matched_factors.append(f"compliance_check_keyword='{kw}'")
                if "合规判断" not in matched_categories:
                    matched_categories.append("合规判断")

        if not matched_factors:
            return None

        return {
            "level": "high",
            "reason": f"命中高风险关键词: {', '.join(matched_categories)}",
            "factors": matched_factors,
        }

    # ----------------------------------------------------------
    # 内部方法: 数值敏感度检测
    # ----------------------------------------------------------
    def _check_numeric_sensitivity(
        self, query_text: str, entities: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        检测数值敏感度: 百分比/金额 + 阈值关键词 → high

        Args:
            query_text: 原始问题文本
            entities: 实体列表（可能包含 percentage/amount 类型）

        Returns:
            命中时返回包含 level/reason/factors 的 dict，未命中返回 None
        """
        if not query_text:
            return None

        has_percentage = bool(PERCENTAGE_PATTERN.search(query_text))
        has_amount = bool(AMOUNT_PATTERN.search(query_text))

        # 也从实体列表中检测
        for entity in entities:
            entity_type = None
            if isinstance(entity, dict):
                entity_type = entity.get("entity_type")
            elif hasattr(entity, "entity_type"):
                entity_type = entity.entity_type

            if entity_type == "percentage":
                has_percentage = True
            elif entity_type == "amount":
                has_amount = True

        has_numeric = has_percentage or has_amount

        # 检测阈值关键词
        has_threshold_kw = any(kw in query_text for kw in self._threshold_keywords)

        if has_numeric and has_threshold_kw:
            numeric_type = []
            if has_percentage:
                numeric_type.append("百分比")
            if has_amount:
                numeric_type.append("金额")
            return {
                "level": "high",
                "reason": f"查询包含{'/'.join(numeric_type)}且涉及阈值关键词，数值敏感度高",
                "factors": [
                    f"numeric_present={'+'.join(numeric_type)}",
                    "threshold_keyword_present=true",
                ],
            }

        return None
