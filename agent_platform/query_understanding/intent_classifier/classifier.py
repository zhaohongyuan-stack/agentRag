"""
规则版意图分类器

基于正则表达式和关键词匹配，将用户问题分类为预定义意图类型。

意图类型（第一版4类 + 辅助5类）:
  核心意图: clause_query, definition, threshold, table_lookup
  辅助意图: comparison, compliance, overview, greeting, unknown

分类策略:
  1. 优先匹配高置信度规则（如精确文号、条款号）
  2. 多规则命中时取置信度最高的
  3. 无规则命中时返回 unknown
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class IntentResult:
    """意图分类结果"""

    intent: str
    confidence: float
    matched_rules: List[str]
    raw_query: str


# ============================================================
# 意图分类规则表
# 规则优先级: 列表前面的优先级更高
# ============================================================
INTENT_RULES: List[Tuple[str, str, float]] = [
    # ── 问候/打招呼 ── L0 直接回复
    ("greeting", r"^(你好|您好|hi|hello|hey|在吗|在不在|谢谢|感谢)\s*[！!？?。.]*$", 0.95),

    # ── 条款查询 ── 包含文号、条款号
    ("clause_query", r"第[一二三四五六七八九十百千\d]+条", 0.90),
    ("clause_query", r"《[^》]+》第[一二三四五六七八九十百千\d]+条", 0.95),
    ("clause_query", r"文号.*?号|〔\d{4}〕\d+号|银发\[\d{4}]\d+号", 0.85),
    ("clause_query", r"条款|条文|第[一二三四五六七八九十百千\d]+章", 0.80),

    # ── 表格取数 ── 包含表名、附件号、指标数值
    ("table_lookup", r"附件\d+|附表\d+", 0.85),
    ("table_lookup", r"表\s*\d+", 0.80),
    ("table_lookup", r"(指标|数值|数据|取值).*(?:多少|是什么|查询)", 0.70),

    # ── 阈值查询 ── 包含最低/最高/比例要求
    ("threshold", r"(?:最低|最高|不少于|不超过|至少|至多|不低于|不高于)", 0.85),
    ("threshold", r"(?:比例|比率|要求).*(?:多少|是什么|几)", 0.75),
    ("threshold", r"(?:充足率|资本要求|杠杆率|拨备率|覆盖率).*(?:多少|要求|标准)", 0.80),

    # ── 定义查询 ── 什么是/是指/定义
    ("definition", r"(?:什么是|什么叫|什么意思|是指|定义|含义|概念)", 0.85),
    ("definition", r"(?:解释|说明).*(?:是什么|含义)", 0.70),

    # ── 比较查询 ── 比较两个制度或指标
    ("comparison", r"(?:比较|对比|区别|差异|不同|vs|versus|VS)", 0.80),
    ("comparison", r"(?:哪个.*(?:高|低|大|小|严|松)|哪个.*好)", 0.75),

    # ── 合规查询 ── 是否符合/满足/合规
    ("compliance", r"(?:是否(?:符合|满足|合规|达标|违反)|合规性|是否可以)", 0.85),
    ("compliance", r"(?:能不能|可不可以|允许.*?吗)", 0.70),

    # ── 概览查询 ── 概述/概览/包括哪些
    ("overview", r"(?:概述|概览|简介|包括哪些|包含哪些|有哪些|主要内容)", 0.75),
    ("overview", r"(?:介绍|说明).*(?:内容|规定|要求)", 0.65),
]


class IntentClassifier:
    """
    规则版意图分类器

    使用正则表达式匹配用户问题，返回置信度最高的意图类型。
    """

    def __init__(self, rules: Optional[List[Tuple[str, str, float]]] = None):
        """
        Args:
            rules: 自定义规则列表，为 None 时使用默认 INTENT_RULES
        """
        self._rules = rules if rules is not None else INTENT_RULES

    def classify(self, query: str) -> IntentResult:
        """
        对用户问题进行意图分类

        Args:
            query: 用户原始问题

        Returns:
            IntentResult，包含意图类型、置信度和匹配的规则
        """
        if not query or not query.strip():
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                matched_rules=[],
                raw_query=query or "",
            )

        query_stripped = query.strip()
        matched: List[Tuple[str, float, str]] = []

        for intent, pattern, base_confidence in self._rules:
            regex = re.compile(pattern, re.IGNORECASE)
            if regex.search(query_stripped):
                matched.append((intent, base_confidence, pattern))

        if not matched:
            return IntentResult(
                intent="unknown",
                confidence=0.3,
                matched_rules=[],
                raw_query=query,
            )

        # 取置信度最高的意图
        matched.sort(key=lambda x: x[1], reverse=True)
        best_intent, best_confidence, best_pattern = matched[0]

        # 如果有多个不同意图命中，降低置信度
        unique_intents = set(m[0] for m in matched)
        if len(unique_intents) > 1:
            best_confidence *= 0.85

        return IntentResult(
            intent=best_intent,
            confidence=round(best_confidence, 4),
            matched_rules=[m[2] for m in matched if m[0] == best_intent],
            raw_query=query,
        )
