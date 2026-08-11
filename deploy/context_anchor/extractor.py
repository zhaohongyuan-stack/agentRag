"""
语境锚点提取器 — 关联强语境识别

从用户问题中提取"关联强语境"锚点：
  - 文档名（书名号《》内容）
  - 条款号 / 章节号
  - 指标名 / 术语
  - 具体数值 / 日期 / 百分比
  - 法规文号

当触发歧义警告时，用这些锚点构建增强检索查询，
优先检索与锚点强相关的文档，而非直接返回澄清请求。

权重体系：
  - doc_name: 0.9  （用户明确引用了文档，最强信号）
  - clause_number: 0.85
  - metric_name: 0.8
  - regulation_no: 0.8
  - date/number: 0.6
  - term: 0.5
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ContextAnchor:
    """语境锚点 — 问题中的强信号片段"""

    anchor_type: str  # doc_name / clause_number / metric_name / ...
    value: str  # 锚点值
    weight: float  # 权重 0-1
    source: str = ""  # 锚点来源（正则匹配 / 实体抽取 / 约束提取）

    def to_dict(self) -> dict:
        return {
            "anchor_type": self.anchor_type,
            "value": self.value,
            "weight": self.weight,
            "source": self.source,
        }


class ContextAnchorExtractor:
    """
    语境锚点提取器

    从用户问题中提取关联强语境锚点，构建增强检索查询。
    用于歧义场景下的优先检索，避免直接返回澄清请求。
    """

    # 锚点正则模式（按权重降序）
    ANCHOR_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
        ("doc_name", re.compile(r"《([^》]+)》"), 0.9),
        ("clause_number", re.compile(r"第([一二三四五六七八九十百零\d]+)条"), 0.85),
        ("chapter_number", re.compile(r"第([一二三四五六七八九十百零\d]+)章"), 0.85),
        ("regulation_no", re.compile(r"第\s*(\d+)\s*号"), 0.8),
        ("attachment_no", re.compile(r"附件\s*([一二三四五六七八九十\d]+)"), 0.75),
        ("percentage", re.compile(r"(\d+(?:\.\d+)?\s*%)"), 0.6),
        ("date", re.compile(r"(\d{4}\s*年(?:\d{1,2}\s*月(?:\d{1,2}\s*日)?)?)"), 0.6),
        ("year", re.compile(r"(\d{4})\s*(?:年|季度)"), 0.55),
    ]

    # 金融监管领域高频指标关键词
    METRIC_KEYWORDS = [
        "资本充足率", "核心一级资本", "一级资本", "二级资本",
        "杠杆率", "流动性覆盖率", "净稳定资金比例", "流动性比例",
        "偿付能力", "偿付能力充足率", "综合偿付能力充足率",
        "核心偿付能力充足率", "风险综合评级",
        "拨备覆盖率", "贷款拨备率", "不良贷款率",
        "折现率", "基础利率曲线", "综合溢价",
        "保险集团", "偿付能力报告",
    ]

    # 术语关键词
    TERM_KEYWORDS = [
        "寿险合同负债", "保险合同", "折现率曲线",
        "资本管理办法", "偿付能力监管规则",
    ]

    def extract(
        self,
        query: str,
        entities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ContextAnchor]:
        """
        从问题中提取语境锚点

        Args:
            query: 用户原始问题
            entities: 已抽取的实体列表（可选，用于补充锚点）

        Returns:
            锚点列表，按权重降序排列
        """
        if not query or not query.strip():
            return []

        anchors: List[ContextAnchor] = []
        seen_values = set()  # 去重

        # 1. 正则模式提取
        for anchor_type, pattern, weight in self.ANCHOR_PATTERNS:
            for match in pattern.finditer(query):
                value = match.group(1).strip()
                if value and value not in seen_values:
                    seen_values.add(value)
                    anchors.append(ContextAnchor(
                        anchor_type=anchor_type,
                        value=value,
                        weight=weight,
                        source="regex",
                    ))

        # 2. 指标关键词提取
        for keyword in self.METRIC_KEYWORDS:
            if keyword in query and keyword not in seen_values:
                seen_values.add(keyword)
                anchors.append(ContextAnchor(
                    anchor_type="metric_name",
                    value=keyword,
                    weight=0.8,
                    source="keyword",
                ))

        # 3. 术语关键词提取
        for keyword in self.TERM_KEYWORDS:
            if keyword in query and keyword not in seen_values:
                seen_values.add(keyword)
                anchors.append(ContextAnchor(
                    anchor_type="term",
                    value=keyword,
                    weight=0.5,
                    source="keyword",
                ))

        # 4. 从实体列表补充
        if entities:
            for entity in entities:
                etype = entity.get("entity_type", "")
                value = entity.get("value", "")
                if not value or value in seen_values:
                    continue

                weight_map = {
                    "doc_name": 0.9,
                    "metric_name": 0.8,
                    "term": 0.5,
                    "clause_number": 0.85,
                    "table_name": 0.7,
                }
                weight = weight_map.get(etype, 0.4)
                seen_values.add(value)
                anchors.append(ContextAnchor(
                    anchor_type=etype,
                    value=value,
                    weight=weight,
                    source="entity",
                ))

        # 按权重降序排列
        anchors.sort(key=lambda a: a.weight, reverse=True)
        return anchors

    def build_enhanced_query(
        self,
        original_query: str,
        anchors: List[ContextAnchor],
        max_anchors: int = 3,
    ) -> str:
        """
        用锚点构建增强检索查询

        将高权重锚点拼接到原始查询前，增强检索相关性。
        仅使用权重 >= 0.7 的锚点。

        Args:
            original_query: 原始查询
            anchors: 语境锚点列表
            max_anchors: 最多使用的锚点数

        Returns:
            增强后的检索查询
        """
        # 筛选高权重锚点
        strong_anchors = [
            a for a in anchors if a.weight >= 0.7
        ][:max_anchors]

        if not strong_anchors:
            return original_query

        # 构建增强查询：锚点 + 原始查询
        anchor_values = [a.value for a in strong_anchors]
        # 去掉原始查询中已包含的锚点（避免重复）
        enhanced_parts = []
        for val in anchor_values:
            if val not in original_query:
                enhanced_parts.append(val)

        if enhanced_parts:
            return " ".join(enhanced_parts) + " " + original_query
        return original_query

    def build_anchor_filters(
        self,
        anchors: List[ContextAnchor],
        max_filters: int = 1,
    ) -> Dict[str, str]:
        """
        从锚点构建检索过滤条件

        仅使用单个最高权重的 doc_name 作为过滤条件（避免多值覆盖问题）。
        其他锚点通过查询增强（非过滤）方式影响检索。

        Args:
            anchors: 语境锚点列表
            max_filters: 最多构建的过滤条件数

        Returns:
            过滤条件字典
        """
        filters: Dict[str, str] = {}

        # 只取第一个 doc_name 锚点作为过滤条件
        doc_name_anchors = [
            a for a in anchors if a.anchor_type == "doc_name"
        ]
        if doc_name_anchors:
            filters["doc_name"] = doc_name_anchors[0].value

        return filters

    def has_strong_context(self, anchors: List[ContextAnchor]) -> bool:
        """
        判断是否存在强语境锚点

        当存在权重 >= 0.7 的锚点时，认为有强语境，
        可以在歧义场景下先尝试检索而非直接澄清。

        Args:
            anchors: 语境锚点列表

        Returns:
            是否存在强语境
        """
        return any(a.weight >= 0.7 for a in anchors)
