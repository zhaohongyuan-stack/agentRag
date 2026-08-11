"""
规则版实体抽取器

从用户问题中抽取结构化实体：
  - 文档名（doc_name）: 《...》 包裹的文本
  - 条款号（clause_number）: 第X条
  - 章节号（chapter_number）: 第X章
  - 附件号（attachment_no）: 附件X / 附表X
  - 表名（table_name）: 表X
  - 指标名（metric_name）: 资本充足率、杠杆率等
  - 术语（term）: 什么是XXX 中的 XXX
  - 日期（date）: YYYY年MM月等
  - 百分比（percentage）: XX%
  - 金额（amount）: XX万亿/亿元
  - 范围（scope）: 大型银行、系统重要性银行等
  - 机构（organization）: 银保监会、人民银行等
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedEntity:
    """抽取的实体"""

    entity_type: str
    value: str
    raw_text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }


# ============================================================
# 实体抽取规则
# ============================================================

# 文档名: 《商业银行资本管理办法》
DOC_NAME_PATTERN = re.compile(r"《([^》]+)》")

# 条款号: 第四十三条 / 第43条
CLAUSE_NUMBER_PATTERN = re.compile(
    r"第([一二三四五六七八九十百千零\d]+)条"
)

# 章节号: 第七章 / 第7章
CHAPTER_NUMBER_PATTERN = re.compile(
    r"第([一二三四五六七八九十百千零\d]+)章"
)

# 附件号: 附件1 / 附表2 / 附件三
ATTACHMENT_PATTERN = re.compile(
    r"附件([一二三四五六七八九十\d]+)|附表([一二三四五六七八九十\d]+)"
)

# 表名: 表1 / 表2.1 / 表三
TABLE_NAME_PATTERN = re.compile(
    r"表([一二三四五六七八九十\d]+(?:\.\d+)?)"
)

# 日期: 2026年1月 / 2026年Q1 / 2026年第一季度
DATE_PATTERN = re.compile(
    r"(\d{4}年(?:\d{1,2}月?|Q[1-4]|第[一二三四]季度))"
)

# 百分比: 5% / 2.5%
PERCENTAGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?%)")

# 金额: 412.56万亿元 / 300亿元 / 500万
AMOUNT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?(?:万亿|亿|万|千)元)"
)

# 机构名称
ORGANIZATION_KEYWORDS = [
    "银保监会", "金融监督管理总局", "国家金融监督管理总局",
    "人民银行", "中国人民银行", "央行",
    "国务院", "财政部", "证监会", "保监会",
]

# 指标名称关键词
METRIC_KEYWORDS = [
    "资本充足率", "核心一级资本充足率", "一级资本充足率",
    "杠杆率", "拨备覆盖率", "拨贷比", "不良贷款率",
    "流动性覆盖率", "流动性比例", "净稳定资金比例",
    "存贷比", "资产利润率", "资本利润率",
    "核心一级资本", "一级资本", "总资本", "风险加权资产",
    "储备资本", "逆周期资本", "附加资本", "系统重要性银行附加资本",
    "总资产", "总负债",
]

# 适用范围关键词
SCOPE_KEYWORDS = [
    "系统重要性银行", "非系统重要性银行",
    "国内系统重要性银行", "全球系统重要性银行",
    "大型商业银行", "中型银行", "小型银行",
    "城市商业银行", "农村商业银行", "民营银行",
    "外资银行", "政策性银行", "开发性金融机构",
    "全部", "所有银行",
]

# 术语定义: 什么是XXX / XXX是指
TERM_PATTERN_1 = re.compile(r"什么是(.+?)[？?。.\n]")
TERM_PATTERN_2 = re.compile(r"(.+?)是指")


class EntityExtractor:
    """
    规则版实体抽取器

    从用户问题中抽取文档名、条款号、指标名、术语等结构化实体。
    """

    def __init__(self):
        """初始化实体抽取器"""
        pass

    def extract(self, query: str) -> List[ExtractedEntity]:
        """
        从用户问题中抽取实体

        Args:
            query: 用户原始问题

        Returns:
            抽取的实体列表
        """
        if not query or not query.strip():
            return []

        entities: List[ExtractedEntity] = []
        query_stripped = query.strip()

        # 文档名
        for match in DOC_NAME_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="doc_name",
                value=match.group(1),
                raw_text=match.group(0),
                confidence=0.95,
            ))

        # 条款号
        for match in CLAUSE_NUMBER_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="clause_number",
                value=match.group(1),
                raw_text=match.group(0),
                confidence=0.90,
            ))

        # 章节号
        for match in CHAPTER_NUMBER_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="chapter_number",
                value=match.group(1),
                raw_text=match.group(0),
                confidence=0.85,
            ))

        # 附件号
        for match in ATTACHMENT_PATTERN.finditer(query_stripped):
            value = match.group(1) or match.group(2)
            entities.append(ExtractedEntity(
                entity_type="attachment_no",
                value=value,
                raw_text=match.group(0),
                confidence=0.85,
            ))

        # 表名
        for match in TABLE_NAME_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="table_name",
                value=match.group(0),
                raw_text=match.group(0),
                confidence=0.80,
            ))

        # 日期
        for match in DATE_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="date",
                value=match.group(1),
                raw_text=match.group(0),
                confidence=0.85,
            ))

        # 百分比
        for match in PERCENTAGE_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="percentage",
                value=match.group(1),
                raw_text=match.group(0),
                confidence=0.90,
            ))

        # 金额
        for match in AMOUNT_PATTERN.finditer(query_stripped):
            entities.append(ExtractedEntity(
                entity_type="amount",
                value=match.group(1),
                raw_text=match.group(0),
                confidence=0.85,
            ))

        # 机构名称（关键词匹配）
        for org in ORGANIZATION_KEYWORDS:
            if org in query_stripped:
                entities.append(ExtractedEntity(
                    entity_type="organization",
                    value=org,
                    raw_text=org,
                    confidence=0.85,
                ))

        # 指标名称（关键词匹配）
        for metric in METRIC_KEYWORDS:
            if metric in query_stripped:
                entities.append(ExtractedEntity(
                    entity_type="metric_name",
                    value=metric,
                    raw_text=metric,
                    confidence=0.85,
                ))

        # 适用范围（关键词匹配）
        for scope in SCOPE_KEYWORDS:
            if scope in query_stripped:
                entities.append(ExtractedEntity(
                    entity_type="scope",
                    value=scope,
                    raw_text=scope,
                    confidence=0.80,
                ))

        # 术语（什么是XXX / XXX是指）
        for match in TERM_PATTERN_1.finditer(query_stripped):
            term = match.group(1).strip()
            if term and len(term) <= 30:
                entities.append(ExtractedEntity(
                    entity_type="term",
                    value=term,
                    raw_text=match.group(0),
                    confidence=0.80,
                ))

        for match in TERM_PATTERN_2.finditer(query_stripped):
            term = match.group(1).strip()
            if term and len(term) <= 30:
                entities.append(ExtractedEntity(
                    entity_type="term",
                    value=term,
                    raw_text=match.group(0),
                    confidence=0.75,
                ))

        # 去重：同一 entity_type + value 只保留置信度最高的
        entities = self._deduplicate(entities)

        return entities

    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """去重：同一 entity_type + value 只保留置信度最高的"""
        seen: Dict[str, ExtractedEntity] = {}
        for entity in entities:
            key = f"{entity.entity_type}:{entity.value}"
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())
