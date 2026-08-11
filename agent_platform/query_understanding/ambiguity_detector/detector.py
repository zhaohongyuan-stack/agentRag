"""
歧义检测器

检测用户问题中的歧义情况：
  - entity_ambiguous: 实体歧义（如 "那个规定" 指代不明）
  - scope_missing: 范围缺失（如 "比例是多少" 未说明哪个指标的比例）
  - version_unclear: 版本不明确（如 "之前的规定" 未指明具体版本）
  - term_polysemous: 术语多义（如 "资本" 可能指多种资本工具）
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..entity_extractor.extractor import ExtractedEntity


@dataclass
class Ambiguity:
    """检测到的歧义"""

    ambiguity_type: str
    description: str
    resolution: str = ""

    def to_dict(self) -> dict:
        return {
            "ambiguity_type": self.ambiguity_type,
            "description": self.description,
            "resolution": self.resolution,
        }


# ============================================================
# 歧义检测规则
# ============================================================

# 指代不明的代词
VAGUE_REFERENCES = [
    "那个", "那个规定", "那个文件", "那个制度",
    "之前的规定", "之前那个", "上次说的", "刚才提到的",
    "它", "它们", "这个", "这些",
]

# 范围缺失的指示词 — 询问比例/数值但未指明具体指标
SCOPE_MISSING_PATTERNS = [
    re.compile(r"(?:比例|比率|要求)(?:是多少|多少|几)"),
    re.compile(r"(?:标准|阈值)(?:是多少|多少|几)"),
    re.compile(r"那个.*(?:是多少|多少)"),
]

# 多义术语
POLYSEMOUS_TERMS = {
    "资本": "资本可能指核心一级资本、一级资本、总资本等不同层级",
    "准备金": "准备金可能指存款准备金、贷款损失准备金等",
    "拨备": "拨备可能指一般准备、专项准备、特种准备等",
    "流动性": "流动性可能指流动性覆盖率、流动性比例、净稳定资金比例等",
    "风险": "风险可能指信用风险、市场风险、操作风险等",
}


class AmbiguityDetector:
    """
    歧义检测器

    检测用户问题中的实体歧义、范围缺失、版本不明确、术语多义等情况。
    """

    def __init__(self):
        pass

    def detect(
        self,
        query: str,
        entities: Optional[List[ExtractedEntity]] = None,
        intent: Optional[str] = None,
    ) -> List[Ambiguity]:
        """
        检测用户问题中的歧义

        Args:
            query: 用户原始问题
            entities: 已抽取的实体列表
            intent: 已分类的意图

        Returns:
            检测到的歧义列表，空列表表示无歧义
        """
        if not query or not query.strip():
            return []

        ambiguities: List[Ambiguity] = []
        query_stripped = query.strip()
        entities = entities or []

        # 1. 指代不明检测
        ambiguities.extend(self._detect_vague_reference(query_stripped))

        # 2. 范围缺失检测
        ambiguities.extend(self._detect_scope_missing(query_stripped, entities, intent))

        # 3. 术语多义检测
        ambiguities.extend(self._detect_polysemous_terms(query_stripped, entities))

        # 4. 版本不明确检测
        ambiguities.extend(self._detect_version_unclear(query_stripped, entities))

        return ambiguities

    def _detect_vague_reference(self, query: str) -> List[Ambiguity]:
        """检测指代不明的代词"""
        results = []
        for ref in VAGUE_REFERENCES:
            if ref in query:
                results.append(Ambiguity(
                    ambiguity_type="entity_ambiguous",
                    description=f"问题中包含模糊指代 '{ref}'，无法确定具体指向的对象",
                    resolution=f"请明确 '{ref}' 指代的具体文档、条款或指标名称",
                ))
                break  # 只报告一个指代不明
        return results

    def _detect_scope_missing(
        self,
        query: str,
        entities: List[ExtractedEntity],
        intent: Optional[str],
    ) -> List[Ambiguity]:
        """检测范围缺失 — 询问数值/比例但未指明具体指标"""
        results = []

        # 如果意图是 threshold 但没有抽到 metric_name 实体
        if intent == "threshold":
            has_metric = any(e.entity_type == "metric_name" for e in entities)
            if not has_metric:
                for pattern in SCOPE_MISSING_PATTERNS:
                    if pattern.search(query):
                        results.append(Ambiguity(
                            ambiguity_type="scope_missing",
                            description="询问比例/阈值但未指明具体的指标名称",
                            resolution="请明确您想查询的具体指标，如'核心一级资本充足率'、'杠杆率'等",
                        ))
                        break

        return results

    def _detect_polysemous_terms(
        self,
        query: str,
        entities: List[ExtractedEntity],
    ) -> List[Ambiguity]:
        """检测多义术语"""
        results = []

        # 收集所有已抽取实体的值，用于排除已被更具体实体包含的多义词
        entity_values = [e.value for e in entities if e.value]

        for term, explanation in POLYSEMOUS_TERMS.items():
            # 检查是否单独出现（不是作为更长术语的一部分）
            if term in query:
                # 排除已经被更具体术语包含的情况
                # 如 "核心一级资本充足率" 已经包含 "资本"，但前者更具体
                has_more_specific = False

                # 检查其他多义词
                for other_term in POLYSEMOUS_TERMS:
                    if other_term != term and term in other_term and other_term in query:
                        has_more_specific = True
                        break

                # 检查已抽取的实体值（如 metric_name="核心一级资本充足率" 包含 "资本"）
                if not has_more_specific:
                    for ev_value in entity_values:
                        if term in ev_value and len(ev_value) > len(term):
                            has_more_specific = True
                            break

                if not has_more_specific:
                    results.append(Ambiguity(
                        ambiguity_type="term_polysemous",
                        description=f"术语 '{term}' 存在多义性：{explanation}",
                        resolution=f"请明确 '{term}' 的具体类型",
                    ))
                    break  # 只报告一个多义术语
        return results

    def _detect_version_unclear(
        self,
        query: str,
        entities: List[ExtractedEntity],
    ) -> List[Ambiguity]:
        """检测版本不明确"""
        results = []

        # 检查 "之前的规定" / "旧版" 等表述
        version_unclear_keywords = ["之前的规定", "旧版", "老版", "原来的", "以前的"]
        for kw in version_unclear_keywords:
            if kw in query:
                results.append(Ambiguity(
                    ambiguity_type="version_unclear",
                    description=f"问题中引用了 '{kw}'，但未指明具体版本",
                    resolution="请指明具体的版本号或发布日期",
                ))
                break

        return results
