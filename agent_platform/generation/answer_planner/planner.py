"""
回答规划器 — 根据意图和证据规划回答结构

根据查询意图类型和证据包，规划回答的章节结构、证据引用顺序、
所需引用数量和回答形态（文本摘录 / 表格数据 / 对比 / 合规判断）。

规划产物 AnswerPlan 会传递给生成器，指导 LLM 按结构组织回答。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# 意图 → 回答章节结构 映射
# 每个意图对应一组回答应当覆盖的章节
# ============================================================
INTENT_STRUCTURE: Dict[str, List[str]] = {
    # 条款查询：法条原文 + 适用范围 + 规范强度
    "clause_query": ["法条原文", "适用范围", "规范强度"],
    # 阈值查询：指标 + 要求 + 主体 + 时间 + 依据
    "threshold": ["指标名称", "最低/最高要求", "适用主体", "生效时间", "法规依据"],
    # 定义查询：定义内容 + 适用范围 + 来源
    "definition": ["定义内容", "适用范围", "来源"],
    # 表格取数：查询结果 + 数据来源
    "table_lookup": ["查询结果", "数据来源"],
    # 比较查询：对比维度 + 对象 + 结论
    "comparison": ["对比维度", "对象A", "对象B", "结论"],
    # 合规查询：判断 + 适用条款 + 结论
    "compliance": ["合规判断", "适用条款", "结论"],
    # 概览查询：概述 + 要点 + 来源
    "overview": ["概述", "要点", "来源"],
    # 未知意图：相关内容 + 来源
    "unknown": ["相关内容", "来源"],
}


# ============================================================
# 意图 → 回答形态 映射
# 决定生成器采用何种输出形态
# ============================================================
INTENT_ANSWER_SHAPE: Dict[str, str] = {
    "clause_query": "text_excerpt",      # 文本摘录
    "threshold": "text_excerpt",         # 文本摘录
    "definition": "text_excerpt",        # 文本摘录
    "table_lookup": "table_data",        # 表格数据
    "comparison": "comparison",          # 对比
    "compliance": "compliance_judgment", # 合规判断
    "overview": "text_excerpt",         # 文本摘录
    "unknown": "text_excerpt",           # 文本摘录
}


@dataclass
class AnswerPlan:
    """回答规划结果

    Attributes:
        intent: 查询意图
        structure: 回答应覆盖的章节标题列表
        evidence_order: 证据 ID 列表（按引用优先级排序，评分高者在前）
        required_citations: 需要标注的最少引用数量
        answer_shape: 回答形态（text_excerpt / table_data / comparison / compliance_judgment）
    """

    intent: str
    structure: List[str] = field(default_factory=list)
    evidence_order: List[str] = field(default_factory=list)
    required_citations: int = 0
    answer_shape: str = "text_excerpt"

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "intent": self.intent,
            "structure": list(self.structure),
            "evidence_order": list(self.evidence_order),
            "required_citations": self.required_citations,
            "answer_shape": self.answer_shape,
        }


class AnswerPlanner:
    """
    回答规划器

    根据查询意图和证据包，规划回答的结构与引用策略。
    规划逻辑为纯规则驱动，不依赖 LLM。
    """

    def plan(
        self,
        intent: str,
        evidence_bundle: Any,
        query_spec: Optional[Dict[str, Any]] = None,
    ) -> AnswerPlan:
        """
        规划回答结构

        Args:
            intent: 查询意图（如 clause_query / threshold / definition 等）
            evidence_bundle: EvidenceBundle 对象，可为 None
            query_spec: 查询规格（可选，用于后续细化规划）

        Returns:
            AnswerPlan 对象
        """
        # 确定回答章节结构
        structure = list(INTENT_STRUCTURE.get(intent, INTENT_STRUCTURE["unknown"]))

        # 确定回答形态
        answer_shape = INTENT_ANSWER_SHAPE.get(intent, "text_excerpt")

        # 按评分降序排列证据，确定引用顺序
        evidence_order: List[str] = []
        if evidence_bundle is not None and getattr(evidence_bundle, "evidence_items", None):
            sorted_items = sorted(
                evidence_bundle.evidence_items,
                key=lambda e: getattr(e, "score", 0.0),
                reverse=True,
            )
            evidence_order = [getattr(e, "evidence_id", "") for e in sorted_items]

        # 所需引用数量上限为 5，避免过度堆砌引用
        required_citations = min(len(evidence_order), 5)

        return AnswerPlan(
            intent=intent,
            structure=structure,
            evidence_order=evidence_order,
            required_citations=required_citations,
            answer_shape=answer_shape,
        )
