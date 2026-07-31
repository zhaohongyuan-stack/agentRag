"""
回答生成器 — Phase 1 模板版

基于证据包（EvidenceBundle）生成回答，不调用真实 LLM。
使用预定义模板，根据意图类型组装回答文本和引用。

Phase 2 会替换为真实 LLM（DeepSeek）生成。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_platform.generation.answer_planner import AnswerPlanner
from agent_platform.generation.citation_formatter import CitationFormatter
from agent_platform.runtime.llm_client import LLMMessage, get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """生成的回答"""

    answer_text: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    claims_with_evidence: List[Dict[str, Any]] = field(default_factory=list)
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer_text": self.answer_text,
            "citations": self.citations,
            "claims_with_evidence": self.claims_with_evidence,
            "is_refusal": self.is_refusal,
            "refusal_reason": self.refusal_reason,
            "confidence": self.confidence,
        }


# ============================================================
# 回答模板
# ============================================================
ANSWER_TEMPLATES = {
    "clause_query": (
        "根据{citation}的内容：\n\n{evidence_snippet}\n\n"
        "该条款的适用范围为{applicable_scope}，规范强度为{normative_level}。"
    ),
    "definition": (
        "{term}的定义如下：\n\n{evidence_snippet}\n\n"
        "来源：{citation}"
    ),
    "threshold": (
        "根据{citation}的规定：\n\n{evidence_snippet}\n\n"
        "该要求适用于{applicable_scope}。"
    ),
    "table_lookup": (
        "查询结果如下：\n\n{evidence_snippet}\n\n"
        "数据来源：{citation}"
    ),
    "comparison": (
        "根据检索到的证据，对比如下：\n\n{evidence_summary}\n\n"
        "来源：{citations}"
    ),
    "compliance": (
        "根据{citation}的规定，{evidence_snippet}\n\n"
        "请结合具体业务场景判断合规性。"
    ),
    "overview": (
        "根据检索到的资料，{evidence_summary}\n\n"
        "来源：{citations}"
    ),
    "greeting": "您好！我是银行业法规智能问答助手，可以帮您查询银行监管法规、资本充足率要求、条款内容等问题。请问您想了解什么？",
    "unknown": (
        "根据检索到的相关资料：\n\n{evidence_summary}\n\n"
        "来源：{citations}"
    ),
}

REFUSAL_TEMPLATE = (
    "抱歉，当前检索到的证据不足以回答您的问题。"
    "已尝试检索但未找到充分的相关信息。"
    "{missing_conditions}"
    "建议您：\n"
    "1. 提供更具体的问题描述（如文档名称、条款号）\n"
    "2. 明确查询的指标名称或适用范围\n"
    "3. 确认问题是否属于现行有效法规的范畴"
)

CLARIFICATION_TEMPLATE = (
    "您的问题存在歧义，需要进一步澄清：\n\n"
    "{ambiguity_descriptions}\n\n"
    "请补充以下信息以便更准确地回答您的问题。"
)


class TemplateGenerator:
    """
    模板回答生成器（Phase 1）

    基于证据包和查询意图，使用预定义模板生成回答。
    不依赖 LLM，用于验证流程设计和状态机正确性。
    """

    def __init__(self):
        pass

    def generate(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str = "",
        ambiguities: Optional[List[Dict[str, Any]]] = None,
    ) -> GeneratedAnswer:
        """
        生成回答

        Args:
            intent: 查询意图
            evidence_bundle: EvidenceBundle 对象
            query_text: 原始查询文本
            ambiguities: 歧义列表

        Returns:
            GeneratedAnswer 对象
        """
        # 问候直接返回
        if intent == "greeting":
            return GeneratedAnswer(
                answer_text=ANSWER_TEMPLATES["greeting"],
                confidence=1.0,
            )

        # 证据不足 → 拒答
        if evidence_bundle is None or evidence_bundle.evidence_count == 0:
            return self._generate_refusal(evidence_bundle)

        if not evidence_bundle.is_sufficient:
            return self._generate_refusal(evidence_bundle)

        # 有充分证据 → 模板生成
        return self._generate_from_template(intent, evidence_bundle, query_text)

    def generate_clarification(
        self, ambiguities: List[Dict[str, Any]]
    ) -> GeneratedAnswer:
        """
        生成澄清请求

        Args:
            ambiguities: 歧义列表

        Returns:
            GeneratedAnswer 对象
        """
        descriptions = []
        for amb in ambiguities:
            desc = amb.get("description", "")
            resolution = amb.get("resolution", "")
            descriptions.append(f"- {desc}")
            if resolution:
                descriptions.append(f"  建议：{resolution}")

        answer_text = CLARIFICATION_TEMPLATE.format(
            ambiguity_descriptions="\n".join(descriptions)
        )

        return GeneratedAnswer(
            answer_text=answer_text,
            is_refusal=False,
            confidence=0.0,
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _generate_from_template(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str,
    ) -> GeneratedAnswer:
        """使用模板生成回答"""
        template = ANSWER_TEMPLATES.get(intent, ANSWER_TEMPLATES["unknown"])

        # 取评分最高的证据
        top_evidence = max(
            evidence_bundle.evidence_items,
            key=lambda e: e.score,
        ) if evidence_bundle.evidence_items else None

        if not top_evidence:
            return self._generate_refusal(evidence_bundle)

        # 构建引用
        citations = self._build_citations(evidence_bundle)

        # 构建证据摘要
        evidence_summary = self._build_evidence_summary(evidence_bundle)

        # 填充模板
        try:
            if intent == "comparison" or intent == "overview" or intent == "unknown":
                answer_text = template.format(
                    evidence_summary=evidence_summary,
                    citations="；".join(c["citation"] for c in citations[:3]),
                )
            else:
                answer_text = template.format(
                    citation=top_evidence.citation or "相关法规",
                    evidence_snippet=top_evidence.evidence_snippet,
                    applicable_scope=top_evidence.metadata.get("applicable_scope", "全部"),
                    normative_level=top_evidence.metadata.get("normative_level", "neutral"),
                    term=query_text.replace("什么是", "").replace("？", "").replace("?", "").strip(),
                )
        except KeyError:
            answer_text = f"根据检索到的资料：\n\n{evidence_summary}\n\n来源：{'; '.join(c['citation'] for c in citations[:3])}"

        # 构建声明-证据对齐
        claims_with_evidence = self._build_claims_evidence(evidence_bundle)

        # 置信度基于证据充分性评分
        confidence = min(evidence_bundle.sufficiency_score, 1.0)

        return GeneratedAnswer(
            answer_text=answer_text,
            citations=citations,
            claims_with_evidence=claims_with_evidence,
            confidence=confidence,
        )

    def _generate_refusal(self, evidence_bundle: Any) -> GeneratedAnswer:
        """生成拒答回答"""
        missing = ""
        if evidence_bundle and evidence_bundle.missing_conditions:
            missing = f"\n\n缺失的条件包括：{', '.join(evidence_bundle.missing_conditions)}。"

        answer_text = REFUSAL_TEMPLATE.format(missing_conditions=missing)

        return GeneratedAnswer(
            answer_text=answer_text,
            is_refusal=True,
            refusal_reason="证据不足",
            confidence=0.0,
        )

    def _build_citations(self, evidence_bundle: Any) -> List[Dict[str, str]]:
        """构建引用列表"""
        citations = []
        seen = set()
        for ev in evidence_bundle.evidence_items:
            if ev.citation and ev.citation not in seen:
                citations.append({
                    "citation": ev.citation,
                    "source_doc": ev.source_doc,
                    "hierarchy_path": ev.hierarchy_path,
                    "chunk_id": ev.chunk_id,
                })
                seen.add(ev.citation)
        return citations

    def _build_evidence_summary(self, evidence_bundle: Any) -> str:
        """构建证据摘要"""
        summaries = []
        for ev in evidence_bundle.evidence_items[:5]:
            snippet = ev.evidence_snippet[:150]
            if len(ev.evidence_snippet) > 150:
                snippet += "..."
            summaries.append(f"- [{ev.citation}] {snippet}")

        return "\n\n".join(summaries) if summaries else "未找到相关证据"

    def _build_claims_evidence(self, evidence_bundle: Any) -> List[Dict[str, Any]]:
        """构建声明-证据对齐列表"""
        result = []
        for claim in evidence_bundle.claim_slots:
            result.append({
                "claim_id": claim.claim_id,
                "description": claim.description,
                "status": claim.status,
                "evidence_ids": claim.evidence_ids,
            })
        return result


# ============================================================
# 基于 LLM 的回答生成器（Phase 2）
# ============================================================


class GroundedGenerator:
    """
    基于 LLM 的接地回答生成器（Phase 2）

    使用 LLMClient 调用 DeepSeek 生成回答，结合 AnswerPlanner 规划回答结构、
    CitationFormatter 格式化引用来源。

    降级策略:
      - 当 LLM 处于 Mock 模式（无 API Key）时，回退到 TemplateGenerator
      - 当 LLM 调用抛出异常时，回退到 TemplateGenerator
    以保证在任何环境下都能输出可用回答。
    """

    # 系统提示：接地回答规则
    SYSTEM_PROMPT = (
        "你是一个银行业监管数据问答助手。请严格按照以下规则回答用户问题：\n"
        "\n"
        "## 核心原则\n"
        "1. 只能使用 evidence_items 中提供的证据，不得编造或添加外部信息\n"
        "2. 数字、日期、比例必须与证据 content 字段中的原文完全一致\n"
        "3. 如果证据不足以回答问题，直接回答\"依据不足\"\n"
        "\n"
        "## 回答结构\n"
        "- 先给出直接答案（一句话回答用户问题中的核心数值或结论）\n"
        "- 再补充必要的上下文说明（1-2句）\n"
        "- 最后标注引用来源，格式：来源：《文档名》Sheet/Cell/Row 信息\n"
        "\n"
        "## 按意图回答的指引\n"
        "- table_lookup（表格查值）：从证据 content 中提取确切数值，明确给出\"XX = 31739.18亿元\"这样的直接答案，不要只引用单元格位置而不给数值\n"
        "- clause_query（条款查询）：引用条款原文关键内容，标注条款编号\n"
        "- threshold（阈值查询）：明确给出阈值数值和适用条件\n"
        "- definition（定义查询）：引用定义原文\n"
        "\n"
        "## 输出格式（JSON）\n"
        "{\n"
        '  "answer": "直接答案 + 简要说明 + 引用来源",\n'
        '  "is_refusal": false,\n'
        '  "confidence": 0.95,\n'
        '  "citations": [{"index": 1, "citation": "来源描述"}]\n'
        "}\n"
        "\n"
        "## 关键提醒\n"
        "- evidence_items 中的 content 字段包含了具体数值，如\"原保险保费收入=31739.18\"，必须提取并体现在回答中\n"
        "- 不要只返回单元格定位信息（如 Range=A5:B5）而不给出实际数值\n"
        "- 回答必须用中文"
    )

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        answer_planner: Optional[AnswerPlanner] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        template_generator: Optional[TemplateGenerator] = None,
    ):
        """
        Args:
            llm_client: LLM 客户端，None 时使用全局单例
            answer_planner: 回答规划器，None 时新建默认实例
            citation_formatter: 引用格式化器，None 时新建默认实例
            template_generator: 模板生成器（降级兜底），None 时新建默认实例
        """
        self._llm_client = llm_client or get_llm_client()
        self._answer_planner = answer_planner or AnswerPlanner()
        self._citation_formatter = citation_formatter or CitationFormatter()
        self._template_generator = template_generator or TemplateGenerator()

    @property
    def is_mock(self) -> bool:
        """LLM 客户端是否处于 Mock 模式"""
        return bool(getattr(self._llm_client, "is_mock", True))

    def generate(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str = "",
        ambiguities: Optional[List[Dict[str, Any]]] = None,
    ) -> GeneratedAnswer:
        """
        生成回答

        Args:
            intent: 查询意图
            evidence_bundle: EvidenceBundle 对象
            query_text: 原始查询文本
            ambiguities: 歧义列表（保留接口，暂不在此处理）

        Returns:
            GeneratedAnswer 对象
        """
        # 问候意图直接复用模板
        if intent == "greeting":
            return self._template_generator.generate(
                intent, evidence_bundle, query_text
            )

        # 证据不足 → 拒答
        if (
            evidence_bundle is None
            or evidence_bundle.evidence_count == 0
            or not evidence_bundle.is_sufficient
        ):
            return self._generate_refusal(evidence_bundle)

        # Mock 模式 → 回退到模板生成器
        if self.is_mock:
            logger.info("LLM 处于 Mock 模式，回退到模板生成器")
            return self._template_generator.generate(
                intent, evidence_bundle, query_text
            )

        # LLM 生成路径
        try:
            return self._generate_with_llm(intent, evidence_bundle, query_text)
        except Exception as e:
            logger.warning(f"LLM 生成失败，回退到模板生成器: {e}", exc_info=True)
            return self._template_generator.generate(
                intent, evidence_bundle, query_text
            )

    def generate_clarification(
        self, ambiguities: List[Dict[str, Any]]
    ) -> GeneratedAnswer:
        """
        生成澄清请求（与模板版一致）

        Args:
            ambiguities: 歧义列表

        Returns:
            GeneratedAnswer 对象
        """
        return self._template_generator.generate_clarification(ambiguities)

    # ============================================================
    # 内部方法
    # ============================================================

    def _generate_with_llm(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str,
    ) -> GeneratedAnswer:
        """使用 LLM 生成回答"""
        # 1. 规划回答结构
        plan = self._answer_planner.plan(intent, evidence_bundle)

        # 2. 构建证据 JSON
        evidence_json = [
            self._evidence_to_dict(ev) for ev in evidence_bundle.evidence_items
        ]

        # 3. 构建用户提示（JSON 形式，包含问题、证据、规划）
        user_prompt = json.dumps(
            {
                "question": query_text,
                "evidence_items": evidence_json,
                "answer_plan": plan.to_dict(),
            },
            ensure_ascii=False,
        )

        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        # 4. 调用 LLM 并解析 JSON 响应
        result = self._llm_client.chat_json(messages=messages, temperature=0.1)

        answer_text = result.get("answer", "")
        is_refusal = result.get("is_refusal", False)
        confidence = float(result.get("confidence", 0.0))

        # 5. LLM 判定证据不足或空回答 → 拒答
        if is_refusal or not answer_text:
            return self._generate_refusal(evidence_bundle)

        # 6. 用 CitationFormatter 格式化引用
        citations = self._citation_formatter.format_citation_list(
            evidence_bundle.evidence_items
        )

        # 7. 声明-证据对齐（复用模板生成器逻辑）
        claims_with_evidence = self._template_generator._build_claims_evidence(
            evidence_bundle
        )

        return GeneratedAnswer(
            answer_text=answer_text,
            citations=citations,
            claims_with_evidence=claims_with_evidence,
            is_refusal=False,
            confidence=confidence,
        )

    def _generate_refusal(self, evidence_bundle: Any) -> GeneratedAnswer:
        """生成拒答回答"""
        missing = ""
        if evidence_bundle and evidence_bundle.missing_conditions:
            missing = (
                f"\n\n缺失的条件包括：{', '.join(evidence_bundle.missing_conditions)}。"
            )

        answer_text = REFUSAL_TEMPLATE.format(missing_conditions=missing)

        return GeneratedAnswer(
            answer_text=answer_text,
            is_refusal=True,
            refusal_reason="证据不足",
            confidence=0.0,
        )

    def _evidence_to_dict(self, ev: Any) -> dict:
        """将 EvidenceItem 转换为可序列化字典（含 metadata）"""
        if isinstance(ev, dict):
            return ev
        return {
            "evidence_id": getattr(ev, "evidence_id", ""),
            "chunk_id": getattr(ev, "chunk_id", ""),
            "content": getattr(ev, "content", ""),
            "evidence_snippet": getattr(ev, "evidence_snippet", ""),
            "citation": getattr(ev, "citation", ""),
            "score": getattr(ev, "score", 0.0),
            "source_doc": getattr(ev, "source_doc", ""),
            "hierarchy_path": getattr(ev, "hierarchy_path", ""),
            "chunk_type": getattr(ev, "chunk_type", ""),
            "metadata": getattr(ev, "metadata", {}) or {},
        }
