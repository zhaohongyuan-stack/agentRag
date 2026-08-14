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
        "## 表格分区识别规则（极其重要）\n"
        "Excel 表格通常包含多个分区（如\"1. 银行业金融机构\"、\"其中：商业银行合计\"、\n"
        "\"2. 大型商业银行\"、\"3. 股份制商业银行\"、\"4. 城市商业银行\"、\"5. 农村金融机构\"、\n"
        "\"6. 其他类金融机构\"等），每个分区都有相同的指标（如\"总资产\"、\"总负债\"），但数值不同。\n"
        "1. 当问题提到\"银行业\"总资产/总负债时，必须选择\"1. 银行业金融机构\"分区的数据\n"
        "2. 检查 evidence_items 中每条证据的 table_name 字段，确认其所属表格分区\n"
        "3. 同一指标在不同分区有不同数值时，必须选择与问题最匹配的分区\n"
        "4. 绝对不能混用不同分区的数据（如用\"其中：商业银行合计\"的Q1值减去\n"
        "   \"1. 银行业金融机构\"的Q4值）\n"
        "5. 在回答中必须标注所用数据的表格分区名称\n"
        "6. 涉及两处取数计算时，两个数值必须来自同一分区\n"
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
        "## 数值计算方向规则（极其重要）\n"
        '当问题涉及两个数值之间的"变化"、"差值"、"增减"时，必须严格按照以下规则确定减法顺序：\n'
        '1. "从A到B的变化" = B - A（终点值减起点值，不是A减B）\n'
        '   示例："从大型商业银行到外资银行的变化" = 外资银行值 - 大型商业银行值\n'
        '2. "A相比B的变化" = A - B（前者减后者）\n'
        '3. "A较B的增减" = A - B\n'
        '4. "从第一季度到第四季度的变化" = 第四季度值 - 第一季度值\n'
        "5. 必须在回答中先写出计算公式（如：外资银行 - 大型商业银行 = 121.77 - 12461.06 = -12339.29），再给出最终答案\n"
        "6. 如果计算结果为负数，必须保留负号，不要取绝对值\n"
        "7. 涉及多步计算时，先列出每一步的公式和结果\n"
        "\n"
        "## 准确性判断规则（\"准确吗\"/\"对吗\"/\"正确吗\"类问题）\n"
        "当用户询问某个表述是否准确时，按以下原则判断：\n"
        "1. 核心判断标准：表述的【核心事实】是否正确，而非【边界精度】是否完全一致\n"
        "   - 核心事实包括：数值量级、主体名称、适用范围的大区间、计算方向\n"
        "   - 边界精度包括：开闭区间（<, ≤, >, ≥）、四舍五入、近似值\n"
        "2. 口语化表述等同原则（极其重要）：\n"
        "   - 用户说\"XX年到XX年\"时，等同于原文中该范围的任何区间表达（含或不含端点均视为准确）\n"
        "   - 即\"20年到40年\" = 20<t≤40 = 20≤t≤40 = 20<t<40，均判定为准确\n"
        "   - 只有当用户说的范围与原文范围【完全不对应】时才判定为不准确\n"
        "3. 准确性判断示例：\n"
        "   原文：基础利率曲线由三段组成，0<t≤20为750日移动平均国债收益率曲线，20<t≤40为终极利率过渡曲线，t>40为终极利率\n"
        "   - \"终极利率过渡曲线适用于20年到40年\" → 准确（区间范围正确，边界差异不影响核心事实）\n"
        "   - \"终极利率过渡曲线适用于0年到20年\" → 不准确（区间范围错误，0-20年属于750日移动平均国债收益率曲线）\n"
        "   - \"750日移动平均国债收益率曲线适用于0到20年\" → 准确（区间范围正确）\n"
        "4. 回答格式：\n"
        "   - 准确时：\"准确。根据《文档名》，[补充原文精确表述]\"\n"
        "   - 不准确时：\"不准确。[指出具体错误并给出正确答案]\"\n"
        "\n"
        "## 输出格式（JSON）\n"
        "{\n"
        '  "answer": "直接答案 + 简要说明 + 引用来源",\n'
        '  "is_refusal": false,\n'
        '  "citations": [{"index": 1, "citation": "来源描述"}]\n'
        "}\n"
        "\n"
        "## 关键提醒\n"
        "- evidence_items 中的 content 字段包含了具体数值，如\"原保险保费收入=31739.18\"，必须提取并体现在回答中\n"
        "- 不要只返回单元格定位信息（如 Range=A5:B5）而不给出实际数值\n"
        "- 回答必须用中文"
    )

    # 数值计算型选项处理补充
    SYSTEM_PROMPT_NUMERIC = (
        "\n\n## 选项处理规则（数值计算型）\n"
        "你将收到若干数值选项（A/B/C/D）和一个不含选项的问题。\n"
        "1. 先基于 evidence_items 中的证据独立计算答案，不要试图匹配任何选项\n"
        "2. 计算完成后再将结果与选项对比，选择数值最接近的选项\n"
        "3. 如果计算结果与所有选项差距很大（超过最大选项值的10%），\n"
        "   说明可能取错了证据，重新检查 evidence_items 中的 table_name 和单元格位置\n"
        "4. 绝对不要为了匹配某个选项而调整计算方法或选择不同分区的数据\n"
        "5. 输出格式：\n"
        "   计算公式：X - Y = Z\n"
        "   最接近选项：选项X（Z亿元）\n"
    )

    # 事实判断型选项处理补充
    SYSTEM_PROMPT_FACTUAL = (
        "\n\n## 选项处理规则（事实判断型）\n"
        "你是一个银行业监管数据事实核查员。\n"
        "你将收到一个问题和若干表述选项（A/B/C/D）。\n\n"
        "请依据 evidence_items 中的证据，逐条判断每个选项的表述是否符合事实：\n"
        "1. 对每个选项，在证据中搜索是否有直接支持该表述的内容\n"
        "2. 如果证据中明确支持该选项的表述，标记为\"支持\"，并引用对应原文\n"
        "3. 如果证据中明确反对该选项（即证据说明实际情况与选项相反），标记为\"反对\"，并引用对应原文\n"
        "4. 如果证据中没有找到与该选项相关的内容，标记为\"证据不足\"\n"
        "5. 最终选择唯一一个标记为\"支持\"的选项作为答案\n"
        "6. 如果多个选项都被标记为\"支持\"，选择证据最直接、最明确的那个\n"
        "7. 如果没有任何选项被\"支持\"，回答\"依据不足，未能找到支持任何选项的证据\"\n\n"
        "## 文档范围严格匹配规则（极其重要）\n"
        "当问题包含以下表述时，表示用户要求依据特定文档回答：\n"
        '  - "根据《XXX》..."\n'
        '  - "检索《XXX》后..."\n'
        '  - "依据《XXX》..."\n'
        '  - "查阅《XXX》..."\n'
        "此时必须严格遵守：\n"
        "1. 只能选择证据 source_doc 或 doc_name 字段包含该文档名（或其核心部分）的选项作为答案\n"
        "2. 即使其他选项在事实层面正确，但如果其证据来自不同文档，必须标记为\"证据不足（文档不匹配）\"\n"
        "3. 如果所有选项的证据都来自不同文档，回答\"依据不足，指定文档《XXX》的内容未在证据中找到\"\n"
        "4. 文档名匹配时允许模糊匹配：用户说的文档名可能是实际文档名的简称或核心部分\n"
        "   例如：用户说《银行函证工作操作指引》，实际文档名为\"398_财政部办公厅_..._银行函证工作操作指引.pdf\"\n"
        "   只要 source_doc 或 doc_name 中包含用户指定的文档名核心部分，即视为匹配\n"
        "5. 特别注意：不要因为某个选项的内容在其他文档中也有提及就标记为\"支持\"，\n"
        "   必须确认证据确实来自用户指定的文档\n\n"
        "输出格式：\n"
        "选项A：支持/反对/证据不足 — 依据：[引用原文片段]\n"
        "选项B：支持/反对/证据不足 — 依据：[引用原文片段]\n"
        "...\n"
        "最终答案：选项X\n"
        "引用来源：《文档名》Sheet/Cell/Row 信息\n"
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
        options: Optional[List[Dict[str, Any]]] = None,
        prompt_mode: str = "normal",
    ) -> GeneratedAnswer:
        """
        生成回答

        Args:
            intent: 查询意图
            evidence_bundle: EvidenceBundle 对象
            query_text: 原始查询文本（已去除选项的纯净问题）
            ambiguities: 歧义列表（保留接口，暂不在此处理）
            options: 选项列表，格式 [{"label": "A", "text": "...", "type": "numeric/textual"}]
            prompt_mode: 生成模式 "normal" | "calculate_and_match" | "verify_each_option"

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
            return self._generate_with_llm(intent, evidence_bundle, query_text, options, prompt_mode)
        except Exception as e:
            logger.warning(f"LLM 生成失败，尝试规则计算兜底: {e}", exc_info=True)
            # 数值计算选择题（LLM 不可用时的可靠性保障）：
            # 从表格证据解析「指标=数值」映射，枚举跨期差值并与选项比对
            if prompt_mode == "calculate_and_match":
                ruled = self._rule_calculate_choice(intent, evidence_bundle, query_text, options)
                if ruled is not None:
                    logger.info("[RuleCalc] LLM失败，规则计算兜底成功")
                    return ruled
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

    def generate_stream(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str,
        options: Optional[List[Dict[str, Any]]] = None,
        prompt_mode: str = "normal",
        on_thinking=None,
    ):
        """
        流式生成回答（生成器）。
        yield token 字符串供前端实时渲染，return GeneratedAnswer。
        调用方用法:
            gen = generator.generate_stream(...)
            try:
                while True:
                    token = next(gen)
                    callback.on_answer_token(token)
            except StopIteration as e:
                answer = e.value  # GeneratedAnswer
        """
        # 1. 构建证据 JSON
        evidence_json = [
            self._evidence_to_dict(ev) for ev in evidence_bundle.evidence_items
        ]

        # 2. 流式专用 system prompt（纯文本输出，非JSON）
        system_prompt = (
            self.SYSTEM_PROMPT
            + "\n\n## 输出要求\n"
            "直接输出回答文本，不要使用JSON格式。\n"
            "回答要简洁、准确，引用证据来源。\n"
            "如果证据不足，直接说明缺少什么数据。"
        )
        if prompt_mode == "calculate_and_match":
            system_prompt += (
                "\n\n## 数值计算题\n"
                "从证据中找到所需数值，计算后给出答案和最接近的选项。\n"
                "格式：计算过程 + 答案 + 选项。"
            )
        elif prompt_mode == "verify_each_option":
            system_prompt += "\n\n## 事实核查题\n逐项验证选项正确性。"

        # 3. 构建用户提示
        user_prompt_data = {
            "question": query_text,
            "evidence_items": evidence_json,
        }
        if options:
            user_prompt_data["options"] = options
        user_prompt = json.dumps(user_prompt_data, ensure_ascii=False)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        # 4. 流式调用 LLM，区分 thinking/token
        full_text = []
        try:
            for item in self._llm_client.chat_stream(
                messages=messages, temperature=0.1
            ):
                if isinstance(item, tuple) and item[0] == "thinking":
                    # 思维链内容，通过回调推送（不混入回答文本）
                    if on_thinking:
                        on_thinking(item[1])
                else:
                    token = item[1] if isinstance(item, tuple) else item
                    full_text.append(token)
                    yield token
        except Exception as e:
            logger.warning(f"[Generator] 流式生成失败: {e}")
            raise

        # 5. 收集完整文本，构造 GeneratedAnswer
        answer_text = "".join(full_text).strip()
        confidence = min(evidence_bundle.sufficiency_score, 1.0)

        if not answer_text:
            return self._generate_refusal(evidence_bundle)

        citations = self._citation_formatter.format_citation_list(
            evidence_bundle.evidence_items
        )
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

    def _generate_with_llm(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str,
        options: Optional[List[Dict[str, Any]]] = None,
        prompt_mode: str = "normal",
    ) -> GeneratedAnswer:
        """使用 LLM 生成回答"""
        # 1. 规划回答结构
        plan = self._answer_planner.plan(intent, evidence_bundle)

        # 2. 构建证据 JSON
        evidence_json = [
            self._evidence_to_dict(ev) for ev in evidence_bundle.evidence_items
        ]

        # 3. 选择 System Prompt（根据 prompt_mode 追加选项处理指引）
        system_prompt = self.SYSTEM_PROMPT
        if prompt_mode == "calculate_and_match":
            system_prompt = self.SYSTEM_PROMPT + self.SYSTEM_PROMPT_NUMERIC
        elif prompt_mode == "verify_each_option":
            system_prompt = self.SYSTEM_PROMPT + self.SYSTEM_PROMPT_FACTUAL

        # 4. 构建用户提示（JSON 形式，包含问题、证据、规划、选项）
        user_prompt_data = {
            "question": query_text,
            "evidence_items": evidence_json,
            "answer_plan": plan.to_dict(),
        }
        if options:
            user_prompt_data["options"] = options

        user_prompt = json.dumps(user_prompt_data, ensure_ascii=False)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        # 4. 调用 LLM 并解析 JSON 响应
        result = self._llm_client.chat_json(messages=messages, temperature=0.1)

        answer_text = result.get("answer", "")
        is_refusal = result.get("is_refusal", False)

        # 置信度由后端基于证据充分性评分动态计算
        # 不使用 LLM 输出的 confidence（LLM 容易照抄 prompt 中的示例值）
        confidence = min(evidence_bundle.sufficiency_score, 1.0)

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

    def _rule_calculate_choice(
        self,
        intent: str,
        evidence_bundle: Any,
        query_text: str,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[GeneratedAnswer]:
        """
        规则计算兜底（LLM 不可用时）：
        从表格证据中按「分区+指标」分组解析数值，枚举同组跨期差值，
        优先匹配问题中的分区关键词，再与数值选项比对选出最接近答案。
        仅适用于数值计算型选择题。失败返回 None（调用方回退模板生成器）。
        """
        import re as _re

        try:
            if not options:
                return None

            # 1. 收集数值选项
            opt_vals = []  # (label, value)
            for opt in options:
                text = str(opt.get("text", "")).replace(",", "").replace("，", "")
                m = _re.search(r"[-+]?\d+(?:\.\d+)?", text)
                if m:
                    opt_vals.append((opt.get("label", "?"), float(m.group())))
            if len(opt_vals) < 2:
                return None

            # 查询中的分区关键词偏好（与表格分区名匹配）
            _PARTITIONS = [
                ("银行业", "1. 银行业金融机构"),
                ("大型", "2. 大型商业银行"),
                ("股份制", "3. 股份制商业银行"),
                ("城市", "4. 城市商业银行"),
                ("农村", "5. 农村金融机构"),
                ("外资", "6. 其他类金融机构"),
                ("商业银行", "其中：商业银行合计"),
            ]

            def _partition_bonus(table_name: str) -> bool:
                for kw, _part in _PARTITIONS:
                    if kw in query_text and _part in (table_name or ""):
                        return True
                return False

            # 2. 解析证据 -> groups[(table_name, label)][col] = value
            groups = {}
            for ev in evidence_bundle.evidence_items:
                content = getattr(ev, "content", "") or ""
                meta = getattr(ev, "metadata", {}) or {}
                label = None
                if getattr(ev, "chunk_type", "") == "table_row":
                    m = _re.search(r"时间_项目=([^\s；;]+)", content)
                    label = m.group(1) if m else None
                if not label:
                    label = meta.get("primary_label")
                if not label or label == "":
                    continue
                tname = meta.get("table_name", "") or ""
                pairs = _re.findall(
                    r"([\u4e00-\u9fffA-Za-z_0-9年月日季度]+)\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?)",
                    content,
                )
                col_vals = groups.setdefault((tname, label), {})
                for col, val in pairs:
                    col = col.strip()
                    if col in ("时间_项目",):
                        continue
                    try:
                        col_vals[col] = float(val.replace(",", ""))
                    except ValueError:
                        pass

            if not groups:
                return None

            # 3. 选择与问题指标匹配的组（归一化双向匹配 + 分区偏好）
            def _norm(s):
                # 归一化：去引号/标点/空白，去"其中:"等前缀
                s = _re.sub(r'["\"\'""'',。、；：\s]+', '', str(s))
                s = s.replace("其中：", "").replace("其中", "").strip()
                return s
            norm_query = _norm(query_text)
            def _label_match(label):
                if not label:
                    return False
                nl = _norm(label)
                if not nl:
                    return False
                # 双向子串（覆盖 label长query短 / query长label短 两种情况）
                if nl in norm_query or norm_query in nl:
                    return True
                # 去常见后缀token匹配（合计/总计/小计）
                for sep in ("合计", "总计", "小计"):
                    if nl.endswith(sep):
                        base = nl[:-len(sep)]
                        if base and base in norm_query:
                            return True
                return False
            matched = [k for k in groups if _label_match(k[1])]
            if matched:
                # 分区偏好优先
                pref = [k for k in matched if _partition_bonus(k[0])]
                pool = pref if pref else matched
                target_key = max(pool, key=lambda k: len(groups[k]))
            else:
                pref_keys = [k for k in groups if _partition_bonus(k[0])]
                # 数值列最多的组，避开明显非目标组
                ordered = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
                target_key = ordered[0]
                if pref_keys:
                    # 有分区偏好且该分区有>=2数值列，优先
                    for k in ordered:
                        if k in pref_keys and len(groups[k]) >= 2:
                            target_key = k
                            break
            if not target_key or len(groups.get(target_key, {})) < 2:
                return None

            col_vals = groups[target_key]
            label_name = target_key[1] or "该指标"

            # 4. 枚举同组列间差值，与选项比对选最接近
            best = None  # (err, opt_label, opt_val, start_col, end_col, diff)
            cols = list(col_vals.keys())
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if i == j:
                        continue
                    diff = round(col_vals[cols[j]] - col_vals[cols[i]], 4)
                    for opt_label, opt_val in opt_vals:
                        err = abs(diff - opt_val)
                        rel = err / (abs(opt_val) + 1e-9)
                        if rel < 0.25:
                            if best is None or err < best[0]:
                                best = (err, opt_label, opt_val, cols[i], cols[j], diff)
            if best is None:
                return None
            _, opt_label, opt_val, start_col, end_col, diff = best

            # 5. 构造回答（标注分区）
            srcs = []
            for ev in evidence_bundle.evidence_items[:3]:
                src = getattr(ev, "source_doc", "") or ""
                if src and src not in srcs:
                    srcs.append(src)
            src_text = "；".join(srcs[:2]) if srcs else "检索到的表格数据"
            start_show = f"{col_vals[start_col]:.4f}" if col_vals[start_col] % 1 else f"{col_vals[start_col]:.0f}"
            end_show = f"{col_vals[end_col]:.4f}" if col_vals[end_col] % 1 else f"{col_vals[end_col]:.0f}"
            diff_show = f"{diff:.4f}" if diff % 1 else f"{diff:.0f}"
            partition_str = f"，分区：{target_key[0]}" if target_key[0] else ""
            answer_text = (
                f"根据检索到的表格数据，{label_name}从{start_col}到{end_col}的变化为：\n"
                f"{end_show} - {start_show} = {diff_show}\n"
                f"最接近选项：选项{opt_label}（{opt_val}）{partition_str}\n\n"
                f"数据来源：{src_text}"
            )
            return GeneratedAnswer(
                answer_text=answer_text,
                citations=[],
                claims_with_evidence=[],
                is_refusal=False,
                confidence=0.9,
            )
        except Exception as e:
            logger.warning(f"[RuleCalc] 规则计算兜底失败: {e}")
            return None

    def _evidence_to_dict(self, ev: Any) -> dict:
        """将 EvidenceItem 转换为可序列化字典（含 metadata 和 table_name）"""
        if isinstance(ev, dict):
            # 如果已经是 dict（如从缓存反序列化），复制后移除 evidence_id
            result = dict(ev)
            result.pop("evidence_id", None)
            return result
        metadata = getattr(ev, "metadata", {}) or {}
        return {
            # evidence_id 已移除：避免 LLM 在输出中暴露内部 ID
            "chunk_id": getattr(ev, "chunk_id", ""),
            "content": getattr(ev, "content", ""),
            "evidence_snippet": getattr(ev, "evidence_snippet", ""),
            "citation": getattr(ev, "citation", ""),
            "score": getattr(ev, "score", 0.0),
            "source_doc": getattr(ev, "source_doc", ""),
            "hierarchy_path": getattr(ev, "hierarchy_path", ""),
            "chunk_type": getattr(ev, "chunk_type", ""),
            "table_name": metadata.get("table_name", ""),
            "metadata": metadata,
        }
