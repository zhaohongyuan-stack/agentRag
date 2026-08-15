"""
Verifier Agent - 验证Agent

问题匹配校验 + 声明级验证 + 有限次补充检索闭环：
  0. 问题匹配校验：回答是否满足问题的形式要求（如 ABCD 选项、数值计算）
  1. 整体一致性检查：回答是否与证据一致
  2. 声明级验证：拆分回答为声明，逐个检查证据支撑
  3. 有限次补充检索：无证据声明触发补充检索（受预算控制）
  4. 标记不确定：仍无证据的声明标记"依据不足"
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_platform.runtime.llm_client import LLMClient, LLMMessage

from .base_agent import BaseAgent
from .agent_context import AgentContext, AgentResult

logger = logging.getLogger(__name__)


class VerifierAgent(BaseAgent):
    """
    验证Agent

    整体+局部结合的验证策略：
    先整体检查一致性，发现可疑声明再逐条验证。
    无证据声明可触发有限次补充检索（由handler控制Loop）。
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="Verifier", temperature=0.1, max_tokens=2048)

    def _build_prompt(self, context: AgentContext) -> List[LLMMessage]:
        """构建Verifier提示词"""
        system_prompt = (
            "你是一个银行业/保险业监管知识问答系统的验证Agent。\n"
            "你的职责分两部分：先校验回答是否符合问题的要求，再验证每个声明是否有证据支撑。\n\n"
            "## 第一步：问题要求匹配校验（必须执行）\n"
            "检查生成的回答是否满足问题的形式要求：\n"
            "- 若问题给出了选项（选项A/选项B/选项C/选项D 或 A.B.C.D 等），回答必须明确给出所选选项字母（如“答案：B”）；\n"
            "  只给出计算结果或描述而未指明选项字母的，判定为不符合，并在 issue 中说明应选择哪个选项及理由\n"
            "- 若问题要求数值计算/变化量，回答必须给出明确的数值结果与计算过程\n"
            "- 若问题针对特定文档/报表，回答必须基于该文档作答\n"
            "- 若问题要求判断对错（如“某某数值是否正确”），回答必须给出明确的判断结论\n\n"
            "## 第二步：声明级验证\n"
            "1. 将回答拆分为多个独立声明（每个含具体数值或结论的句子）\n"
            "2. 对每个声明，在证据中查找是否有直接支撑\n"
            "3. 有证据支撑的标记verified，无证据的标记unverified\n"
            "4. 对unverified声明，判断是否可能通过补充检索找到证据\n"
            "\n"
            "## 你必须输出以下JSON格式\n"
            "{\n"
            '  "query_conformance": {"conforms": true, "issue": ""},\n'
            '  "verified": false,\n'
            '  "claims": [\n'
            '    {"text": "声明内容", "status": "verified", "evidence_id": "ev-xxx"},\n'
            '    {"text": "声明内容", "status": "unverified", "reason": "未找到证据"}\n'
            '  ],\n'
            '  "needs_retry": true,\n'
            '  "retry_query": "补充检索的查询文本",\n'
            '  "unverified_count": 1\n'
            "}\n\n"
            "## 验证规则\n"
            "- 数值声明必须找到完全匹配的证据（数值、单位一致）\n"
            "- 条款引用必须找到对应条款号的证据\n"
            "- 推论性声明（如\"满足监管要求\"）需找到阈值依据\n"
            "- 如果声明是对证据原文的直接引用，标记verified\n"
            "- 由证据数值经加减乘除推导出的计算声明，若证据中有原始数值且算式正确，标记verified\n"
            "- needs_retry=true时必须提供retry_query\n"
            "- 所有声明都verified时needs_retry=false\n"
            "- query_conformance.conforms=false时，在issue中说明不符合之处及应如何回答\n"
        )

        # 构建回答和证据信息（证据做截断优化，避免超长导致 LLM 超时）
        answer_dict = {}
        if context.generated_answer:
            answer_dict = context.generated_answer.to_dict()

        bundle_dict = {}
        if context.evidence_bundle:
            bundle_dict = self._limited_bundle_dict(context.evidence_bundle)

        user_prompt = json.dumps({
            "query": context.query,
            "generated_answer": answer_dict,
            "evidence_bundle": bundle_dict,
        }, ensure_ascii=False)

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

    def _limited_bundle_dict(self, bundle) -> dict:
        """
        限制证据包大小，避免 LLM 输入过长导致超时：
          - 只保留评分最高的前 N 条证据
          - 每条证据 content/snippet 截断到固定长度
        """
        try:
            full = bundle.to_dict()
            items = full.get("evidence_items", [])
            # 按 score 降序取前 8 条
            items = sorted(
                items, key=lambda x: x.get("score", 0) or 0, reverse=True
            )[:8]
            # 截断长文本
            for it in items:
                content = it.get("content", "") or ""
                snippet = it.get("evidence_snippet", "") or ""
                if len(content) > 400:
                    it["content"] = content[:400] + "…[截断]"
                if len(snippet) > 250:
                    it["evidence_snippet"] = snippet[:250] + "…[截断]"
            full["evidence_items"] = items
            full["_note"] = "证据已截断（最多8条，每条400字符），如需完整证据请补充检索"
            return full
        except Exception:
            return bundle.to_dict() if hasattr(bundle, "to_dict") else {}

    def _parse_response(self, response: dict) -> Tuple[str, Dict[str, Any]]:
        """解析Verifier响应"""
        verified = response.get("verified", False)
        needs_retry = response.get("needs_retry", False)

        # 问题要求匹配校验结果（缺失时默认符合，向后兼容）
        conformance = response.get("query_conformance") or {}
        conforms = conformance.get("conforms", True)
        conformance_issue = (conformance.get("issue") or "").strip()

        claims = list(response.get("claims", []))
        # 不符合问题要求时，以显式声明的形式暴露给前端/日志
        if not conforms and conformance_issue:
            claims.insert(0, {
                "text": f"回答不符合问题要求：{conformance_issue}",
                "status": "unverified",
                "reason": "query_conformance",
            })

        if verified and not needs_retry:
            decision = "verified"
        elif needs_retry:
            decision = "needs_retry"
        else:
            decision = "partial_verified"

        structured = {
            "verified": verified,
            "claims": claims,
            "needs_retry": needs_retry,
            "retry_query": response.get("retry_query", ""),
            "unverified_count": response.get("unverified_count", 0),
            "query_conformance": {
                "conforms": conforms,
                "issue": conformance_issue,
            },
        }

        return decision, structured

    def run(self, context: AgentContext, thinking_callback: Optional[Callable[[str], None]] = None) -> AgentResult:
        """执行验证，将结果写入context"""
        result = super().run(context, thinking_callback=thinking_callback)

        # 将验证结果写入context
        vr = result.data
        # 问题要求不匹配（如选项题未给出 ABCD 字母）：强制触发一次纠正性重试，
        # 借助 verifier_retry 通道重新检索+重新生成，让 Generator 有机会按问题形式补全回答
        conformance = vr.get("query_conformance") or {}
        if not conformance.get("conforms", True) and not vr.get("needs_retry"):
            vr["needs_retry"] = True
            if not (vr.get("retry_query") or "").strip():
                vr["retry_query"] = context.query
            logger.info(
                "[Verifier] 回答不符合问题要求，触发纠正性重试: %s",
                conformance.get("issue", "")[:120],
            )
        context.verification_result = vr

        return result
