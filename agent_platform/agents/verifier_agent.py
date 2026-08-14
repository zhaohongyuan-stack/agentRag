"""
Verifier Agent - 验证Agent

声明级验证 + 有限次补充检索闭环：
  1. 整体一致性检查：回答是否与证据一致
  2. 声明级验证：拆分回答为声明，逐个检查证据支撑
  3. 有限次补充检索：无证据声明触发补充检索（受预算控制）
  4. 标记不确定：仍无证据的声明标记"依据不足"
"""

import json
import logging
from typing import Any, Dict, List, Tuple

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
            "你的职责是验证生成回答中的每个声明是否有证据支撑。\n\n"
            "## 验证流程\n"
            "1. 将回答拆分为多个独立声明（每个含具体数值或结论的句子）\n"
            "2. 对每个声明，在证据中查找是否有直接支撑\n"
            "3. 有证据支撑的标记verified，无证据的标记unverified\n"
            "4. 对unverified声明，判断是否可能通过补充检索找到证据\n"
            "\n"
            "## 你必须输出以下JSON格式\n"
            "{\n"
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
            "- needs_retry=true时必须提供retry_query\n"
            "- 所有声明都verified时needs_retry=false\n"
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

        if verified and not needs_retry:
            decision = "verified"
        elif needs_retry:
            decision = "needs_retry"
        else:
            decision = "partial_verified"

        structured = {
            "verified": verified,
            "claims": response.get("claims", []),
            "needs_retry": needs_retry,
            "retry_query": response.get("retry_query", ""),
            "unverified_count": response.get("unverified_count", 0),
        }

        return decision, structured

    def run(self, context: AgentContext) -> AgentResult:
        """执行验证，将结果写入context"""
        result = super().run(context)

        # 将验证结果写入context
        context.verification_result = result.data

        return result
