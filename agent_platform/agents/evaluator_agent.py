"""
Evaluator Agent - 评估Agent

4维度评估证据充分性：
  1. 声明槽位覆盖：检查关键声明是否都有证据支撑
  2. 证据冲突检测：检测同一指标不同数值的冲突
  3. 来源权威与版本有效性：检查证据来源和版本状态
  4. 检索方向建议：输出具体建议驱动Loop

与SufficiencyScorer的关系：
  SufficiencyScorer提供五维度数值评分（规则计算）
  Evaluator LLM基于评分+证据内容做综合判断+输出检索建议
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from agent_platform.runtime.llm_client import LLMClient, LLMMessage

from .base_agent import BaseAgent
from .agent_context import AgentContext, AgentResult

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """
    评估Agent

    基于SufficiencyScorer的五维度评分+LLM综合判断，
    决定证据是否充分，不足时输出检索方向建议。
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="Evaluator", temperature=0.1, max_tokens=2048)

    def _build_prompt(self, context: AgentContext) -> List[LLMMessage]:
        """构建Evaluator提示词"""
        system_prompt = (
            "你是一个银行业/保险业监管知识问答系统的评估Agent。\n"
            "你的职责是评估检索到的证据是否充分回答用户问题。\n\n"
            "## 评估维度\n"
            "1. 声明槽位覆盖：关键声明是否都有证据支撑\n"
            "2. 证据冲突检测：同一指标是否存在不同数值\n"
            "3. 来源权威与版本有效性：证据来源是否权威、版本是否现行有效\n"
            "4. 检索方向建议：证据不足时建议下一步检索方向\n\n"
            "## 你必须输出以下JSON格式\n"
            "{\n"
            '  "is_sufficient": false,\n'
            '  "sufficiency_score": 0.72,\n'
            '  "dimensions": {\n'
            '    "coverage": 0.8,\n'
            '    "authority": 0.9,\n'
            '    "version_validity": 0.6,\n'
            '    "conflict": 0.85\n'
            '  },\n'
            '  "missing_claims": ["缺少XX指标的具体数值"],\n'
            '  "conflicts": [{"type": "version_conflict", "description": "..."}],\n'
            '  "retrieval_suggestion": {\n'
            '    "direction": "补充表格数据",\n'
            '    "suggested_strategy": "table",\n'
            '    "suggested_query": "XX指标 表格数据",\n'
            '    "reason": "当前证据缺少表格中的具体数值"\n'
            '  }\n'
            "}\n\n"
            "## 判断规则\n"
            "- 证据充分（is_sufficient=true）时不输出retrieval_suggestion\n"
            "- 证据不足时必须输出具体的retrieval_suggestion\n"
            "- 检测到版本冲突时在conflicts中标记\n"
            "- missing_claims列出缺少证据支撑的声明\n"
        )

        # 构建证据信息
        bundle_dict = {}
        if context.evidence_bundle:
            bundle_dict = context.evidence_bundle.to_dict()

        # 检索历史
        history = context.retrieval_history[-3:] if context.retrieval_history else []

        user_prompt = json.dumps({
            "query": context.query,
            "loop_count": context.loop_count,
            "evidence_bundle": bundle_dict,
            "retrieval_history": history,
        }, ensure_ascii=False)

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

    def _parse_response(self, response: dict) -> Tuple[str, Dict[str, Any]]:
        """解析Evaluator响应"""
        is_sufficient = response.get("is_sufficient", False)
        decision = "sufficient" if is_sufficient else "insufficient"

        structured = {
            "is_sufficient": is_sufficient,
            "sufficiency_score": response.get("sufficiency_score", 0.0),
            "dimensions": response.get("dimensions", {}),
            "missing_claims": response.get("missing_claims", []),
            "conflicts": response.get("conflicts", []),
            "retrieval_suggestion": response.get("retrieval_suggestion", {}),
        }

        return decision, structured

    def run(self, context: AgentContext) -> AgentResult:
        """执行评估，将结果写入context"""
        result = super().run(context)

        # 将评估结果写入context
        context.evaluation_result = result.data

        # 如果不足，将检索建议写入context供Retriever读取
        if not result.data.get("is_sufficient", False):
            context.retrieval_suggestion = result.data.get("retrieval_suggestion", {})

        return result
