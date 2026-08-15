"""
Planner Agent - 规划Agent

4项职责：
  1. 领域越界检测：判断问题是否属于监管知识领域
  2. 意图与复杂度判断：结合V1的QuerySpec做二次确认
  3. 检索策略规划：选择检索通道、策略组合
  4. 问题拆解：按类型拆解（如comparison拆为多个子查询）

输出：retrieval_plan写入AgentContext，驱动Retriever执行
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_platform.runtime.llm_client import LLMClient, LLMMessage

from .base_agent import BaseAgent
from .agent_context import AgentContext, AgentResult

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    规划Agent

    分析用户问题，输出检索计划（策略、子查询、越界判断）。
    LLM驱动，返回结构化JSON。
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="Planner", temperature=0.1, max_tokens=2048)

    def _build_prompt(self, context: AgentContext) -> List[LLMMessage]:
        """构建Planner提示词"""
        query_spec_dict = context.query_spec.to_dict() if context.query_spec else {}
        route_dict = {}
        if context.route_decision:
            rd = context.route_decision
            route_dict = {
                "intent": rd.intent,
                "level": rd.level,
                "channels": rd.channels,
                "top_k": rd.top_k,
                "need_decomposition": rd.need_decomposition,
            }

        system_prompt = (
            "你是一个银行业/保险业监管知识问答系统的规划Agent。\n"
            "你的职责是分析用户问题，制定检索计划。\n\n"
            "## 领域范围\n"
            "本系统覆盖：银行监管法规、保险监管法规、资本充足率、偿付能力、\n"
            "银行业金融机构数据、保险业数据、监管指标、合规要求等。\n"
            "不属于本系统的问题包括：天气预报、体育新闻、娱乐八卦、技术编程、\n"
            "医疗健康、旅游攻略等非监管领域问题。\n\n"
            "## 你必须输出以下JSON格式\n"
            "{\n"
            '  "is_out_of_domain": false,\n'
            '  "domain_confidence": 0.9,\n'
            '  "intent": "table_lookup",\n'
            '  "complexity": "L2",\n'
            '  "sub_queries": [\n'
            '    {"id": "q1", "text": "子查询文本", "strategy": "hybrid", "filters": {}}\n'
            '  ],\n'
            '  "retrieval_plan": {\n'
            '    "strategies": ["hybrid"],\n'
            '    "top_k": 10,\n'
            '    "rerank": true,\n'
            '    "expand_context": false\n'
            '  },\n'
            '  "reasoning": "分析理由"\n'
            "}\n\n"
            "## 检索策略说明\n"
            "- hybrid: 混合检索(BM25+Dense)，适合通用查询\n"
            "- bm25: 词法检索，适合条款编号、精确匹配\n"
            "- dense: 语义检索，适合概念性查询\n"
            "- table: 表格检索，适合数值、指标查询\n"
            "- exact: 精确匹配，适合条款号、定义查询\n"
            "- metadata: 元数据过滤，适合按文档名/类型筛选\n\n"
            "## 问题拆解规则\n"
            "- comparison类问题必须拆解为多个独立子查询\n"
            "- 简单查询（clause_query/definition/threshold）不拆解\n"
            "- table_lookup不拆解但可追加table策略\n"
        )

        user_prompt = json.dumps({
            "query": context.query,
            "query_spec": query_spec_dict,
            "route_decision": route_dict,
        }, ensure_ascii=False)

        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

    def _parse_response(self, response: dict) -> Tuple[str, Dict[str, Any]]:
        """解析Planner响应"""
        is_out_of_domain = response.get("is_out_of_domain", False)

        if is_out_of_domain:
            decision = "out_of_domain"
        else:
            decision = "proceed"

        # 确保sub_queries至少有一个
        sub_queries = response.get("sub_queries", [])
        if not sub_queries and not is_out_of_domain:
            sub_queries = [{
                "id": "q1",
                "text": response.get("query", ""),
                "strategy": "hybrid",
                "filters": {},
            }]

        structured = {
            "is_out_of_domain": is_out_of_domain,
            "domain_confidence": response.get("domain_confidence", 0.5),
            "intent": response.get("intent", "unknown"),
            "complexity": response.get("complexity", "L2"),
            "sub_queries": sub_queries,
            "retrieval_plan": response.get("retrieval_plan", {
                "strategies": ["hybrid"],
                "top_k": 10,
                "rerank": False,
                "expand_context": False,
            }),
            "reasoning": response.get("reasoning", ""),
        }

        return decision, structured

    def run(self, context: AgentContext, thinking_callback: Optional[Callable[[str], None]] = None) -> AgentResult:
        """执行规划，将结果写入context"""
        result = super().run(context, thinking_callback=thinking_callback)

        # 将检索计划写入context
        context.retrieval_plan = result.data

        return result
