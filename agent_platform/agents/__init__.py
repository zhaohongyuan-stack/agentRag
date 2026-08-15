"""
Agent 协作层 - V2 Agentic RAG

4个LLM驱动的Agent通过AgentContext共享上下文直接通信：
  Planner    -> 规划检索策略、越界检测、问题拆解
  Retriever  -> 调用检索层API、组装证据包
  Evaluator  -> 评估证据充分性、输出检索方向建议（驱动Loop）
  Verifier   -> 声明级验证、有限次补充检索闭环

与V1的关系：增量改造，V1的检索逻辑留在检索层，Agent层只调度不替换。
"""

from .agent_context import AgentContext, AgentResult
from .base_agent import BaseAgent
from .planner_agent import PlannerAgent
from .evaluator_agent import EvaluatorAgent
from .verifier_agent import VerifierAgent

__all__ = [
    "AgentContext",
    "AgentResult",
    "BaseAgent",
    "PlannerAgent",
    "EvaluatorAgent",
    "VerifierAgent",
]
