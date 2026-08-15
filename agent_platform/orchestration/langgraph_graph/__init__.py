"""
LangGraph 编排层（阶段 2-3）

用手写编排的等价替换：StateGraph + 条件边表达「检索-评估 Loop」与
「Verifier 重试 Loop」。业务硬约束（doc_name 安全锁、语义兜底、预算控制、
65s 时间闸等）在 nodes.py 中逐条保留。

模块结构:
  state.py  — GraphState / GraphRuntime 定义
  nodes.py  — 节点函数（复用 V1 组件与 3 个 LLM Agent）
  graph.py  — StateGraph 装配（build_graph）

开关: handler 通过 USE_LANGGRAPH=true 启用，异常时自动回退 _run_agent_flow。
"""

from .graph import build_graph
from .state import GraphRuntime, GraphState

__all__ = ["build_graph", "GraphRuntime", "GraphState"]
