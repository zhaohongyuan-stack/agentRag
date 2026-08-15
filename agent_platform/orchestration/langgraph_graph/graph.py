"""
LangGraph 图装配（阶段 2）

用 StateGraph + 条件边表达两个 Loop：
  1. 检索-评估 Loop: retriever → evidence → evaluator → loop_control → retriever
  2. Verifier 重试 Loop: verifier → verifier_retry → verifier

build_graph() 返回编译后的可执行图（CompiledStateGraph），
handler._run_graph_flow 通过 graph.invoke(initial_state) 运行。
本阶段不挂 Checkpointer（阶段 5 引入 SQLite Checkpointer）。
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    evaluator_node,
    evidence_node,
    final_judge_node,
    generator_node,
    loop_control_node,
    planner_node,
    refuse_node,
    respond_node,
    retriever_node,
    route_after_evaluator,
    route_after_evidence,
    route_after_final_judge,
    route_after_loop_control,
    route_after_planner,
    route_after_verifier,
    route_after_verifier_retry,
    verifier_node,
    verifier_retry_node,
)
from .state import GraphState


def build_graph():
    """
    装配并编译 Agent 协作图

    拓扑（与原 _run_agent_flow 逐条对应）:

      START → planner ──越界──→ refuse → END
                 │
                 ↓ 正常
              retriever ⇄ loop_control
                 ↓            ↑          ↓ 预算STOP
              evidence → evaluator ─空轮/预算→ final_judge ─不足→ refuse
                 │           │ 充分            │ 充分
                 └─V1充分────┴─────→ generator ↓
                                              verifier
    """
    graph = StateGraph(GraphState)

    # ── 节点注册 ──
    graph.add_node("planner", planner_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("loop_control", loop_control_node)
    graph.add_node("final_judge", final_judge_node)
    graph.add_node("generator", generator_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("verifier_retry", verifier_retry_node)
    graph.add_node("respond", respond_node)

    # ── 入口 ──
    graph.add_edge(START, "planner")

    # ── planner: 越界 → 拒答；正常 → 检索 ──
    graph.add_conditional_edges("planner", route_after_planner, {
        "refuse": "refuse",
        "retriever": "retriever",
    })

    # ── 检索-评估 Loop ──
    graph.add_edge("retriever", "evidence")
    graph.add_conditional_edges("evidence", route_after_evidence, {
        "generator": "generator",   # V1 规则评分充分，跳过 Evaluator LLM
        "evaluator": "evaluator",
    })
    graph.add_conditional_edges("evaluator", route_after_evaluator, {
        "generator": "generator",   # 充分（任一判定充分即退出）
        "final_judge": "final_judge",  # 空轮次保护 / 预算耗尽 → 最终判断
        "loop_control": "loop_control",
    })
    graph.add_conditional_edges("loop_control", route_after_loop_control, {
        "retriever": "retriever",   # 调整策略后继续 Loop
        "final_judge": "final_judge",  # 预算 STOP → 最终判断
    })
    graph.add_conditional_edges("final_judge", route_after_final_judge, {
        "generator": "generator",   # 最终判断充分 → 生成
        "refuse": "refuse",         # 证据不足 → 澄清/拒答
    })

    # ── 生成与验证 ──
    graph.add_edge("generator", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier, {
        "respond": "respond",
        "verifier_retry": "verifier_retry",
    })
    graph.add_conditional_edges("verifier_retry", route_after_verifier_retry, {
        "verifier": "verifier",     # 有新证据 → 重新验证
        "respond": "respond",       # 无新证据 → 保留原回答
    })

    # ── 出口 ──
    graph.add_edge("refuse", END)
    graph.add_edge("respond", END)

    return graph.compile()
