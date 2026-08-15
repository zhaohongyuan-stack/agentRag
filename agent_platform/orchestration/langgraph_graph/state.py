"""
LangGraph 图状态定义（阶段 2）

设计原则（对应指导书决策点 2）：
  - State 承载流程数据（检索计划、证据包、评估结果、控制标志）
  - 复杂运行时对象（QuerySpec/EvidenceBundle/AgentContext 等）以 Any 携带，
    本阶段不做 Checkpointer 持久化（阶段 5 再引入），进程内 invoke 即可
  - GraphRuntime 聚合「本次查询的输入 + handler 侧服务组件」，节点函数
    通过 state["runtime"] 读取，避免节点与 RequestHandler 强耦合

State 字段说明见 GraphState 注释。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class GraphRuntime:
    """
    Graph 运行时上下文 — 本次查询的不可变输入 + handler 侧服务组件

    由 handler._run_graph_flow 构建后放入 GraphState["runtime"]。
    节点函数只读使用（AgentContext 除外，Agent 产出会写回 context）。
    """

    # ── handler 引用（复用 _build_response/_finalize 等收尾逻辑）──
    handler: Any = None

    # ── 本次查询输入（handle_query 前置阶段产出）──
    request: Any = None                    # QueryRequest
    request_id: str = ""
    session: Any = None                    # Session
    sm: Any = None                         # StateMachine（保留为 trace 记录器）
    start_time: float = 0.0
    query_spec: Any = None                 # QuerySpec
    route_decision: Any = None             # RouteDecision
    rewritten: Any = None                  # RewriteResult
    search_query: str = ""                 # 改写后的检索 query
    trace_collector: Any = None            # TraceCollector（可为 None）
    event_callback: Any = None             # EventCallback（可为 None）

    # ── 服务组件（从 handler 注入，便于测试替换）──
    retrieval_client: Any = None
    evidence_builder: Any = None
    generator: Any = None
    planner_agent: Any = None
    evaluator_agent: Any = None
    verifier_agent: Any = None
    redis: Any = None                      # Redis 工具缓存（可为 None）
    budget_controller: Any = None          # 外部注入的预算控制器（可为 None）
    query_spec_builder: Any = None         # QuerySpecBuilder（clean_query 重建 filters 用）


class GraphState(TypedDict, total=False):
    """
    LangGraph 图状态

    节点函数读取所需字段，返回部分更新 dict（LangGraph 自动 merge）。
    """

    # ── 运行时（入口注入，全程只读）──
    runtime: Any                     # GraphRuntime
    context: Any                     # AgentContext（Agent 间共享上下文）

    # ── Planner 输出 ──
    retrieval_plan: Dict[str, Any]
    is_out_of_domain: bool

    # ── 检索-评估 Loop ──
    base_filters: Dict[str, Any]     # 初始过滤条件（doc_name 安全锁所在）
    current_query: str               # 当前轮检索 query（Loop 中会被建议/槽位改写）
    current_filters: Dict[str, Any]  # 当前轮过滤条件
    accumulated_hits: List[dict]     # 跨轮次累积检索结果（chunk_id 去重）
    new_hit_count: int               # 本轮新增证据数（空轮次保护用）
    retrieval_result: Any            # 最近一轮 RetrievalResult（Verifier 重试合并用）
    actual_filters: Dict[str, Any]   # 本轮实际生效 filters（语义兜底后，证据组装入参）
    last_retrieval_latency: int      # 本轮检索耗时（on_loop_round 事件用）
    evidence_bundle: Any             # EvidenceBundle（每轮从累积 hits 重建）
    evaluation_result: Dict[str, Any]
    retrieval_suggestion: Dict[str, Any]
    is_sufficient: bool              # 双重判断（V1 规则 or Evaluator LLM）
    loop_stopped_reason: str         # Loop 退出原因（budget_exhausted/empty_round/...）

    # ── 拒答/澄清 ──
    refuse_reason: str               # out_of_domain / insufficient / clarify

    # ── 生成与验证 ──
    generated_answer: Any            # GeneratedAnswer
    verification_result: Dict[str, Any]
    verifier_retry_done: bool        # Verifier 补充检索最多 1 次
    needs_retry: bool

    # ── 输出 ──
    response: Any                    # 最终 QueryResponse
    error: Optional[str]
