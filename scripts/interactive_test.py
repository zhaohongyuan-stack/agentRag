"""
ACE-RAG 交互式测试脚本 — 可观测的端到端提问测试

用法:
    python scripts/interactive_test.py
    python scripts/interactive_test.py "核心一级资本充足率最低要求是多少"
    python scripts/interactive_test.py --multi   # 多轮对话模式

特性:
  1. 使用 Mock 检索服务（无需启动真实检索）
  2. 集成 Tracer 链路追踪 — 打印完整 Span 树
  3. 集成 MetricsCollector — 打印指标采集结果
  4. 集成结构化日志 — 打印 JSON 格式执行日志
  5. 支持多轮对话 — 测试指代消解和记忆
  6. 打印路由路径（P0-P4）、风险级别、证据充分性
"""

import asyncio
import json
import logging
import sys
import os
from typing import Optional

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_platform.gateway.request_handler import (
    QueryRequest,
    RequestHandler,
    RetrievalClient,
)
from agent_platform.gateway.session_handler import SessionManager
from agent_platform.runtime.observability import (
    MetricsCollector,
    SpanContext,
    StructuredLogFormatter,
    Tracer,
    generate_request_id,
    get_collector,
)


# ============================================================
# 可观测日志捕获器
# ============================================================

class ObservabilityHandler(logging.Handler):
    """捕获日志到列表，用于事后展示"""

    def __init__(self):
        super().__init__()
        self.logs: list[str] = []

    def emit(self, record):
        formatter = StructuredLogFormatter()
        self.logs.append(formatter.format(record))


def setup_observability():
    """设置可观测性 — 返回 (tracer, collector, log_handler)"""
    tracer = Tracer()
    collector = get_collector()
    collector.reset()

    log_handler = ObservabilityHandler()
    log_handler.setLevel(logging.DEBUG)

    # 给 agent_platform logger 加上可观测 handler
    ap_logger = logging.getLogger("agent_platform")
    ap_logger.addHandler(log_handler)
    ap_logger.setLevel(logging.DEBUG)

    return tracer, collector, log_handler


# ============================================================
# 打印工具
# ============================================================

def print_separator(title: str = ""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def print_trace_tree(span, indent: int = 0):
    """递归打印 Span 树"""
    prefix = "  " * indent
    status_icon = "✓" if span.status == "ok" else "✗"
    duration = f"{span.duration_ms:.1f}ms" if span.duration_ms >= 0 else "?"

    print(f"{prefix}{status_icon} [{span.name}] ({duration})")

    # 属性
    for key, value in span.attributes.items():
        print(f"{prefix}    · {key}: {value}")

    # 事件
    for event in span.events:
        evt_name = event.get("name", "")
        print(f"{prefix}    ⚡ {evt_name}", end="")
        attrs = event.get("attributes", {})
        if attrs:
            print(f" {json.dumps(attrs, ensure_ascii=False)}", end="")
        print()

    # 子 Span
    for child in span.children:
        print_trace_tree(child, indent + 1)


def print_metrics(collector: MetricsCollector):
    """打印指标采集结果"""
    export = collector.export()

    print("\n📊 指标采集:")
    print(f"  查询总数:       {export.get('agent_queries_total', {}).get('values', {}).get('', 0)}")
    print(f"  检索调用数:     {export.get('retrieval_calls_total', {}).get('values', {}).get('', 0)}")

    latency = export.get('agent_query_latency_seconds', {})
    if latency.get('count', 0) > 0:
        print(f"  查询延迟(avg):   {latency.get('avg', 0):.3f}s (count={latency.get('count')})")

    retrieval_lat = export.get('retrieval_latency_seconds', {})
    if retrieval_lat.get('count', 0) > 0:
        print(f"  检索延迟(avg):   {retrieval_lat.get('avg', 0):.3f}s")

    suff = export.get('evidence_sufficiency_score', {})
    if suff.get('count', 0) > 0:
        print(f"  证据充分性:      {suff.get('avg', 0):.3f}")

    budget = export.get('budget_consumed_ratio', {})
    if 'value' in budget:
        print(f"  预算消耗:        {budget.get('value', 0):.1%}")


def print_logs(log_handler: ObservabilityHandler, request_id: Optional[str] = None):
    """打印结构化日志"""
    print(f"\n📝 执行日志 ({len(log_handler.logs)} 条):")
    for log_line in log_handler.logs:
        try:
            entry = json.loads(log_line)
            # 可选: 按 request_id 过滤
            if request_id and entry.get("request_id") != request_id:
                continue
            level = entry.get("level", "?")
            module = entry.get("module", "?")
            msg = entry.get("message", "")
            icon = {"INFO": "ℹ", "DEBUG": "·", "WARNING": "⚠", "ERROR": "✗"}.get(level, "?")
            print(f"  {icon} [{level:5}] {module}: {msg}")
        except json.JSONDecodeError:
            print(f"  {log_line}")


def print_response_detail(response):
    """打印响应详情"""
    print_separator("响应详情")
    print(f"  请求 ID:       {response.request_id}")
    print(f"  会话 ID:       {response.session_id}")
    print(f"  意图:          {response.intent}")
    print(f"  复杂度:        {response.complexity}")
    print(f"  风险级别:      {getattr(response, 'risk_level', 'N/A')}")
    print(f"  执行路径:      {getattr(response, 'path_id', 'N/A')}")
    print(f"  证据数:        {response.evidence_count}")
    print(f"  充分性评分:    {response.sufficiency_score:.3f}")
    print(f"  置信度:        {response.confidence:.3f}")
    print(f"  延迟:          {response.latency_ms:.1f}ms")
    print(f"  是否拒答:      {'是' if response.is_refusal else '否'}")
    if response.refusal_reason:
        print(f"  拒答原因:      {response.refusal_reason}")

    print(f"\n  状态轨迹:")
    for i, state in enumerate(response.state_trace):
        print(f"    [{i+1}] {state}")

    print(f"\n  引用列表 ({len(response.citations)} 条):")
    for i, cite in enumerate(response.citations):
        print(f"    [{i+1}] {cite.get('source_doc', '?')} — {cite.get('citation', '?')}")

    print(f"\n  回答:")
    print(f"  {'-'*50}")
    print(f"  {response.answer}")
    print(f"  {'-'*50}")


# ============================================================
# 核心测试函数
# ============================================================

def run_query(
    handler: RequestHandler,
    query: str,
    session_id: Optional[str] = None,
    show_trace: bool = True,
    show_metrics: bool = True,
    show_logs: bool = True,
) -> str:
    """
    执行一次查询并打印可观测性信息

    Returns:
        session_id（用于多轮对话）
    """
    tracer, collector, log_handler = setup_observability()
    request_id = generate_request_id()

    print_separator(f"提问: {query}")
    print(f"  request_id: {request_id}")

    # 用 Tracer 包裹整个查询流程
    with SpanContext(
        tracer, tracer.start_span("QueryRequest", query=query, request_id=request_id)
    ) as root:
        root.set_attribute("session_id", session_id or "new")

        # 执行查询
        request = QueryRequest(query=query, session_id=session_id)
        response = handler.handle_query(request)

        root.set_attribute("intent", response.intent)
        root.set_attribute("complexity", response.complexity)
        root.set_attribute("evidence_count", response.evidence_count)
        root.set_attribute("sufficiency", response.sufficiency_score)
        root.set_attribute("is_refusal", response.is_refusal)

        # 记录指标
        collector.counter("agent_queries_total").inc()
        collector.histogram("agent_query_latency_seconds").observe(
            response.latency_ms / 1000
        )
        collector.histogram("evidence_sufficiency_score").observe(
            response.sufficiency_score
        )

    # 打印响应详情
    print_response_detail(response)

    # 打印链路追踪
    if show_trace:
        print_separator("链路追踪")
        trace = tracer.finish()
        if trace:
            print_trace_tree(trace)
        else:
            print("  (无追踪数据)")

    # 打印指标
    if show_metrics:
        print_metrics(collector)

    # 打印日志
    if show_logs:
        print_logs(log_handler, request_id=request_id)

    return response.session_id


# ============================================================
# 交互式主循环
# ============================================================

def interactive_mode(handler: RequestHandler, multi_turn: bool = False):
    """交互式提问模式"""
    print_separator("ACE-RAG 交互式测试")
    print("  输入问题进行测试，输入 'quit' 或 'exit' 退出")
    print("  输入 'simple' 切换简化模式（只显示回答）")
    print("  输入 'full' 切换完整模式（追踪+指标+日志）")
    print()

    session_id = None
    show_all = True

    while True:
        try:
            query = input("🤔 你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        if query.lower() == "simple":
            show_all = False
            print("已切换到简化模式")
            continue
        if query.lower() == "full":
            show_all = True
            print("已切换到完整模式")
            continue

        # 执行查询
        try:
            session_id = run_query(
                handler,
                query,
                session_id=session_id if multi_turn else None,
                show_trace=show_all,
                show_metrics=show_all,
                show_logs=show_all,
            )
        except Exception as e:
            print(f"\n✗ 执行出错: {e}")
            import traceback
            traceback.print_exc()

        print()


# ============================================================
# 推荐测试问题
# ============================================================

RECOMMENDED_QUERIES = {
    "L1 精确条款查询 (P1路径)": [
        "《商业银行资本管理办法》第43条",
        "《商业银行资本管理办法》第23条是什么",
    ],
    "L2 阈值查询 (P2/P3路径)": [
        "核心一级资本充足率最低要求是多少",
        "杠杆率不得低于多少",
        "流动性覆盖率最低标准是什么",
    ],
    "L2 定义查询 (P2路径)": [
        "什么是系统重要性银行",
        "什么是核心一级资本",
    ],
    "L3 比较查询 (P3路径)": [
        "核心一级资本充足率和一级资本充足率有什么区别",
        "商业银行资本管理办法和巴塞尔协议III的区别",
    ],
    "L4 合规判断 (P4路径，高风险)": [
        "某银行核心一级资本充足率为4%，是否符合监管要求",
        "银行杠杆率为3%是否合规",
    ],
    "多轮对话（指代消解）": [
        "第1轮: 核心一级资本充足率最低要求是多少",
        "第2轮: 这个比例适用于哪些银行",
        "第3轮: 那个文件还有哪些要求",
    ],
    "边界场景": [
        "空结果",           # 触发空结果拒答
        "超时",             # 触发超时
        "版本冲突",         # 触发版本冲突
    ],
}


def print_recommendations():
    """打印推荐测试问题"""
    print_separator("推荐测试问题")
    for category, queries in RECOMMENDED_QUERIES.items():
        print(f"\n  【{category}】")
        for q in queries:
            print(f"    · {q}")


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ACE-RAG 交互式测试")
    parser.add_argument("query", nargs="?", help="直接提问（不进入交互模式）")
    parser.add_argument("--multi", action="store_true", help="多轮对话模式（保持会话）")
    parser.add_argument("--list", action="store_true", help="打印推荐测试问题")
    parser.add_argument("--simple", action="store_true", help="简化模式（只显示回答）")
    args = parser.parse_args()

    if args.list:
        print_recommendations()
        return

    # 创建 RequestHandler（Mock 模式）
    handler = RequestHandler(
        retrieval_client=RetrievalClient(in_process=True),
        session_manager=SessionManager(),
    )

    show_all = not args.simple

    if args.query:
        # 直接提问模式
        run_query(
            handler,
            args.query,
            show_trace=show_all,
            show_metrics=show_all,
            show_logs=show_all,
        )
    else:
        # 交互模式
        print_recommendations()
        interactive_mode(handler, multi_turn=args.multi)


if __name__ == "__main__":
    main()
