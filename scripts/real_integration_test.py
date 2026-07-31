"""
ACE-RAG 真实联调测试脚本 — 连接真实检索服务 + 真实 LLM (DeepSeek)

前置条件:
  1. 启动真实检索服务: python -m retrieval_service.server (port 8000)
  2. 配置 .env 中的 LLM_API_KEY (DeepSeek)

用法:
    python scripts/real_integration_test.py
    python scripts/real_integration_test.py "银行业总资产是多少"
    python scripts/real_integration_test.py --multi   # 多轮对话

特性:
  1. HTTP 模式连接真实检索服务（非 Mock）
  2. 真实 LLM 回答生成（DeepSeek API）
  3. 完整可观测性：链路追踪 + 指标采集 + 结构化日志
  4. 分层执行日志打印（进入路由层 XXX 等）
"""

import json
import logging
import os
import sys
import time
from typing import Optional

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 加载 .env 环境变量
from pathlib import Path

_env_file = Path(_PROJECT_ROOT) / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

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


class RealtimeProgressHandler(logging.Handler):
    """实时进度输出到 stdout — 仅显示 [步骤] 日志"""

    def emit(self, record):
        msg = record.getMessage()
        if "[步骤]" in msg or "检索失败" in msg or "处理查询异常" in msg:
            print(f"  ⏳ {msg}", flush=True)


def setup_observability():
    """设置可观测性 — 返回 (tracer, collector, log_handler)"""
    tracer = Tracer()
    collector = get_collector()
    collector.reset()

    log_handler = ObservabilityHandler()
    log_handler.setLevel(logging.DEBUG)

    # 实时进度输出（仅 [步骤] 级别日志）
    progress_handler = RealtimeProgressHandler()
    progress_handler.setLevel(logging.INFO)

    ap_logger = logging.getLogger("agent_platform")
    # 清理旧 handler，避免交互模式下重复输出
    ap_logger.handlers.clear()
    ap_logger.addHandler(log_handler)
    ap_logger.addHandler(progress_handler)
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

    for key, value in span.attributes.items():
        print(f"{prefix}    · {key}: {value}")

    for event in span.events:
        evt_name = event.get("name", "")
        print(f"{prefix}    ⚡ {evt_name}", end="")
        attrs = event.get("attributes", {})
        if attrs:
            print(f" {json.dumps(attrs, ensure_ascii=False)}", end="")
        print()

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
# 真实联调核心函数
# ============================================================

def run_real_query(
    handler: RequestHandler,
    query: str,
    session_id: Optional[str] = None,
    show_trace: bool = True,
    show_metrics: bool = True,
    show_logs: bool = True,
) -> str:
    """
    执行一次真实查询并打印可观测性信息

    Returns:
        session_id（用于多轮对话）
    """
    tracer, collector, log_handler = setup_observability()
    request_id = generate_request_id()

    print_separator(f"提问: {query}")
    print(f"  request_id: {request_id}")
    print(f"  模式: 真实联调（HTTP 检索 + DeepSeek LLM）")
    print(f"  ⏳ 开始处理 ...\n")

    start_total = time.perf_counter()

    # 用 Tracer 包裹整个查询流程
    with SpanContext(
        tracer, tracer.start_span("QueryRequest", query=query, request_id=request_id)
    ) as root:
        root.set_attribute("session_id", session_id or "new")
        root.set_attribute("mode", "real_integration")

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

    total_ms = (time.perf_counter() - start_total) * 1000

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

    print(f"\n  ⏱ 总耗时: {total_ms:.1f}ms (含可观测性开销)")

    return response.session_id


# ============================================================
# 检索服务健康检查
# ============================================================

def check_retrieval_service(base_url: str = "http://127.0.0.1:8000") -> bool:
    """检查真实检索服务是否可用"""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            docs = data.get("docs", 0)
            chunks = data.get("chunks", 0)
            print(f"  ✓ 检索服务可用 — {docs} 文档, {chunks} chunks")
            return True
    except Exception as e:
        print(f"  ✗ 检索服务不可用: {e}")
        return False


def check_llm_service() -> bool:
    """检查 LLM API 是否可用"""
    from agent_platform.runtime.llm_client import LLMClient

    client = LLMClient()
    if client.is_mock:
        print("  ✗ LLM 运行在 Mock 模式（LLM_API_KEY 未配置）")
        return False

    print(f"  ✓ LLM 就绪 — 模型: {client.model}, 后端: {client.backend}")
    client.close()
    return True


# ============================================================
# 交互式主循环
# ============================================================

def interactive_mode(handler: RequestHandler, multi_turn: bool = False):
    """交互式提问模式"""
    print_separator("ACE-RAG 真实联调测试")
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
            session_id = run_real_query(
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
# 推荐测试问题（基于真实数据：银行业资产负债月度数据）
# ============================================================

RECOMMENDED_QUERIES = {
    "基础查询": [
        "银行业总资产是多少",
        "银行业总负债是多少",
        "银行业资产负债情况",
    ],
    "多轮对话": [
        "第1轮: 银行业总资产是多少",
        "第2轮: 这个数据的来源是什么",
        "第3轮: 还有其他相关指标吗",
    ],
    "边界场景": [
        "你好",
        "不存在的数据查询测试",
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

    parser = argparse.ArgumentParser(description="ACE-RAG 真实联调测试")
    parser.add_argument("query", nargs="?", help="直接提问（不进入交互模式）")
    parser.add_argument("--multi", action="store_true", help="多轮对话模式（保持会话）")
    parser.add_argument("--list", action="store_true", help="打印推荐测试问题")
    parser.add_argument("--simple", action="store_true", help="简化模式（只显示回答）")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000",
                        help="检索服务 URL（默认 http://127.0.0.1:8000）")
    args = parser.parse_args()

    if args.list:
        print_recommendations()
        return

    # ── 前置检查 ──
    print_separator("环境检查")
    print("  [1/2] 检索服务:")
    retrieval_ok = check_retrieval_service(args.retrieval_url)
    print("  [2/2] LLM 服务:")
    llm_ok = check_llm_service()

    if not retrieval_ok:
        print("\n  ⚠ 检索服务未启动！请先运行:")
        print("    cd knowledge_platform/retrieval")
        print("    python -m retrieval_service.server")
        print("\n  将使用 Mock 检索模式继续...")

    if not llm_ok:
        print("\n  ⚠ LLM 未配置！请在 .env 中设置 LLM_API_KEY")
        print("\n  将使用 Mock LLM 模式继续...")

    if not retrieval_ok and not llm_ok:
        print("\n  ✗ 检索服务和 LLM 均不可用，无法进行真实联调。")
        return

    # ── 创建 RequestHandler ──
    # 真实联调: HTTP 模式检索 + 真实 LLM
    handler = RequestHandler(
        retrieval_client=RetrievalClient(
            base_url=args.retrieval_url,
            in_process=False,  # HTTP 模式
            timeout_ms=10000,
        ),
        session_manager=SessionManager(),
    )

    show_all = not args.simple

    if args.query:
        # 直接提问模式
        run_real_query(
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
