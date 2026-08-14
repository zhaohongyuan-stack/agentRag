"""
EventCallback — Agent 执行事件回调机制

用于 Handler 在 Agent 执行过程中推送实时事件，供 SSE 端点消费。
不传 callback 时行为完全不变，零侵入向后兼容。

事件类型:
  agent:start       Agent 开始执行
  agent:thinking    Agent 正在推理（逐行思考过程）
  agent:result      Agent 输出决策结果
  tool:call         Agent 调用工具
  tool:result       工具返回结果
  loop:round        检索-评估 Loop 新轮次
  answer:token      Generator 逐 token 输出
  error             执行异常
  done              流程结束，返回完整响应
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable


class EventType(str, Enum):
    """事件类型枚举"""
    AGENT_START = "agent:start"
    AGENT_THINKING = "agent:thinking"
    AGENT_RESULT = "agent:result"
    TOOL_CALL = "tool:call"
    TOOL_RESULT = "tool:result"
    LOOP_ROUND = "loop:round"
    ANSWER_TOKEN = "answer:token"
    ERROR = "error"
    DONE = "done"


@dataclass
class AgentEvent:
    """
    单个 Agent 事件

    所有事件统一格式，前端按 type 区分渲染。
    """
    type: str                         # EventType 值
    agent: str = ""                   # Agent 名称 (Planner/Retriever/Evaluator/Verifier)
    round: int = 0                    # 检索轮次 (0=单轮)
    decision: str = ""                # 决策摘要 (proceed/sufficient/verified 等)
    data: Dict[str, Any] = field(default_factory=dict)   # 结构化数据
    latency_ms: int = 0               # 耗时（毫秒）
    timestamp: float = 0.0            # 事件发生时间戳

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "agent": self.agent,
            "round": self.round,
            "decision": self.decision,
            "data": self.data,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp or time.time(),
        }


class EventCallback:
    """
    Agent 事件回调基类

    Handler 在每个 Agent 执行点调用对应方法。
    子类可重写方法以自定义事件处理逻辑（如 SSE 推送、日志记录等）。

    所有方法都有默认空实现，不传 callback 时零开销。
    """

    def on_agent_start(self, agent: str, round: int = 0, detail: Optional[Dict] = None):
        """Agent 开始执行"""
        pass

    def on_agent_thinking(self, agent: str, content: str, round: int = 0):
        """Agent 正在推理（逐行思考过程）"""
        pass

    def on_agent_result(
        self,
        agent: str,
        decision: str,
        detail: Optional[Dict] = None,
        latency_ms: int = 0,
        round: int = 0,
    ):
        """Agent 输出决策结果"""
        pass

    def on_tool_call(
        self,
        agent: str,
        tool_name: str,
        args: Optional[Dict] = None,
        round: int = 0,
    ):
        """Agent 调用工具"""
        pass

    def on_tool_result(
        self,
        agent: str,
        tool_name: str,
        result: Optional[Dict] = None,
        round: int = 0,
    ):
        """工具返回结果"""
        pass

    def on_loop_round(
        self,
        round: int,
        query: str,
        hits: int,
        score: float,
        strategy: str = "",
        latency_ms: int = 0,
    ):
        """检索-评估 Loop 新轮次"""
        pass

    def on_answer_token(self, token: str):
        """Generator 逐 token 输出（用于实时流式渲染）"""
        pass

    def on_error(self, agent: str, message: str, round: int = 0):
        """执行异常"""
        pass

    def on_done(self, response: Dict[str, Any]):
        """流程结束，返回完整响应"""
        pass


class SSEEventCallback(EventCallback):
    """
    SSE 事件回调实现

    将 Agent 事件推送到 asyncio.Queue，供 SSE StreamingResponse 消费。
    通过 loop.call_soon_threadsafe 从 worker 线程安全推送事件。
    """

    def __init__(self, queue: "asyncio.Queue", loop: Optional["asyncio.AbstractEventLoop"] = None):
        self._queue = queue
        # 持有拥有 queue 的事件循环（在 async 端点中通过 asyncio.get_running_loop() 获取）
        self._loop = loop
        self._start_times: Dict[str, float] = {}

    def _put(self, event: AgentEvent):
        """将事件推入队列（线程安全）"""
        try:
            import asyncio
            payload = event.to_dict()
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)
            else:
                # 兜底：尝试直接放入（同线程场景）
                self._queue.put_nowait(payload)
        except Exception:
            pass  # 队列满或事件循环不可用时静默丢弃

    def on_agent_start(self, agent: str, round: int = 0, detail: Optional[Dict] = None):
        key = f"{agent}_{round}"
        self._start_times[key] = time.time()
        self._put(AgentEvent(
            type=EventType.AGENT_START,
            agent=agent,
            round=round,
            data=detail or {},
        ))

    def on_agent_thinking(self, agent: str, content: str, round: int = 0):
        self._put(AgentEvent(
            type=EventType.AGENT_THINKING,
            agent=agent,
            round=round,
            data={"content": content},
        ))

    def on_agent_result(
        self,
        agent: str,
        decision: str,
        detail: Optional[Dict] = None,
        latency_ms: int = 0,
        round: int = 0,
    ):
        key = f"{agent}_{round}"
        if latency_ms == 0 and key in self._start_times:
            latency_ms = int((time.time() - self._start_times[key]) * 1000)
        self._put(AgentEvent(
            type=EventType.AGENT_RESULT,
            agent=agent,
            round=round,
            decision=decision,
            data=detail or {},
            latency_ms=latency_ms,
        ))

    def on_tool_call(
        self,
        agent: str,
        tool_name: str,
        args: Optional[Dict] = None,
        round: int = 0,
    ):
        self._put(AgentEvent(
            type=EventType.TOOL_CALL,
            agent=agent,
            round=round,
            data={"tool": tool_name, "args": args or {}},
        ))

    def on_tool_result(
        self,
        agent: str,
        tool_name: str,
        result: Optional[Dict] = None,
        round: int = 0,
    ):
        self._put(AgentEvent(
            type=EventType.TOOL_RESULT,
            agent=agent,
            round=round,
            data={"tool": tool_name, "result": result or {}},
        ))

    def on_loop_round(
        self,
        round: int,
        query: str,
        hits: int,
        score: float,
        strategy: str = "",
        latency_ms: int = 0,
        snippets: Optional[List[str]] = None,
    ):
        self._put(AgentEvent(
            type=EventType.LOOP_ROUND,
            agent="Retriever",
            round=round,
            data={
                "query": query,
                "hits": hits,
                "score": score,
                "strategy": strategy,
                "latency_ms": latency_ms,
                "snippets": snippets or [],
            },
        ))

    def on_answer_token(self, token: str):
        self._put(AgentEvent(
            type=EventType.ANSWER_TOKEN,
            data={"token": token},
        ))

    def on_error(self, agent: str, message: str, round: int = 0):
        self._put(AgentEvent(
            type=EventType.ERROR,
            agent=agent,
            round=round,
            data={"message": message},
        ))

    def on_done(self, response: Dict[str, Any]):
        self._put(AgentEvent(
            type=EventType.DONE,
            data=response,
        ))


class LoggingEventCallback(EventCallback):
    """
    日志记录回调实现

    将事件写入 logger，适合调试和开发环境。
    """

    def __init__(self, logger=None):
        import logging
        self._logger = logger or logging.getLogger(__name__)
        self._start_times: Dict[str, float] = {}

    def on_agent_start(self, agent: str, round: int = 0, detail: Optional[Dict] = None):
        key = f"{agent}_{round}"
        self._start_times[key] = time.time()
        self._logger.info("[Event] %s 开始 (round=%d)", agent, round)

    def on_agent_result(
        self,
        agent: str,
        decision: str,
        detail: Optional[Dict] = None,
        latency_ms: int = 0,
        round: int = 0,
    ):
        key = f"{agent}_{round}"
        if latency_ms == 0 and key in self._start_times:
            latency_ms = int((time.time() - self._start_times[key]) * 1000)
        self._logger.info(
            "[Event] %s 完成 → %s (%dms, round=%d)",
            agent, decision, latency_ms, round,
        )

    def on_tool_call(self, agent: str, tool_name: str, args: Optional[Dict] = None, round: int = 0):
        self._logger.info("[Event] %s 调用工具 %s (round=%d)", agent, tool_name, round)

    def on_tool_result(self, agent: str, tool_name: str, result: Optional[Dict] = None, round: int = 0):
        self._logger.info("[Event] %s 工具 %s 返回 (round=%d)", agent, tool_name, round)

    def on_loop_round(self, round: int, query: str, hits: int, score: float, strategy: str = "", latency_ms: int = 0, snippets: Optional[List[str]] = None):
        self._logger.info(
            "[Event] Loop 第%d轮 → hits=%d, score=%.3f, strategy=%s (%dms)",
            round, hits, score, strategy, latency_ms,
        )