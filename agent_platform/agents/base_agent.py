"""
Agent 基类 - BaseAgent

所有4个Agent（Planner/Retriever/Evaluator/Verifier）继承此类。
统一流程：构建提示词 -> 调用LLM -> 解析JSON -> 返回AgentResult

子类只需实现：
  _build_prompt(context) -> List[LLMMessage]
  _parse_response(data) -> (decision, structured_data)
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_platform.runtime.llm_client import (
    LLMClient,
    LLMMessage,
    get_llm_client,
    parse_json_text,
)

from .agent_context import AgentContext, AgentResult

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Agent基类 - 统一结构化输出

    子类通过实现_build_prompt和_parse_response来定制行为。
    run方法封装统一的执行流程：计时 -> 构建提示词 -> LLM调用 -> 解析 -> 返回AgentResult

    容错策略：LLM调用失败直接报错（用户需求：不重试）
    """

    def __init__(
        self,
        llm_client: LLMClient,
        name: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Args:
            llm_client: LLM客户端实例
            name: Agent名称（如"Planner"/"Evaluator"）
            temperature: LLM温度参数（Agent决策建议低温度保证稳定性）
            max_tokens: 最大生成token数
        """
        self._llm = llm_client
        self.name = name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def run(self, context: AgentContext, thinking_callback: Optional[Callable[[str], None]] = None) -> AgentResult:
        """
        统一入口：构建提示词 -> 调用LLM -> 解析JSON -> 返回AgentResult

        Args:
            context: Agent间共享上下文
            thinking_callback: 可选的流式回调；提供时优先以流式方式调用 LLM，
                将 token 实时推送给前端（思考过程流式输出），失败自动回退非流式

        Returns:
            AgentResult统一结构化输出

        Raises:
            Exception: LLM调用失败时直接报错（不重试）
        """
        start_ms = _now_ms()

        # 1. 构建提示词
        messages = self._build_prompt(context)

        # 2. 调用LLM（chat_json确保返回dict；有回调时先尝试流式）
        raw_response = None
        if thinking_callback is not None:
            try:
                raw_response = self._chat_json_stream(messages, thinking_callback)
            except Exception as stream_err:
                # mock/httpx 后端不支持流式或流中断 → 回退非流式
                logger.warning(
                    "[%s] 流式JSON调用失败，回退非流式: %s", self.name, stream_err
                )
        if raw_response is None:
            raw_response = self._llm.chat_json(
                messages=messages,
                temperature=self._temperature,
                max_tokens=max(self._max_tokens, 16384),
            )

        latency_ms = _now_ms() - start_ms

        # 3. 解析响应
        decision, structured_data = self._parse_response(raw_response)

        logger.info(
            "[%s] 决策: %s, 耗时: %dms",
            self.name,
            decision,
            latency_ms,
        )

        return AgentResult(
            agent_name=self.name,
            decision=decision,
            data=structured_data,
            latency_ms=latency_ms,
            success=True,
        )

    def _chat_json_stream(
        self,
        messages: List[LLMMessage],
        thinking_callback: Callable[[str], None],
    ) -> Dict[str, Any]:
        """流式调用 LLM 并收集 JSON 输出

        token 边产生边通过 thinking_callback 推送（前端逐字可见），
        全部收集完后用宽容解析器提取 JSON（容忍截断/围栏）。
        注意：content token 同时转发给回调，Agent 的结构化输出过程对前端实时可见。
        max_tokens 提到 16384：推理模型的思维链会占用完成预算，
        预算过小会导致正文为空或 JSON 被截断（解析失败回退非流式）。
        """
        buf: List[str] = []
        for kind, text in self._llm.chat_stream(
            messages=messages,
            temperature=self._temperature,
            max_tokens=max(self._max_tokens, 16384),
        ):
            if kind == "token":
                buf.append(text)
                thinking_callback(text)
            elif kind == "thinking":
                # 推理模型的思维链字段，同样流式展示
                thinking_callback(text)
        return parse_json_text("".join(buf))

    # ============================================================
    # 子类必须实现的方法
    # ============================================================

    def _build_prompt(self, context: AgentContext) -> List[LLMMessage]:
        """
        构建LLM提示词

        子类实现：根据context构建system+user消息列表。
        """
        raise NotImplementedError(f"{self.name} 未实现 _build_prompt")

    def _parse_response(self, response: dict) -> Tuple[str, Dict[str, Any]]:
        """
        解析LLM的JSON响应为结构化结果

        子类实现：从response dict中提取decision和structured_data。

        Returns:
            (decision摘要, 完整结构化数据)
        """
        raise NotImplementedError(f"{self.name} 未实现 _parse_response")


def _now_ms() -> int:
    """当前时间戳（毫秒）"""
    return int(time.time() * 1000)
