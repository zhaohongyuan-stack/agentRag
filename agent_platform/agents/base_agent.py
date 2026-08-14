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
from typing import Any, Dict, List, Tuple

from agent_platform.runtime.llm_client import LLMClient, LLMMessage, get_llm_client

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

    def run(self, context: AgentContext) -> AgentResult:
        """
        统一入口：构建提示词 -> 调用LLM -> 解析JSON -> 返回AgentResult

        Args:
            context: Agent间共享上下文

        Returns:
            AgentResult统一结构化输出

        Raises:
            Exception: LLM调用失败时直接报错（不重试）
        """
        start_ms = _now_ms()

        # 1. 构建提示词
        messages = self._build_prompt(context)

        # 2. 调用LLM（chat_json确保返回dict）
        raw_response = self._llm.chat_json(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
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
