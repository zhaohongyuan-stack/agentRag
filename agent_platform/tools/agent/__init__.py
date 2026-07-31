"""
工具调用智能体模块

组成:
  - ToolCallingAgent: 工具调用智能体（LLM 决策 → 工具执行 → 结果回传 → 最终回答）
  - RedisToolCache: 基于 Redis 的工具结果缓存（Redis 不可用时降级为内存）
  - AgentResult: 智能体运行结果

用法:
    from agent_platform.tools.agent import ToolCallingAgent, RedisToolCache

    agent = ToolCallingAgent(
        llm_client=llm_client,
        tool_executor=executor,
        max_tool_calls=5,
    )
    result = agent.run("2023年10月人身险公司原保险保费收入是多少？")
    print(result.answer)
"""

from .tool_cache import RedisToolCache
from .tool_calling_agent import AgentResult, ToolCallingAgent

__all__ = [
    "ToolCallingAgent",
    "AgentResult",
    "RedisToolCache",
]
