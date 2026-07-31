"""
工具注册与调用平台 — M5.3/M5.4 工具模块

模块组成:
  - ToolRegistry: 工具注册表（注册、查询、Schema 校验）
  - PermissionChecker: 权限检查器
  - ToolExecutor: 工具调用执行器（校验→权限→幂等→执行→降级→日志）
  - ToolCallingAgent: 工具调用智能体（LLM 决策→工具执行→结果回传→最终回答）
  - RedisToolCache: Redis 工具结果缓存
  - 内置工具: calculator(11种运算), version_checker, date_parser
  - 检索工具: search_chunks, get_chunk_content, search_table, get_cell_value, list_documents

用法（完整智能体）:
    from agent_platform.tools import create_agent

    agent = create_agent(retrieval_base_url="http://127.0.0.1:8000")
    result = agent.run("2023年10月人身险公司原保险保费收入是多少？")
    print(result.answer)

用法（仅工具平台）:
    from agent_platform.tools import create_default_platform
    platform = create_default_platform()
    result = platform.invoke("calculator", {"operation": "add", "a": 1, "b": 2})
"""

from .adapters.calculator import CALCULATOR_MANIFEST, calculator_handler
from .adapters.date_parser import DATE_PARSER_MANIFEST, date_parser_handler
from .adapters.retrieval_tools import RetrievalToolFactory
from .adapters.version_checker import (
    VERSION_CHECKER_MANIFEST,
    version_checker_handler,
)
from .agent import AgentResult, RedisToolCache, ToolCallingAgent
from .executor.executor import ToolExecutor
from .permissions.checker import PermissionChecker
from .registry.registry import ToolRegistry
from .tool_models import RetryPolicy, ToolEvent, ToolManifest, ToolResult

__all__ = [
    # 数据模型
    "RetryPolicy",
    "ToolManifest",
    "ToolResult",
    "ToolEvent",
    # 核心组件
    "ToolRegistry",
    "PermissionChecker",
    "ToolExecutor",
    # 智能体
    "ToolCallingAgent",
    "AgentResult",
    "RedisToolCache",
    # 内置工具
    "CALCULATOR_MANIFEST",
    "calculator_handler",
    "VERSION_CHECKER_MANIFEST",
    "version_checker_handler",
    "DATE_PARSER_MANIFEST",
    "date_parser_handler",
    # 检索工具
    "RetrievalToolFactory",
    # 工厂函数
    "create_default_platform",
    "create_agent",
]


def create_default_platform() -> ToolExecutor:
    """
    创建默认工具平台（已注册所有内置工具）

    注册: calculator, version_checker, date_parser
    不含检索工具（检索工具需要 RetrievalClient 配置）。

    Returns:
        ToolExecutor 实例
    """
    registry = ToolRegistry()
    registry.register(CALCULATOR_MANIFEST, calculator_handler)
    registry.register(VERSION_CHECKER_MANIFEST, version_checker_handler)
    registry.register(DATE_PARSER_MANIFEST, date_parser_handler)

    executor = ToolExecutor(registry)
    return executor


def create_agent(
    retrieval_base_url: str = "http://127.0.0.1:8000",
    max_tool_calls: int = 5,
    use_cache: bool = True,
    llm_client=None,
) -> ToolCallingAgent:
    """
    创建完整的工具调用智能体

    注册全部工具:
      - calculator (11 种运算)
      - search_chunks, get_chunk_content, search_table, get_cell_value, list_documents
      - version_checker, date_parser

    Args:
        retrieval_base_url: 检索服务地址
        max_tool_calls: 最大工具调用轮数
        use_cache: 是否启用 Redis 工具结果缓存
        llm_client: LLM 客户端，None 时使用全局单例

    Returns:
        ToolCallingAgent 实例
    """
    from ..gateway.request_handler.retrieval_client import RetrievalClient
    from ..runtime.llm_client import get_llm_client

    # 1. 创建工具注册表
    registry = ToolRegistry()

    # 2. 注册内置工具
    registry.register(CALCULATOR_MANIFEST, calculator_handler)
    registry.register(VERSION_CHECKER_MANIFEST, version_checker_handler)
    registry.register(DATE_PARSER_MANIFEST, date_parser_handler)

    # 3. 注册检索工具
    retrieval_client = RetrievalClient(base_url=retrieval_base_url)
    factory = RetrievalToolFactory(retrieval_client=retrieval_client)
    factory.register_all(registry)

    # 4. 创建执行器
    executor = ToolExecutor(registry)

    # 5. 创建缓存（可选）
    cache = RedisToolCache() if use_cache else None

    # 6. 创建 LLM 客户端
    if llm_client is None:
        llm_client = get_llm_client()

    # 7. 创建智能体
    agent = ToolCallingAgent(
        llm_client=llm_client,
        tool_executor=executor,
        max_tool_calls=max_tool_calls,
        cache=cache,
    )

    return agent
