"""
工具注册与调用平台 — M5.3 工具模块

模块组成:
  - ToolRegistry: 工具注册表（注册、查询、Schema 校验）
  - PermissionChecker: 权限检查器
  - ToolExecutor: 工具调用执行器（校验→权限→幂等→执行→降级→日志）
  - 内置工具: calculator, version_checker, date_parser

用法:
    from agent_platform.tools import create_default_platform
    platform = create_default_platform()
    result = platform.invoke("calculator", {"expression": "8 * 1.25"})
"""

from .adapters.calculator import CALCULATOR_MANIFEST, calculator_handler
from .adapters.date_parser import DATE_PARSER_MANIFEST, date_parser_handler
from .adapters.version_checker import (
    VERSION_CHECKER_MANIFEST,
    version_checker_handler,
)
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
    # 内置工具
    "CALCULATOR_MANIFEST",
    "calculator_handler",
    "VERSION_CHECKER_MANIFEST",
    "version_checker_handler",
    "DATE_PARSER_MANIFEST",
    "date_parser_handler",
    # 工厂函数
    "create_default_platform",
]


def create_default_platform() -> ToolExecutor:
    """
    创建默认工具平台（已注册所有内置工具）

    Returns:
        ToolExecutor 实例，已注册 calculator, version_checker, date_parser
    """
    registry = ToolRegistry()
    registry.register(CALCULATOR_MANIFEST, calculator_handler)
    registry.register(VERSION_CHECKER_MANIFEST, version_checker_handler)
    registry.register(DATE_PARSER_MANIFEST, date_parser_handler)

    executor = ToolExecutor(registry)
    return executor
