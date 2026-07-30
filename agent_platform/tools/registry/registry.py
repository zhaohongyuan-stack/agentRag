"""
工具注册表 — M5.3 工具模块

管理工具的注册、查询和注销。每个工具由 ToolManifest 定义，
并绑定一个可调用的执行函数。

职责:
  1. 注册工具（manifest + handler）
  2. 按名称查询工具
  3. 列出所有已注册工具
  4. 注销工具
  5. Schema 校验
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册表

    管理工具的注册和查询。每个工具包含:
      - ToolManifest: 元信息（名称、Schema、权限等）
      - handler: 可调用执行函数

    用法:
        registry = ToolRegistry()
        registry.register(manifest, handler)
        manifest = registry.get("calculator")
        handler = registry.get_handler("calculator")
    """

    def __init__(self):
        self._manifests: Dict[str, ToolManifest] = {}
        self._handlers: Dict[str, Callable] = {}

    def register(
        self, manifest: ToolManifest, handler: Callable
    ) -> None:
        """
        注册工具

        Args:
            manifest: 工具清单
            handler: 可调用执行函数，签名: handler(input_data: dict) -> Any

        Raises:
            ValueError: 工具名称已存在
        """
        name = manifest.name
        if name in self._manifests:
            raise ValueError(f"工具已存在: {name}")

        self._manifests[name] = manifest
        self._handlers[name] = handler
        logger.info("注册工具: %s v%s", name, manifest.version)

    def unregister(self, name: str) -> bool:
        """
        注销工具

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        if name not in self._manifests:
            return False
        del self._manifests[name]
        del self._handlers[name]
        logger.info("注销工具: %s", name)
        return True

    def get(self, name: str) -> Optional[ToolManifest]:
        """获取工具清单"""
        return self._manifests.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        """获取工具执行函数"""
        return self._handlers.get(name)

    def list_tools(self) -> List[ToolManifest]:
        """列出所有已注册工具"""
        return list(self._manifests.values())

    def list_names(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._manifests.keys())

    def exists(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._manifests

    def validate_input(
        self, name: str, input_data: Dict[str, Any]
    ) -> tuple:
        """
        校验输入数据是否符合工具的 input_schema

        简化校验: 检查 required 字段是否存在

        Args:
            name: 工具名称
            input_data: 输入数据

        Returns:
            (is_valid, error_message)
        """
        manifest = self.get(name)
        if manifest is None:
            return False, f"工具不存在: {name}"

        schema = manifest.input_schema
        if not schema:
            return True, ""

        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required_fields:
            if field_name not in input_data:
                return False, f"缺少必填字段: {field_name}"

            # 类型检查
            field_schema = properties.get(field_name, {})
            expected_type = field_schema.get("type", "")
            if expected_type and not self._check_type(
                input_data[field_name], expected_type
            ):
                return False, (
                    f"字段 '{field_name}' 类型错误: "
                    f"期望 {expected_type}"
                )

        return True, ""

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """检查值类型是否符合 JSON Schema 类型"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_python = type_map.get(expected_type)
        if expected_python is None:
            return True
        return isinstance(value, expected_python)
