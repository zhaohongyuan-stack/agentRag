"""
权限检查器 — M5.3 工具模块

在工具调用前检查调用者是否有权限使用该工具。

权限级别:
  - public: 任何调用者可用
  - internal: 需要认证的调用者可用
  - restricted: 仅限特定角色可用

调用者角色:
  - anonymous: 匿名用户
  - authenticated: 已认证用户
  - admin: 管理员
"""

import logging
from typing import Optional

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)

# 权限级别优先级（数字越大权限越高）
_PERMISSION_PRIORITY = {
    "public": 0,
    "internal": 1,
    "restricted": 2,
}

# 调用者角色优先级
_CALLER_PRIORITY = {
    "anonymous": 0,
    "authenticated": 1,
    "admin": 2,
}


class PermissionChecker:
    """
    权限检查器

    检查调用者是否有权限调用指定工具。

    规则:
      - public 工具: 所有角色可用
      - internal 工具: 需 authenticated 或 admin
      - restricted 工具: 仅 admin 可用

    用法:
        checker = PermissionChecker()
        allowed, reason = checker.check(manifest, caller_role="authenticated")
        if not allowed:
            raise PermissionError(reason)
    """

    def check(
        self, manifest: ToolManifest, caller_role: str = "anonymous"
    ) -> tuple:
        """
        检查权限

        Args:
            manifest: 工具清单
            caller_role: 调用者角色

        Returns:
            (allowed, reason) — allowed 为 True 时 reason 为空
        """
        required_level = manifest.permission_level
        caller_level = _CALLER_PRIORITY.get(caller_role, 0)
        tool_level = _PERMISSION_PRIORITY.get(required_level, 0)

        if caller_level >= tool_level:
            return True, ""

        reason = (
            f"权限不足: 工具 '{manifest.name}' 要求 '{required_level}' 级别，"
            f"调用者角色 '{caller_role}' 不满足"
        )
        logger.warning(reason)
        return False, reason
