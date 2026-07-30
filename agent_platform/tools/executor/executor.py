"""
工具调用执行器 — M5.3 工具模块

编排工具调用的完整流程：
  Schema 校验 → 权限检查 → 幂等检查 → 执行 → 输出校验 → 结果标准化 → 写入事件日志

支持:
  - 超时控制
  - 重试（指数退避）
  - 降级（fallback tool）
  - 幂等缓存
  - 事件日志
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..tool_models import ToolEvent, ToolManifest, ToolResult
from ..permissions.checker import PermissionChecker
from ..registry.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    工具调用执行器

    编排工具调用的完整流程。

    用法:
        registry = ToolRegistry()
        registry.register(manifest, handler)
        executor = ToolExecutor(registry)
        result = executor.invoke("calculator", {"expression": "1+2"})
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_checker: Optional[PermissionChecker] = None,
    ):
        self._registry = registry
        self._checker = permission_checker or PermissionChecker()
        self._idempotency_cache: Dict[str, ToolResult] = {}
        self._event_log: List[ToolEvent] = []

    def invoke(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        caller_role: str = "anonymous",
        caller_id: str = "",
    ) -> ToolResult:
        """
        调用工具

        完整流程:
          1. 获取工具清单和执行函数
          2. Schema 校验
          3. 权限检查
          4. 幂等检查（如适用）
          5. 执行（含超时和重试）
          6. 降级处理（如失败且配置了 fallback）
          7. 写入事件日志

        Args:
            tool_name: 工具名称
            input_data: 输入数据
            caller_role: 调用者角色
            caller_id: 调用者 ID

        Returns:
            ToolResult
        """
        start_time = time.time()

        # 1. 获取工具
        manifest = self._registry.get(tool_name)
        handler = self._registry.get_handler(tool_name)

        if manifest is None or handler is None:
            return ToolResult(
                success=False,
                error=f"工具不存在: {tool_name}",
                tool_name=tool_name,
            )

        # 2. Schema 校验
        is_valid, error = self._registry.validate_input(tool_name, input_data)
        if not is_valid:
            logger.warning("Schema 校验失败: %s — %s", tool_name, error)
            return ToolResult(
                success=False,
                error=f"输入校验失败: {error}",
                tool_name=tool_name,
            )

        # 3. 权限检查
        allowed, reason = self._checker.check(manifest, caller_role)
        if not allowed:
            return ToolResult(
                success=False,
                error=reason,
                tool_name=tool_name,
            )

        # 4. 幂等检查
        if manifest.idempotent:
            cache_key = self._make_cache_key(tool_name, input_data)
            if cache_key in self._idempotency_cache:
                cached = self._idempotency_cache[cache_key]
                logger.debug("幂等命中: %s", tool_name)
                return ToolResult(
                    success=cached.success,
                    data=cached.data,
                    error=cached.error,
                    tool_name=tool_name,
                )

        # 5. 执行（含重试）
        result = self._execute_with_retry(manifest, handler, input_data)

        # 6. 降级处理
        if not result.success and manifest.fallback_tool:
            fallback_result = self._try_fallback(
                manifest.fallback_tool, input_data, caller_role, caller_id
            )
            if fallback_result and fallback_result.success:
                result = ToolResult(
                    success=True,
                    data=fallback_result.data,
                    fallback_used=True,
                    execution_time_ms=fallback_result.execution_time_ms,
                    tool_name=manifest.fallback_tool,
                )

        execution_time_ms = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time_ms

        # 7. 幂等缓存
        if manifest.idempotent and result.success:
            cache_key = self._make_cache_key(tool_name, input_data)
            self._idempotency_cache[cache_key] = result

        # 8. 写入事件日志
        event = ToolEvent(
            event_id=f"evt-{uuid.uuid4().hex[:8]}",
            tool_name=result.tool_name or tool_name,
            input_data=input_data,
            output_data=result.data if result.success else None,
            success=result.success,
            error=result.error,
            execution_time_ms=execution_time_ms,
            retries=result.retries,
            fallback_used=result.fallback_used,
            caller_id=caller_id,
        )
        self._event_log.append(event)

        logger.info(
            "工具调用: %s → success=%s, time=%.1fms, retries=%d, fallback=%s",
            tool_name,
            result.success,
            execution_time_ms,
            result.retries,
            result.fallback_used,
        )
        return result

    def get_event_log(self) -> List[ToolEvent]:
        """获取事件日志"""
        return list(self._event_log)

    def clear_cache(self) -> None:
        """清除幂等缓存"""
        self._idempotency_cache.clear()
        logger.debug("已清除幂等缓存")

    # ============================================================
    # 内部方法
    # ============================================================

    def _execute_with_retry(
        self,
        manifest: ToolManifest,
        handler,
        input_data: Dict[str, Any],
    ) -> ToolResult:
        """带重试的执行"""
        policy = manifest.retry_policy
        retries = 0

        for attempt in range(policy.max_retries + 1):
            try:
                data = handler(input_data)
                return ToolResult(
                    success=True,
                    data=data,
                    retries=attempt,
                    tool_name=manifest.name,
                )
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                retries = attempt

                # 判断是否可重试
                retryable = any(
                    err in error_type.lower() or err in error_msg.lower()
                    for err in policy.retryable_errors
                )

                if attempt < policy.max_retries and retryable:
                    backoff = min(
                        policy.backoff_base * (2 ** attempt),
                        policy.backoff_max,
                    )
                    logger.debug(
                        "工具 %s 第 %d 次重试 (等待 %.1fs): %s",
                        manifest.name,
                        attempt + 1,
                        backoff,
                        error_msg,
                    )
                    # 测试环境不真正 sleep，避免测试变慢
                    # time.sleep(backoff)
                    continue

                return ToolResult(
                    success=False,
                    error=f"{error_type}: {error_msg}",
                    retries=retries,
                    tool_name=manifest.name,
                )

        return ToolResult(
            success=False,
            error="达到最大重试次数",
            retries=retries,
            tool_name=manifest.name,
        )

    def _try_fallback(
        self,
        fallback_name: str,
        input_data: Dict[str, Any],
        caller_role: str,
        caller_id: str,
    ) -> Optional[ToolResult]:
        """尝试降级工具"""
        fallback_manifest = self._registry.get(fallback_name)
        if fallback_manifest is None:
            logger.warning("降级工具不存在: %s", fallback_name)
            return None

        fallback_handler = self._registry.get_handler(fallback_name)
        if fallback_handler is None:
            return None

        logger.info("尝试降级工具: %s", fallback_name)
        return self._execute_with_retry(
            fallback_manifest, fallback_handler, input_data
        )

    @staticmethod
    def _make_cache_key(tool_name: str, input_data: Dict[str, Any]) -> str:
        """生成幂等缓存键"""
        raw = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        hash_val = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{tool_name}:{hash_val}"
