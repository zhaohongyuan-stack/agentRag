"""
幂等键管理器 — M5.4 网关模块

防止重复请求的重复执行。相同幂等键的请求返回缓存响应。

流程:
  1. 检查幂等键是否已存在 → 命中则返回缓存
  2. 不存在则设置幂等键（TTL 600s）
  3. 请求处理完成后缓存响应
  4. TTL 过期后允许重试

Redis Key: ace-rag:idempotent:{key} → String(JSON)
"""

import json
import logging
import time
from typing import Any, Optional

from ...memory.session_state.session_manager import _MockRedis

logger = logging.getLogger(__name__)

_IDEMPOTENCY_PREFIX = "ace-rag:idempotent:"
_DEFAULT_TTL = 600  # 10 分钟


class IdempotencyResult:
    """幂等检查结果"""

    def __init__(
        self,
        status: str,
        cached_response: Any = None,
        reason: str = "",
    ):
        # status: "cached" | "new" | "expired"
        self.status = status
        self.cached_response = cached_response
        self.reason = reason

    @property
    def is_cached(self) -> bool:
        return self.status == "cached"

    @property
    def is_new(self) -> bool:
        return self.status == "new"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "cached_response": self.cached_response,
            "reason": self.reason,
        }


class IdempotencyHandler:
    """
    幂等键管理器

    用法:
        handler = IdempotencyHandler()
        result = handler.check_or_cache("idem-key-123")
        if result.is_cached:
            return result.cached_response  # 返回缓存
        # 处理请求...
        handler.cache_response("idem-key-123", response_data)
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        ttl_seconds: int = _DEFAULT_TTL,
    ):
        self._client = redis_client if redis_client is not None else _MockRedis()
        self._ttl = ttl_seconds

    def check_or_cache(self, idempotency_key: str) -> IdempotencyResult:
        """
        检查幂等键

        如果键已存在且有缓存响应 → 返回 cached
        如果键不存在 → 设置键（标记为处理中），返回 new

        Args:
            idempotency_key: 幂等键

        Returns:
            IdempotencyResult
        """
        if not idempotency_key:
            return IdempotencyResult(status="new", reason="无幂等键，正常处理")

        key = f"{_IDEMPOTENCY_PREFIX}{idempotency_key}"
        raw = self._client.get(key)

        if raw is not None:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                cached = json.loads(raw)
                # 检查是否还在处理中
                if cached.get("status") == "processing":
                    return IdempotencyResult(
                        status="cached",
                        cached_response=None,
                        reason="请求正在处理中",
                    )
                return IdempotencyResult(
                    status="cached",
                    cached_response=cached.get("response"),
                )
            except json.JSONDecodeError:
                pass  # 数据损坏，继续设置为新请求

        # 设置为处理中
        payload = json.dumps(
            {"status": "processing", "created_at": time.time()},
            ensure_ascii=False,
        )
        self._client.set(key, payload, ex=self._ttl)

        return IdempotencyResult(status="new")

    def cache_response(
        self, idempotency_key: str, response: Any
    ) -> None:
        """
        缓存响应

        请求处理完成后，将响应缓存到幂等键。

        Args:
            idempotency_key: 幂等键
            response: 响应数据（可序列化）
        """
        if not idempotency_key:
            return

        key = f"{_IDEMPOTENCY_PREFIX}{idempotency_key}"
        payload = json.dumps(
            {
                "status": "completed",
                "response": response,
                "completed_at": time.time(),
            },
            ensure_ascii=False,
        )
        self._client.set(key, payload, ex=self._ttl)
        logger.debug("缓存幂等响应: key=%s", idempotency_key)

    def check_expired(self, idempotency_key: str) -> bool:
        """
        检查幂等键是否已过期（不存在）

        Args:
            idempotency_key: 幂等键

        Returns:
            True 表示已过期（允许重试）
        """
        key = f"{_IDEMPOTENCY_PREFIX}{idempotency_key}"
        return not self._client.exists(key)

    def delete(self, idempotency_key: str) -> None:
        """删除幂等键"""
        key = f"{_IDEMPOTENCY_PREFIX}{idempotency_key}"
        self._client.delete(key)

    def clear_all(self) -> None:
        """清除所有幂等键（仅 Mock 模式）"""
        if isinstance(self._client, _MockRedis):
            keys_to_delete = [
                k for k in self._client._data if k.startswith(_IDEMPOTENCY_PREFIX)
            ]
            for k in keys_to_delete:
                self._client.delete(k)
