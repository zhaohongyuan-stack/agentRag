"""
滑动窗口限流器 — M5.4 网关模块

基于 Redis 的滑动窗口限流，按用户/API Key 限制请求频率。

限流策略:
  - anonymous: 10 次/分钟
  - authenticated: 100 次/分钟
  - premium: 500 次/分钟

实现方式: Redis Sorted Set（按时间戳打分，清理过期请求，计数判断）
"""

import logging
import time
from typing import Any, Optional

from ...memory.session_state.session_manager import _MockRedis

logger = logging.getLogger(__name__)

# 默认限流配置
RATE_LIMITS = {
    "anonymous": {"requests": 10, "window_seconds": 60},
    "authenticated": {"requests": 100, "window_seconds": 60},
    "premium": {"requests": 500, "window_seconds": 60},
}

_RATE_PREFIX = "ace-rag:ratelimit:"


class RateLimitResult:
    """限流检查结果"""

    def __init__(
        self,
        allowed: bool,
        remaining: int = 0,
        reset_at: float = 0.0,
        reason: str = "",
    ):
        self.allowed = allowed
        self.remaining = remaining
        self.reset_at = reset_at
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "reason": self.reason,
        }


class RateLimiter:
    """
    滑动窗口限流器

    用法:
        limiter = RateLimiter()
        result = limiter.check("user-123", "authenticated")
        if not result.allowed:
            return 429  # Too Many Requests
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        limits: Optional[dict] = None,
    ):
        self._client = redis_client if redis_client is not None else _MockRedis()
        self._limits = limits or RATE_LIMITS

    def check(
        self,
        identifier: str,
        tier: str = "anonymous",
    ) -> RateLimitResult:
        """
        检查请求是否被限流

        Args:
            identifier: 请求标识（用户 ID、API Key、IP 等）
            tier: 用户级别（anonymous / authenticated / premium）

        Returns:
            RateLimitResult
        """
        config = self._limits.get(tier, self._limits["anonymous"])
        max_requests = config["requests"]
        window = config["window_seconds"]

        key = f"{_RATE_PREFIX}{tier}:{identifier}"
        now = time.time()
        window_start = now - window

        if isinstance(self._client, _MockRedis):
            return self._check_mock(key, now, window_start, max_requests, window)
        else:
            return self._check_redis(key, now, window_start, max_requests, window)

    def _check_mock(
        self,
        key: str,
        now: float,
        window_start: float,
        max_requests: int,
        window: int,
    ) -> RateLimitResult:
        """Mock Redis 限流检查"""
        # 使用列表模拟 Sorted Set
        if key not in self._client._data:
            self._client._data[key] = []
            self._client._type[key] = "list"

        timestamps = self._client._data[key]
        if not isinstance(timestamps, list):
            timestamps = []

        # 清理过期请求
        timestamps = [t for t in timestamps if t > window_start]

        # 检查是否超限
        if len(timestamps) >= max_requests:
            oldest = min(timestamps) if timestamps else now
            reset_at = oldest + window
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                reason="请求频率超限",
            )

        # 记录本次请求
        timestamps.append(now)
        self._client._data[key] = timestamps
        self._client._expiry[key] = now + window

        remaining = max_requests - len(timestamps)
        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_at=now + window,
        )

    def _check_redis(
        self,
        key: str,
        now: float,
        window_start: float,
        max_requests: int,
        window: int,
    ) -> RateLimitResult:
        """Redis 限流检查（Sorted Set 实现）"""
        pipe = self._client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)
        results = pipe.execute()

        count = results[2]
        if count > max_requests:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + window,
                reason="请求频率超限",
            )

        return RateLimitResult(
            allowed=True,
            remaining=max_requests - count,
            reset_at=now + window,
        )

    def reset(self, identifier: str, tier: str = "anonymous") -> None:
        """重置限流计数"""
        key = f"{_RATE_PREFIX}{tier}:{identifier}"
        self._client.delete(key)
