"""
Redis 工具结果缓存 — M5.4 工具调用智能体模块

为 ToolCallingAgent 提供工具调用结果的缓存层。
同一工具 + 相同参数的调用直接返回缓存结果，避免重复执行。

设计要点:
  1. Redis 不可用时自动降级为内存缓存（与 RedisSessionManager 一致）
  2. 缓存键: tool_name + sorted(input_data) 的 MD5 哈希
  3. 可配置 TTL（默认 300 秒 = 5 分钟）
  4. 仅缓存成功结果（success=True），失败结果不缓存

Redis Key 设计:
  ace-rag:tool_cache:{tool_name}:{hash}  → String(JSON): 工具结果
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Redis Key 前缀
_TOOL_CACHE_PREFIX = "ace-rag:tool_cache:"

# 默认 TTL（5 分钟）
_DEFAULT_TTL = 300


class _MemoryCache:
    """
    内存缓存 — Redis 不可用时的降级方案

    使用字典 + 过期时间戳模拟 Redis 的 GET/SET/DELETE。
    非线程安全，仅用于单线程开发/测试。
    """

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._expiry: Dict[str, Optional[float]] = {}

    def _check_expired(self, key: str) -> bool:
        exp = self._expiry.get(key)
        if exp is not None and time.time() >= exp:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def get(self, key: str) -> Optional[str]:
        if self._check_expired(key):
            return None
        return self._data.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._data[key] = value
        self._expiry[key] = (time.time() + ex) if ex else None
        return True

    def delete(self, key: str) -> bool:
        existed = key in self._data
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        return existed

    def exists(self, key: str) -> bool:
        if self._check_expired(key):
            return False
        return key in self._data


class RedisToolCache:
    """
    Redis 工具结果缓存

    为工具调用提供结果缓存，避免相同参数的重复执行。
    当 Redis 不可用时自动降级为内存缓存。

    用法:
        cache = RedisToolCache()  # 自动从环境变量读取 Redis 配置
        cache = RedisToolCache(mock=True)  # 强制内存模式

        # 缓存命中检查
        cached = cache.get("search_chunks", {"query": "资本充足率"})
        if cached is not None:
            return cached

        # 执行后缓存
        cache.set("search_chunks", {"query": "资本充足率"}, result_dict, ttl=300)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: Optional[int] = None,
        mock: bool = False,
    ):
        """
        Args:
            redis_url: Redis 连接 URL（如 redis://:pass@host:6379/0）。
                       为 None 时从环境变量构造。
            ttl: 缓存默认 TTL（秒），None 时读取 TOOL_CACHE_TTL 环境变量（默认 300）
            mock: 强制使用内存模式（开发/测试）
        """
        self._ttl = (
            ttl
            if ttl is not None
            else int(os.getenv("TOOL_CACHE_TTL", str(_DEFAULT_TTL)))
        )
        self._mock = mock
        self._client = None

        if not mock:
            url = redis_url or self._build_redis_url()
            try:
                import redis

                self._client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_timeout=3,
                    socket_connect_timeout=3,
                )
                self._client.ping()
                logger.info("工具结果缓存已连接 Redis: %s", _safe_url(url))
            except Exception as e:
                logger.warning("Redis 连接失败，工具缓存降级为内存模式: %s", e)
                self._client = _MemoryCache()
                self._mock = True
        else:
            self._client = _MemoryCache()
            logger.info("工具结果缓存运行在内存模式（mock）")

    @staticmethod
    def _build_redis_url() -> str:
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        db = os.getenv("REDIS_DB", "0")
        password = os.getenv("REDIS_PASSWORD", "")
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

    @staticmethod
    def _make_cache_key(tool_name: str, input_data: Dict[str, Any]) -> str:
        """生成缓存键: tool_name + sorted(input_data) 的 MD5"""
        raw = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        hash_val = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{_TOOL_CACHE_PREFIX}{tool_name}:{hash_val}"

    def get(self, tool_name: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        查询缓存

        Args:
            tool_name: 工具名称
            input_data: 工具输入参数

        Returns:
            缓存的结果字典，未命中返回 None
        """
        key = self._make_cache_key(tool_name, input_data)
        raw = self._client.get(key)
        if raw is None:
            logger.debug("工具缓存未命中: %s", tool_name)
            return None
        logger.debug("工具缓存命中: %s", tool_name)
        return json.loads(raw)

    def set(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        写入缓存

        仅缓存成功结果（result.get("success", True) 不为 False）。

        Args:
            tool_name: 工具名称
            input_data: 工具输入参数
            result: 工具执行结果
            ttl: TTL（秒），None 时使用默认值
        """
        # 失败结果不缓存
        if isinstance(result, dict) and result.get("success") is False:
            logger.debug("工具结果为失败，跳过缓存: %s", tool_name)
            return

        key = self._make_cache_key(tool_name, input_data)
        self._client.set(
            key,
            json.dumps(result, ensure_ascii=False),
            ex=ttl or self._ttl,
        )
        logger.debug("工具结果已缓存: %s (TTL=%ds)", tool_name, ttl or self._ttl)

    def delete(self, tool_name: str, input_data: Dict[str, Any]) -> bool:
        """删除指定缓存项"""
        key = self._make_cache_key(tool_name, input_data)
        return self._client.delete(key)

    def clear_pattern(self, tool_name: Optional[str] = None) -> int:
        """
        清除缓存

        Args:
            tool_name: 指定工具名称，None 则清除全部工具缓存

        Returns:
            清除的缓存项数量（内存模式下精确计数，Redis 模式下为近似值）
        """
        if self._mock:
            pattern = f"{_TOOL_CACHE_PREFIX}{tool_name or ''}"
            keys_to_delete = [
                k for k in self._client._data if k.startswith(pattern)
            ]
            for k in keys_to_delete:
                self._client.delete(k)
            return len(keys_to_delete)
        else:
            # Redis 模式: 使用 SCAN + DELETE
            pattern = f"{_TOOL_CACHE_PREFIX}{tool_name or '*'}*"
            count = 0
            for key in self._client.scan_iter(match=pattern, count=100):
                self._client.delete(key)
                count += 1
            return count

    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def client(self):
        return self._client


def _safe_url(url: str) -> str:
    """脱敏 Redis URL（隐藏密码）"""
    if "@" in url:
        scheme, rest = url.split("://", 1)
        if ":" in rest.split("@", 1)[0]:
            creds, host_part = rest.split("@", 1)
            return f"{scheme}://***@{host_part}"
    return url
