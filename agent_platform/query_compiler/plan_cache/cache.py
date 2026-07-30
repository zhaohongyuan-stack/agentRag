"""
计划缓存 — 物理计划的复用缓存（Redis 优先，内存回退）

职责:
  1. 按 cache_key 缓存编译好的 PhysicalPlan，避免重复编译
  2. 支持 TTL 自动过期（默认 30 分钟）
  3. 支持按知识库版本（epoch）批量失效
  4. 提供命中/未命中统计

部署模式:
  - 传入 redis_client: 生产模式，计划序列化为 JSON 存入 Redis
  - redis_client=None: 开发/测试模式，使用内存字典模拟（进程内有效）

注意: Redis 为可选依赖，不可用时自动回退到内存字典，不硬依赖 Redis。

模式参考: physical_planner/planner.py 的 dataclass + to_dict 风格
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..physical_planner.planner import PhysicalPlan, PlanStage
from ..query_ir.ir_builder import StopCondition


logger = logging.getLogger(__name__)


# 默认 TTL（秒）：30 分钟
DEFAULT_TTL: int = 1800


@dataclass
class _MemoryEntry:
    """
    内存缓存条目

    Attributes:
        plan: 缓存的物理计划对象
        expiry: 过期时间戳（time.time），None 表示永不过期
        epoch: 关联的知识库版本（用于按版本失效）
    """

    plan: PhysicalPlan
    expiry: Optional[float] = None
    epoch: str = ""


class PlanCache:
    """
    物理计划缓存

    redis_client 为 None 时使用内存字典模拟（开发/测试用），
    传入 Redis 客户端时使用 Redis 持久化缓存。

    用法:
        # 内存模式
        cache = PlanCache()
        cache.put(key, plan)
        cached = cache.get(key)

        # Redis 模式
        import redis
        cache = PlanCache(redis_client=redis.Redis(...))
        cache.put(key, plan, ttl=1800, epoch="kb-2026-07")
    """

    def __init__(
        self,
        redis_client: Any = None,
        key_prefix: str = "plancache:",
    ):
        """
        Args:
            redis_client: Redis 客户端实例，为 None 时使用内存字典模拟。
            key_prefix: Redis 键前缀，便于隔离命名空间。
        """
        self._redis = redis_client
        self._key_prefix = key_prefix

        # 内存模式专用状态
        self._store: Dict[str, _MemoryEntry] = {}
        self._epoch_index: Dict[str, Set[str]] = {}

        # 统计计数器（两种模式共用，进程内有效）
        self._hits = 0
        self._misses = 0

        mode = "redis" if self._redis is not None else "memory"
        logger.info("PlanCache 初始化完成，模式=%s", mode)

    # ============================================================
    # 公共接口
    # ============================================================

    def get(self, cache_key: str) -> Optional[PhysicalPlan]:
        """
        从缓存获取计划

        命中且未过期时返回 PhysicalPlan，否则返回 None 并计为未命中。

        Args:
            cache_key: 缓存键

        Returns:
            缓存的物理计划，未命中时返回 None
        """
        if self._redis is not None:
            return self._redis_get(cache_key)
        return self._memory_get(cache_key)

    def put(
        self,
        cache_key: str,
        plan: PhysicalPlan,
        ttl: int = DEFAULT_TTL,
        epoch: str = "",
    ) -> None:
        """
        缓存计划

        Args:
            cache_key: 缓存键
            plan: 待缓存的物理计划
            ttl: 生存时间（秒），默认 1800（30 分钟）；<=0 表示永不过期
            epoch: 知识库版本标识，传入后支持 invalidate_by_epoch 批量失效
        """
        if self._redis is not None:
            self._redis_put(cache_key, plan, ttl, epoch)
        else:
            self._memory_put(cache_key, plan, ttl, epoch)

    def invalidate_by_epoch(self, old_epoch: str) -> int:
        """
        知识库版本更新时批量失效

        删除所有关联到 old_epoch 的缓存条目。
        需在 put 时传入对应 epoch 才能被索引到。

        Args:
            old_epoch: 旧的知识库版本标识

        Returns:
            失效的条目数量
        """
        if self._redis is not None:
            return self._redis_invalidate_by_epoch(old_epoch)
        return self._memory_invalidate_by_epoch(old_epoch)

    def clear(self) -> None:
        """
        清空所有缓存

        移除全部缓存条目与版本索引（统计计数器保留）。
        """
        if self._redis is not None:
            self._redis_clear()
        else:
            self._store.clear()
            self._epoch_index.clear()
        logger.info("PlanCache 已清空所有缓存")

    def stats(self) -> dict:
        """
        缓存统计

        Returns:
            包含命中数、未命中数、当前缓存大小的字典
        """
        size = self._size()
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": size,
            "hit_rate": round(hit_rate, 4),
        }

    # ============================================================
    # 内存模式实现
    # ============================================================

    def _memory_get(self, cache_key: str) -> Optional[PhysicalPlan]:
        """内存模式：获取计划"""
        entry = self._store.get(cache_key)
        if entry is None:
            self._misses += 1
            return None
        # 检查过期
        if entry.expiry is not None and time.time() >= entry.expiry:
            self._evict_memory(cache_key)
            self._misses += 1
            logger.debug("缓存键过期: %s", cache_key[:12] + "...")
            return None
        self._hits += 1
        return entry.plan

    def _memory_put(
        self,
        cache_key: str,
        plan: PhysicalPlan,
        ttl: int,
        epoch: str,
    ) -> None:
        """内存模式：写入计划"""
        expiry = None if ttl <= 0 else time.time() + ttl
        self._store[cache_key] = _MemoryEntry(
            plan=plan, expiry=expiry, epoch=epoch
        )
        # 维护版本索引
        if epoch:
            self._epoch_index.setdefault(epoch, set()).add(cache_key)
        logger.debug(
            "内存缓存写入: key=%s, ttl=%s, epoch=%s",
            cache_key[:12] + "...",
            ttl,
            epoch or "-",
        )

    def _memory_invalidate_by_epoch(self, old_epoch: str) -> int:
        """内存模式：按版本批量失效"""
        keys = self._epoch_index.pop(old_epoch, set())
        count = 0
        for key in keys:
            if self._evict_memory(key):
                count += 1
        logger.info(
            "按版本失效完成: epoch=%s, 失效条目=%d", old_epoch, count
        )
        return count

    def _evict_memory(self, cache_key: str) -> bool:
        """内存模式：移除单个条目并清理版本索引"""
        entry = self._store.pop(cache_key, None)
        if entry is None:
            return False
        if entry.epoch and entry.epoch in self._epoch_index:
            self._epoch_index[entry.epoch].discard(cache_key)
            if not self._epoch_index[entry.epoch]:
                self._epoch_index.pop(entry.epoch, None)
        return True

    def _memory_size(self) -> int:
        """内存模式：有效条目数（清理过期后统计）"""
        now = time.time()
        # 清理过期条目
        expired = [
            k
            for k, e in self._store.items()
            if e.expiry is not None and now >= e.expiry
        ]
        for k in expired:
            self._evict_memory(k)
        return len(self._store)

    # ============================================================
    # Redis 模式实现
    # ============================================================

    def _redis_key(self, cache_key: str) -> str:
        """拼接完整 Redis 键"""
        return f"{self._key_prefix}{cache_key}"

    def _redis_epoch_set(self, epoch: str) -> str:
        """版本索引集合的 Redis 键"""
        return f"{self._key_prefix}epoch:{epoch}"

    def _redis_get(self, cache_key: str) -> Optional[PhysicalPlan]:
        """Redis 模式：获取计划"""
        try:
            raw = self._redis.get(self._redis_key(cache_key))
        except Exception as exc:  # Redis 异常时回退为未命中
            logger.warning("Redis 读取失败，视为未命中: %s", exc)
            self._misses += 1
            return None
        if raw is None:
            self._misses += 1
            return None
        try:
            data = json.loads(raw)
            self._hits += 1
            return self._plan_from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("缓存计划反序列化失败，视为未命中: %s", exc)
            self._misses += 1
            return None

    def _redis_put(
        self,
        cache_key: str,
        plan: PhysicalPlan,
        ttl: int,
        epoch: str,
    ) -> None:
        """Redis 模式：写入计划"""
        redis_key = self._redis_key(cache_key)
        payload = json.dumps(plan.to_dict(), ensure_ascii=False)
        try:
            if ttl <= 0:
                self._redis.set(redis_key, payload)
            else:
                self._redis.setex(redis_key, ttl, payload)
            # 维护版本索引
            if epoch:
                self._redis.sadd(self._redis_epoch_set(epoch), redis_key)
            logger.debug(
                "Redis 缓存写入: key=%s, ttl=%s, epoch=%s",
                cache_key[:12] + "...",
                ttl,
                epoch or "-",
            )
        except Exception as exc:
            logger.warning("Redis 写入失败: %s", exc)

    def _redis_invalidate_by_epoch(self, old_epoch: str) -> int:
        """Redis 模式：按版本批量失效"""
        set_key = self._redis_epoch_set(old_epoch)
        try:
            keys = self._redis.smembers(set_key)
            if not keys:
                return 0
            # 兼容 bytes / str 键
            decoded = [
                k.decode("utf-8") if isinstance(k, bytes) else k
                for k in keys
            ]
            if decoded:
                self._redis.delete(*decoded)
            self._redis.delete(set_key)
            count = len(decoded)
            logger.info(
                "按版本失效完成: epoch=%s, 失效条目=%d", old_epoch, count
            )
            return count
        except Exception as exc:
            logger.warning("Redis 按版本失效失败: %s", exc)
            return 0

    def _redis_clear(self) -> None:
        """Redis 模式：清空前缀下所有键"""
        try:
            count = 0
            for key in self._redis.scan_iter(
                match=f"{self._key_prefix}*", count=100
            ):
                self._redis.delete(key)
                count += 1
            logger.debug("Redis 清空完成，删除键数=%d", count)
        except Exception as exc:
            logger.warning("Redis 清空失败: %s", exc)

    def _redis_size(self) -> int:
        """Redis 模式：统计前缀下键数"""
        try:
            return sum(
                1 for _ in self._redis.scan_iter(
                    match=f"{self._key_prefix}*", count=100
                )
            )
        except Exception as exc:
            logger.warning("Redis 统计大小失败: %s", exc)
            return 0

    # ============================================================
    # 序列化辅助
    # ============================================================

    def _size(self) -> int:
        """获取当前缓存大小（按模式分发）"""
        if self._redis is not None:
            return self._redis_size()
        return self._memory_size()

    @staticmethod
    def _plan_from_dict(data: Dict[str, Any]) -> PhysicalPlan:
        """
        从字典重建 PhysicalPlan

        与 PhysicalPlan.to_dict() 对应，用于 Redis 反序列化。
        """
        stages: List[PlanStage] = []
        for s in data.get("stages", []):
            stages.append(
                PlanStage(
                    name=s.get("name", ""),
                    channels=list(s.get("channels", [])),
                    top_k=s.get("top_k", 10),
                    rerank=s.get("rerank", False),
                    timeout_ms=s.get("timeout_ms", 5000),
                    operations=list(s.get("operations", [])),
                    condition=s.get("condition", ""),
                    claim_ids=list(s.get("claim_ids", [])),
                )
            )

        stop_conditions = [
            StopCondition(
                condition=sc.get("condition", ""),
                description=sc.get("description", ""),
            )
            for sc in data.get("stop_conditions", [])
        ]

        return PhysicalPlan(
            plan_id=data.get("plan_id", ""),
            intent=data.get("intent", ""),
            stages=stages,
            stop_conditions=stop_conditions,
            budget_ms=data.get("budget_ms", 0),
        )
