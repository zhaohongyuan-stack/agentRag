"""计划缓存模块 — M4.1

缓存编译好的物理计划，避免重复编译，支持 TTL 过期与按知识库版本失效。

主要组件:
  - PlanCache: 计划缓存（Redis 优先，内存回退）
  - CacheKeyGenerator: 缓存键生成器
  - CacheContext: 缓存上下文（权限 / 知识库版本 / 约束哈希）
"""

from .cache import PlanCache
from .cache_key import CacheContext, CacheKeyGenerator

__all__ = [
    "PlanCache",
    "CacheKeyGenerator",
    "CacheContext",
]
