"""
Plan Cache 单元测试 (M4.1)

测试范围:
  - PlanCache: 缓存命中/未命中、TTL 过期、版本失效、清空、统计
  - CacheKeyGenerator: 缓存键生成、一致性、权限隔离

测试用例表（开发计划）:
  | 测试用例       | 输入                 | 预期                     |
  | 缓存命中       | 相同 cache_key       | 返回缓存计划             |
  | 缓存未命中     | 不同 cache_key       | 返回 None                |
  | 缓存失效       | epoch 变更           | 旧计划被清除             |
  | 缓存键生成     | 相同问题不同权限     | 不同 key                 |
  | 缓存键一致性   | 相同IR+context       | 相同 key                 |
  | 清空缓存       | clear()              | 缓存为空                 |
  | 缓存统计       | put+get              | 命中数和未命中数正确     |
  | TTL过期        | 内存模式模拟         | 过期后返回None           |

模式参考: agent_platform/tests/unit/test_query_compiler.py
  - 使用 pytest
  - 定义测试数据工厂函数
"""

import time

import pytest

from agent_platform.query_compiler import (
    IRBuilder,
    LogicalPlanner,
    PhysicalPlanner,
)
from agent_platform.query_compiler.plan_cache.cache import PlanCache
from agent_platform.query_compiler.plan_cache.cache_key import (
    CacheContext,
    CacheKeyGenerator,
)
from agent_platform.query_compiler.plan_cache import cache as cache_module
from agent_platform.query_compiler.physical_planner.planner import (
    PhysicalPlan,
    PlanStage,
)


# ============================================================
# 测试数据工厂
# ============================================================

def make_query_spec(
    intent: str = "clause_query",
    risk_level: str = "medium",
    entities: list = None,
    constraints: dict = None,
    top_k: int = 10,
    **kwargs,
) -> dict:
    """构造 query_spec 字典

    query_spec 需要有 "intent" 和 "risk_level" 字段。
    其余字段（entities / constraints / top_k 等）为可选项。
    """
    spec = {
        "intent": intent,
        "risk_level": risk_level,
        "entities": entities or [],
        "constraints": constraints or {},
        "top_k": top_k,
    }
    spec.update(kwargs)
    return spec


def make_claims(count: int = 2, slot_type: str = "metric|required") -> list:
    """构造声明槽位列表（dict 格式）

    slot_type 编码 "{template_key}|required" 或 "{template_key}|optional"，
    默认必填（required），用于 PlanValidator 的覆盖校验。
    """
    return [
        {
            "claim_id": f"c{i}",
            "description": f"声明槽位 {i}",
            "slot_type": slot_type,
            "status": "pending",
            "evidence_ids": [],
        }
        for i in range(count)
    ]


def build_ir(intent: str = "clause_query", risk_level: str = "medium"):
    """构建 QueryIR"""
    spec = make_query_spec(intent=intent, risk_level=risk_level)
    return IRBuilder().build(spec, make_claims(2))


def build_physical_plan(
    intent: str = "clause_query", risk_level: str = "medium"
) -> PhysicalPlan:
    """构建物理计划：IR → 逻辑计划 → 物理计划"""
    ir = build_ir(intent=intent, risk_level=risk_level)
    logical_plan = LogicalPlanner().plan(ir)
    return PhysicalPlanner().plan(logical_plan, ir)


def make_context(
    user_permissions: str = "reader",
    index_epoch: str = "kb-2026-07",
    constraints_hash: str = "hash-001",
) -> CacheContext:
    """构造缓存上下文"""
    return CacheContext(
        user_permissions=user_permissions,
        index_epoch=index_epoch,
        constraints_hash=constraints_hash,
    )


def make_key(
    ir=None,
    context: CacheContext = None,
) -> str:
    """生成缓存键（便捷封装）"""
    if ir is None:
        ir = build_ir()
    if context is None:
        context = make_context()
    return CacheKeyGenerator().make_key(ir, context)


# ============================================================
# PlanCache 测试
# ============================================================

class TestPlanCache:
    """物理计划缓存测试（内存模式）"""

    def test_cache_hit(self):
        """缓存命中：相同 cache_key 返回缓存计划"""
        cache = PlanCache()
        plan = build_physical_plan(intent="clause_query")
        key = "test-key-hit"

        cache.put(key, plan, ttl=1800, epoch="kb-v1")
        cached = cache.get(key)

        assert cached is not None
        assert isinstance(cached, PhysicalPlan)
        assert cached.plan_id == plan.plan_id
        assert cached.intent == plan.intent

    def test_cache_miss(self):
        """缓存未命中：不同 cache_key 返回 None"""
        cache = PlanCache()
        plan = build_physical_plan(intent="clause_query")

        cache.put("key-exists", plan, ttl=1800, epoch="kb-v1")
        result = cache.get("key-not-exists")

        assert result is None

    def test_cache_miss_on_empty_cache(self):
        """空缓存获取返回 None"""
        cache = PlanCache()
        assert cache.get("any-key") is None

    def test_cache_invalidate_by_epoch(self):
        """缓存失效：epoch 变更后旧计划被清除"""
        cache = PlanCache()
        plan1 = build_physical_plan(intent="clause_query")
        plan2 = build_physical_plan(intent="threshold")

        cache.put("key-1", plan1, ttl=1800, epoch="kb-v1")
        cache.put("key-2", plan2, ttl=1800, epoch="kb-v2")

        # 失效 kb-v1 的缓存
        count = cache.invalidate_by_epoch("kb-v1")
        assert count == 1

        # key-1 已失效
        assert cache.get("key-1") is None
        # key-2 仍在
        assert cache.get("key-2") is not None

    def test_cache_invalidate_unknown_epoch(self):
        """失效不存在的 epoch 返回 0"""
        cache = PlanCache()
        plan = build_physical_plan()
        cache.put("key-1", plan, ttl=1800, epoch="kb-v1")

        count = cache.invalidate_by_epoch("kb-nonexistent")
        assert count == 0
        # 原缓存仍有效
        assert cache.get("key-1") is not None

    def test_cache_invalidate_multiple_keys_same_epoch(self):
        """同一 epoch 下多个键同时失效"""
        cache = PlanCache()
        plan = build_physical_plan()

        cache.put("key-a", plan, ttl=1800, epoch="kb-v1")
        cache.put("key-b", plan, ttl=1800, epoch="kb-v1")
        cache.put("key-c", plan, ttl=1800, epoch="kb-v1")

        count = cache.invalidate_by_epoch("kb-v1")
        assert count == 3

        assert cache.get("key-a") is None
        assert cache.get("key-b") is None
        assert cache.get("key-c") is None

    def test_clear_cache(self):
        """清空缓存：clear() 后缓存为空"""
        cache = PlanCache()
        plan = build_physical_plan()

        cache.put("key-1", plan, ttl=1800, epoch="kb-v1")
        cache.put("key-2", plan, ttl=1800, epoch="kb-v1")

        cache.clear()

        assert cache.get("key-1") is None
        assert cache.get("key-2") is None
        stats = cache.stats()
        assert stats["size"] == 0

    def test_cache_stats_hits_and_misses(self):
        """缓存统计：put + get 后命中数和未命中数正确"""
        cache = PlanCache()
        plan = build_physical_plan()

        cache.put("key-hit", plan, ttl=1800, epoch="kb-v1")

        # 一次命中
        cache.get("key-hit")
        # 两次未命中
        cache.get("key-miss-1")
        cache.get("key-miss-2")

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    def test_cache_stats_hit_rate(self):
        """缓存统计：命中率计算正确"""
        cache = PlanCache()
        plan = build_physical_plan()
        cache.put("key-1", plan, ttl=1800, epoch="kb-v1")

        cache.get("key-1")  # hit
        cache.get("key-1")  # hit
        cache.get("key-miss")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        # hit_rate = 2 / 3 = 0.6667
        assert stats["hit_rate"] == round(2 / 3, 4)

    def test_cache_stats_empty(self):
        """空缓存统计：hits=0, misses=0, hit_rate=0"""
        cache = PlanCache()
        stats = cache.stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0

    def test_cache_stats_contains_required_fields(self):
        """统计字典包含必要字段"""
        cache = PlanCache()
        stats = cache.stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert "hit_rate" in stats

    def test_cache_put_overwrite(self):
        """相同 key 重复 put 覆盖旧值"""
        cache = PlanCache()
        plan1 = build_physical_plan(intent="clause_query")
        plan2 = build_physical_plan(intent="threshold")

        cache.put("key-1", plan1, ttl=1800, epoch="kb-v1")
        cache.put("key-1", plan2, ttl=1800, epoch="kb-v1")

        cached = cache.get("key-1")
        assert cached is not None
        assert cached.plan_id == plan2.plan_id
        assert cached.intent == "threshold"

    def test_cache_ttl_never_expires(self):
        """TTL <= 0 表示永不过期"""
        cache = PlanCache()
        plan = build_physical_plan()

        cache.put("key-permanent", plan, ttl=0, epoch="kb-v1")
        cached = cache.get("key-permanent")

        assert cached is not None
        assert cached.plan_id == plan.plan_id

    def test_cache_ttl_expiry(self, monkeypatch):
        """TTL 过期：过期后返回 None"""
        base_time = 1_000_000.0
        monkeypatch.setattr(cache_module.time, "time", lambda: base_time)

        cache = PlanCache()
        plan = build_physical_plan()

        # 写入 TTL=100 秒的计划
        cache.put("key-ttl", plan, ttl=100, epoch="kb-v1")

        # 写入时刻应命中
        assert cache.get("key-ttl") is not None

        # 推进时间到 TTL 之后
        monkeypatch.setattr(cache_module.time, "time", lambda: base_time + 200)

        # 过期后应返回 None
        assert cache.get("key-ttl") is None

    def test_cache_ttl_not_yet_expired(self, monkeypatch):
        """TTL 未过期：在 TTL 内仍可命中"""
        base_time = 1_000_000.0
        monkeypatch.setattr(cache_module.time, "time", lambda: base_time)

        cache = PlanCache()
        plan = build_physical_plan()

        cache.put("key-ttl", plan, ttl=100, epoch="kb-v1")

        # 推进时间到 TTL 之内
        monkeypatch.setattr(cache_module.time, "time", lambda: base_time + 50)

        assert cache.get("key-ttl") is not None

    def test_cache_size_reflects_entries(self):
        """缓存大小反映条目数"""
        cache = PlanCache()
        plan = build_physical_plan()

        assert cache.stats()["size"] == 0

        cache.put("key-1", plan, ttl=1800, epoch="kb-v1")
        assert cache.stats()["size"] == 1

        cache.put("key-2", plan, ttl=1800, epoch="kb-v1")
        assert cache.stats()["size"] == 2

    def test_cache_stats_persist_after_clear(self):
        """统计计数器在 clear() 后保留"""
        cache = PlanCache()
        plan = build_physical_plan()

        cache.put("key-1", plan, ttl=1800, epoch="kb-v1")
        cache.get("key-1")  # hit

        cache.clear()

        stats = cache.stats()
        # 计数器保留（clear 只清条目，不清统计）
        assert stats["hits"] == 1
        assert stats["size"] == 0

    def test_cache_roundtrip_preserves_stages(self):
        """缓存往返：取出的计划阶段结构完整"""
        cache = PlanCache()
        plan = build_physical_plan(intent="threshold")

        cache.put("key-rt", plan, ttl=1800, epoch="kb-v1")
        cached = cache.get("key-rt")

        assert cached is not None
        assert len(cached.stages) == len(plan.stages)
        assert cached.stages[0].channels == plan.stages[0].channels
        assert cached.stages[0].top_k == plan.stages[0].top_k
        assert cached.stages[0].rerank == plan.stages[0].rerank


# ============================================================
# CacheKeyGenerator 测试
# ============================================================

class TestCacheKeyGenerator:
    """缓存键生成器测试"""

    def test_make_key_returns_hex_string(self):
        """生成的缓存键为 64 字符 SHA256 十六进制串"""
        ir = build_ir()
        context = make_context()
        key = CacheKeyGenerator().make_key(ir, context)

        assert isinstance(key, str)
        assert len(key) == 64
        # 十六进制字符
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_ir_and_context_same_key(self):
        """缓存键一致性：相同 IR + context → 相同 key"""
        ir = build_ir(intent="clause_query")
        context = make_context()

        gen = CacheKeyGenerator()
        key1 = gen.make_key(ir, context)
        key2 = gen.make_key(ir, context)

        assert key1 == key2

    def test_different_permissions_different_key(self):
        """缓存键生成：相同问题不同权限 → 不同 key"""
        ir = build_ir(intent="clause_query")
        gen = CacheKeyGenerator()

        key_reader = gen.make_key(ir, make_context(user_permissions="reader"))
        key_admin = gen.make_key(ir, make_context(user_permissions="admin"))

        assert key_reader != key_admin

    def test_different_epoch_different_key(self):
        """不同知识库版本 → 不同 key"""
        ir = build_ir(intent="clause_query")
        gen = CacheKeyGenerator()

        key_v1 = gen.make_key(ir, make_context(index_epoch="kb-v1"))
        key_v2 = gen.make_key(ir, make_context(index_epoch="kb-v2"))

        assert key_v1 != key_v2

    def test_different_constraints_hash_different_key(self):
        """不同约束哈希 → 不同 key"""
        ir = build_ir(intent="clause_query")
        gen = CacheKeyGenerator()

        key_h1 = gen.make_key(ir, make_context(constraints_hash="hash-001"))
        key_h2 = gen.make_key(ir, make_context(constraints_hash="hash-002"))

        assert key_h1 != key_h2

    def test_different_intent_different_key(self):
        """不同意图 → 不同 key"""
        gen = CacheKeyGenerator()
        context = make_context()

        ir_clause = build_ir(intent="clause_query")
        ir_threshold = build_ir(intent="threshold")

        key_clause = gen.make_key(ir_clause, context)
        key_threshold = gen.make_key(ir_threshold, context)

        assert key_clause != key_threshold

    def test_different_risk_level_different_key(self):
        """不同风险级别 → 不同 key"""
        gen = CacheKeyGenerator()
        context = make_context()

        ir_medium = build_ir(intent="clause_query", risk_level="medium")
        ir_high = build_ir(intent="clause_query", risk_level="high")

        key_medium = gen.make_key(ir_medium, context)
        key_high = gen.make_key(ir_high, context)

        assert key_medium != key_high

    def test_different_claims_different_key(self):
        """不同声明槽位 → 不同 key"""
        gen = CacheKeyGenerator()
        context = make_context()

        spec1 = make_query_spec(intent="clause_query")
        spec2 = make_query_spec(intent="clause_query")
        ir1 = IRBuilder().build(spec1, make_claims(2))
        ir2 = IRBuilder().build(spec2, make_claims(3))

        key1 = gen.make_key(ir1, context)
        key2 = gen.make_key(ir2, context)

        assert key1 != key2

    def test_key_stable_across_generator_instances(self):
        """不同 CacheKeyGenerator 实例生成相同 key"""
        ir = build_ir(intent="clause_query")
        context = make_context()

        key1 = CacheKeyGenerator().make_key(ir, context)
        key2 = CacheKeyGenerator().make_key(ir, context)

        assert key1 == key2

    def test_default_context_produces_valid_key(self):
        """默认 CacheContext（空字符串）仍生成有效 key"""
        ir = build_ir()
        key = CacheKeyGenerator().make_key(ir, CacheContext())

        assert isinstance(key, str)
        assert len(key) == 64


# ============================================================
# 缓存键 + 缓存集成测试
# ============================================================

class TestCacheKeyIntegration:
    """缓存键与 PlanCache 集成测试"""

    def test_same_key_cache_hit(self):
        """相同 IR + context 生成相同 key → 缓存命中"""
        ir = build_ir(intent="clause_query")
        context = make_context()

        gen = CacheKeyGenerator()
        key = gen.make_key(ir, context)

        cache = PlanCache()
        plan = build_physical_plan(intent="clause_query")
        cache.put(key, plan, ttl=1800, epoch=context.index_epoch)

        # 用相同 IR + context 再生成 key → 命中
        key_again = gen.make_key(ir, context)
        cached = cache.get(key_again)

        assert cached is not None
        assert cached.plan_id == plan.plan_id

    def test_different_key_cache_miss(self):
        """不同权限生成不同 key → 缓存未命中"""
        ir = build_ir(intent="clause_query")
        gen = CacheKeyGenerator()

        key_reader = gen.make_key(ir, make_context(user_permissions="reader"))
        key_admin = gen.make_key(ir, make_context(user_permissions="admin"))

        cache = PlanCache()
        plan = build_physical_plan(intent="clause_query")
        cache.put(key_reader, plan, ttl=1800, epoch="kb-v1")

        # 用 admin 权限的 key 查询 → 未命中
        assert cache.get(key_admin) is None

    def test_epoch_invalidation_with_generated_key(self):
        """通过生成的 key 缓存后，epoch 失效生效"""
        ir = build_ir(intent="clause_query")
        context = make_context(index_epoch="kb-v1")

        key = CacheKeyGenerator().make_key(ir, context)

        cache = PlanCache()
        plan = build_physical_plan(intent="clause_query")
        cache.put(key, plan, ttl=1800, epoch="kb-v1")

        # 命中
        assert cache.get(key) is not None

        # 失效旧版本
        cache.invalidate_by_epoch("kb-v1")

        # 未命中
        assert cache.get(key) is None

    def test_different_queries_cached_separately(self):
        """不同意图的查询各自独立缓存"""
        gen = CacheKeyGenerator()
        context = make_context()

        ir_clause = build_ir(intent="clause_query")
        ir_threshold = build_ir(intent="threshold")

        key_clause = gen.make_key(ir_clause, context)
        key_threshold = gen.make_key(ir_threshold, context)

        cache = PlanCache()
        plan_clause = build_physical_plan(intent="clause_query")
        plan_threshold = build_physical_plan(intent="threshold")

        cache.put(key_clause, plan_clause, ttl=1800, epoch="kb-v1")
        cache.put(key_threshold, plan_threshold, ttl=1800, epoch="kb-v1")

        cached_clause = cache.get(key_clause)
        cached_threshold = cache.get(key_threshold)

        assert cached_clause is not None
        assert cached_clause.intent == "clause_query"
        assert cached_threshold is not None
        assert cached_threshold.intent == "threshold"
