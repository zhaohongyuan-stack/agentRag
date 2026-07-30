"""
网关模块单元测试 — M5.4 限流+幂等+鉴权

测试用例覆盖:
  - 限流通过（10次/分钟内通过）
  - 限流拒绝（第11次拒绝）
  - 幂等命中（相同 key → 返回缓存）
  - 幂等新请求（不同 key → 正常处理）
  - 幂等过期（TTL 过期 → 允许重试）
  - 鉴权通过（有效 token）
  - 鉴权失败（无效 token → 401）
  - 鉴权过期（token 过期）
"""

import time

import pytest

from agent_platform.gateway import (
    AuthHandler,
    IdempotencyHandler,
    RATE_LIMITS,
    RateLimiter,
)


# ============================================================
# 限流测试
# ============================================================


class TestRateLimiter:
    """滑动窗口限流测试"""

    @pytest.fixture
    def limiter(self):
        # 使用小窗口便于测试
        limits = {
            "anonymous": {"requests": 10, "window_seconds": 60},
            "authenticated": {"requests": 100, "window_seconds": 60},
            "premium": {"requests": 500, "window_seconds": 60},
        }
        return RateLimiter(limits=limits)

    def test_within_limit_passes(self, limiter):
        """10次/分钟内 → 全部通过"""
        for i in range(10):
            result = limiter.check("user-1", "anonymous")
            assert result.allowed is True, f"第 {i+1} 次请求应通过"

    def test_exceed_limit_rejected(self, limiter):
        """第11次 → 拒绝"""
        for _ in range(10):
            limiter.check("user-1", "anonymous")

        result = limiter.check("user-1", "anonymous")
        assert result.allowed is False
        assert "超限" in result.reason

    def test_different_users_independent(self, limiter):
        """不同用户限流独立"""
        for _ in range(10):
            limiter.check("user-1", "anonymous")

        # user-2 仍有配额
        result = limiter.check("user-2", "anonymous")
        assert result.allowed is True

    def test_premium_higher_limit(self, limiter):
        """premium 用户有更高限额"""
        for _ in range(20):
            result = limiter.check("premium-user", "premium")
            assert result.allowed is True

    def test_remaining_decreases(self, limiter):
        """剩余配额递减"""
        result1 = limiter.check("user-1", "anonymous")
        result2 = limiter.check("user-1", "anonymous")

        assert result1.remaining > result2.remaining

    def test_reset(self, limiter):
        """重置后恢复配额"""
        for _ in range(10):
            limiter.check("user-1", "anonymous")

        limiter.reset("user-1", "anonymous")

        result = limiter.check("user-1", "anonymous")
        assert result.allowed is True


# ============================================================
# 幂等测试
# ============================================================


class TestIdempotency:
    """幂等键管理测试"""

    @pytest.fixture
    def handler(self):
        return IdempotencyHandler(ttl_seconds=600)

    def test_new_request(self, handler):
        """新幂等键 → 正常处理"""
        result = handler.check_or_cache("key-1")

        assert result.is_new is True

    def test_cached_response(self, handler):
        """相同 key → 返回缓存"""
        # 第一次请求
        handler.check_or_cache("key-1")
        handler.cache_response("key-1", {"answer": "结果A"})

        # 第二次相同 key → 缓存命中
        result = handler.check_or_cache("key-1")

        assert result.is_cached is True
        assert result.cached_response == {"answer": "结果A"}

    def test_different_keys_independent(self, handler):
        """不同 key → 独立处理"""
        handler.check_or_cache("key-1")
        handler.cache_response("key-1", {"answer": "A"})

        result = handler.check_or_cache("key-2")
        assert result.is_new is True

    def test_processing_state(self, handler):
        """处理中状态 → 返回 cached（无响应）"""
        # 第一次请求 → 设置为 processing
        handler.check_or_cache("key-1")

        # 第二次请求相同 key → 仍在处理中
        result = handler.check_or_cache("key-1")

        assert result.is_cached is True
        assert result.cached_response is None

    def test_expired_allows_retry(self, handler):
        """TTL 过期 → 允许重试"""
        # 使用短 TTL
        short_handler = IdempotencyHandler(ttl_seconds=1)

        short_handler.check_or_cache("key-1")
        short_handler.cache_response("key-1", {"answer": "A"})

        # 手动模拟过期
        key = "ace-rag:idempotent:key-1"
        short_handler._client._expiry[key] = time.time() - 1  # 已过期

        # 过期后允许新请求
        result = short_handler.check_or_cache("key-1")
        assert result.is_new is True

    def test_empty_key(self, handler):
        """空幂等键 → 正常处理"""
        result = handler.check_or_cache("")
        assert result.is_new is True

    def test_delete_key(self, handler):
        """删除幂等键后可重新请求"""
        handler.check_or_cache("key-1")
        handler.cache_response("key-1", {"answer": "A"})

        handler.delete("key-1")

        result = handler.check_or_cache("key-1")
        assert result.is_new is True


# ============================================================
# 鉴权测试
# ============================================================


class TestAuthHandler:
    """鉴权处理测试"""

    @pytest.fixture
    def handler(self):
        h = AuthHandler()
        h.register_token("valid-token-123", "user-1", "authenticated")
        h.register_token("premium-token-456", "user-2", "premium")
        return h

    def test_valid_token(self, handler):
        """有效 token → 鉴权通过"""
        result = handler.authenticate("Bearer valid-token-123")

        assert result.authenticated is True
        assert result.caller_id == "user-1"
        assert result.role == "authenticated"

    def test_invalid_token(self, handler):
        """无效 token → 鉴权失败"""
        result = handler.authenticate("Bearer invalid-token")

        assert result.authenticated is False
        assert "无效" in result.reason

    def test_no_auth_header(self, handler):
        """无认证头 → 失败"""
        result = handler.authenticate("")

        assert result.authenticated is False
        assert "缺少" in result.reason

    def test_wrong_format(self, handler):
        """格式错误 → 失败"""
        result = handler.authenticate("valid-token-123")

        assert result.authenticated is False
        assert "格式错误" in result.reason

    def test_expired_token(self, handler):
        """过期 token → 失败"""
        # 注册一个短有效期 token
        handler.register_token(
            "short-token", "user-3", "authenticated", expires_in=0
        )
        time.sleep(0.1)

        result = handler.authenticate("Bearer short-token")

        assert result.authenticated is False
        assert "过期" in result.reason

    def test_premium_role(self, handler):
        """premium 角色"""
        result = handler.authenticate("Bearer premium-token-456")

        assert result.authenticated is True
        assert result.role == "premium"

    def test_revoke_token(self, handler):
        """撤销 token 后失效"""
        handler.revoke_token("valid-token-123")

        result = handler.authenticate("Bearer valid-token-123")
        assert result.authenticated is False
