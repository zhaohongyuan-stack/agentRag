"""
网关模块 — 请求入口、会话管理、限流、幂等、鉴权

Phase 1: 请求处理、会话管理
Phase 5 (M5.4): 限流、幂等、鉴权
"""

from .auth.auth_handler import AuthHandler, AuthResult
from .auth.idempotency import IdempotencyHandler, IdempotencyResult
from .rate_limit.rate_limiter import RATE_LIMITS, RateLimitResult, RateLimiter
from .request_handler import (
    QueryRequest,
    QueryResponse,
    RequestHandler,
    RetrievalClient,
)
from .session_handler import SessionManager, SessionState

__all__ = [
    # Phase 1
    "RequestHandler",
    "QueryRequest",
    "QueryResponse",
    "RetrievalClient",
    "SessionManager",
    "SessionState",
    # Phase 5 - M5.4
    "RateLimiter",
    "RateLimitResult",
    "RATE_LIMITS",
    "AuthHandler",
    "AuthResult",
    "IdempotencyHandler",
    "IdempotencyResult",
]
