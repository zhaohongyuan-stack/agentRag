"""鉴权与幂等模块"""
from .auth_handler import AuthHandler, AuthResult
from .idempotency import IdempotencyHandler, IdempotencyResult
from .jwt_handler import (
    create_access_token,
    extract_payload,
    extract_role,
    extract_user_id,
    verify_token,
)
from .user_store import (
    get_user_by_id,
    hash_password,
    init_db as init_user_db,
    list_users,
    verify_password,
    verify_user,
)

__all__ = [
    "AuthHandler",
    "AuthResult",
    "IdempotencyHandler",
    "IdempotencyResult",
    "create_access_token",
    "verify_token",
    "extract_user_id",
    "extract_role",
    "extract_payload",
    "verify_user",
    "get_user_by_id",
    "list_users",
    "hash_password",
    "verify_password",
    "init_user_db",
]
