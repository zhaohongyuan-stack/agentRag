"""
鉴权处理器 — M5.4 网关模块

验证 API Token / JWT 的有效性，返回调用者身份和角色。

鉴权方式:
  - Bearer Token: 简单令牌验证（开发环境）
  - JWT: 生产环境使用 JWT（需 PyJWT）

角色映射:
  - 有效 token → authenticated / premium
  - 无 token / 无效 token → anonymous
"""

import hashlib
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 模拟 token 存储（生产环境替换为数据库/Redis）
_TOKEN_STORE = {
    # token_hash → {role, expires_at}
}


class AuthResult:
    """鉴权结果"""

    def __init__(
        self,
        authenticated: bool,
        caller_id: str = "",
        role: str = "anonymous",
        reason: str = "",
    ):
        self.authenticated = authenticated
        self.caller_id = caller_id
        self.role = role
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "authenticated": self.authenticated,
            "caller_id": self.caller_id,
            "role": self.role,
            "reason": self.reason,
        }


class AuthHandler:
    """
    鉴权处理器

    用法:
        handler = AuthHandler()
        result = handler.authenticate("Bearer abc123")
        if not result.authenticated:
            return 401  # Unauthorized
    """

    def __init__(self, token_store: Optional[dict] = None):
        """
        Args:
            token_store: 令牌存储字典 {token_hash: {role, caller_id, expires_at}}
                         None 时使用全局 _TOKEN_STORE
        """
        self._store = token_store if token_store is not None else _TOKEN_STORE

    def register_token(
        self,
        token: str,
        caller_id: str,
        role: str = "authenticated",
        expires_in: int = 3600,
    ) -> None:
        """
        注册令牌

        Args:
            token: 原始令牌
            caller_id: 调用者 ID
            role: 角色（authenticated / premium / admin）
            expires_in: 有效期（秒）
        """
        token_hash = self._hash_token(token)
        self._store[token_hash] = {
            "caller_id": caller_id,
            "role": role,
            "expires_at": time.time() + expires_in,
        }
        logger.info("注册令牌: caller_id=%s, role=%s", caller_id, role)

    def authenticate(self, auth_header: str) -> AuthResult:
        """
        鉴权

        Args:
            auth_header: Authorization 头部值（如 "Bearer abc123"）

        Returns:
            AuthResult
        """
        if not auth_header:
            return AuthResult(
                authenticated=False,
                reason="缺少认证信息",
            )

        # 解析 Bearer Token
        token = self._extract_token(auth_header)
        if token is None:
            return AuthResult(
                authenticated=False,
                reason="认证格式错误，需 'Bearer <token>'",
            )

        # 查找令牌
        token_hash = self._hash_token(token)
        token_info = self._store.get(token_hash)

        if token_info is None:
            return AuthResult(
                authenticated=False,
                reason="无效的令牌",
            )

        # 检查过期
        if time.time() > token_info.get("expires_at", 0):
            return AuthResult(
                authenticated=False,
                reason="令牌已过期",
            )

        return AuthResult(
            authenticated=True,
            caller_id=token_info["caller_id"],
            role=token_info["role"],
        )

    def revoke_token(self, token: str) -> bool:
        """撤销令牌"""
        token_hash = self._hash_token(token)
        if token_hash in self._store:
            del self._store[token_hash]
            logger.info("撤销令牌")
            return True
        return False

    @staticmethod
    def _extract_token(auth_header: str) -> Optional[str]:
        """从 Authorization 头部提取令牌"""
        if not auth_header:
            return None
        parts = auth_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return None

    @staticmethod
    def _hash_token(token: str) -> str:
        """令牌哈希（安全存储）"""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
