"""
JWT 认证模块 — ACE-RAG Agent 平台

简化版 JWT 实现（HMAC-SHA256），不依赖 PyJWT 第三方库。

功能:
  - 生成 JWT token（有效期 24 小时）
  - 验证 JWT token 的签名与有效期
  - 从 token 中提取 user_id 和 role

token 载荷格式:
  {
      "user_id":   用户唯一 ID,
      "username":  用户名,
      "role":      角色（admin / user）,
      "exp":       过期时间戳（秒）,
      "iat":       签发时间戳（秒）
  }
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
JWT_SECRET = "ace-rag-secret-key-2026"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24  # token 有效期（小时）

# JWT 头部（固定）
_HEADER = {"alg": JWT_ALGORITHM, "typ": "JWT"}


# ============================================================
# 内部工具：Base64URL 编解码
# ============================================================
def _b64url_encode(data: bytes) -> str:
    """Base64URL 编码（无 padding）"""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    """Base64URL 解码（自动补齐 padding）"""
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data = data + ("=" * padding)
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def _sign(message: str) -> str:
    """使用 HMAC-SHA256 对 message 签名，返回 Base64URL 编码的签名"""
    signature = hmac.new(
        JWT_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(signature)


# ============================================================
# 核心 API
# ============================================================
def create_access_token(
    user_id: str,
    username: str,
    role: str,
    expire_hours: int = JWT_EXPIRE_HOURS,
) -> str:
    """
    生成 JWT token

    Args:
        user_id:    用户唯一 ID
        username:   用户名
        role:       角色（admin / user）
        expire_hours: 有效期（小时），默认 24

    Returns:
        JWT token 字符串（header.payload.signature）
    """
    now = int(time.time())
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + expire_hours * 3600,
    }

    header_segment = _b64url_encode(
        json.dumps(_HEADER, separators=(",", ":")).encode("utf-8")
    )
    payload_segment = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    signing_input = f"{header_segment}.{payload_segment}"
    signature_segment = _sign(signing_input)

    token = f"{signing_input}.{signature_segment}"
    logger.debug("[JWT] 生成 token: user_id=%s, role=%s", user_id, role)
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    验证 JWT token 的签名与有效期

    Args:
        token: JWT token 字符串

    Returns:
        验证成功返回 payload 字典，失败返回 None
    """
    if not token or not isinstance(token, str):
        return None

    parts = token.split(".")
    if len(parts) != 3:
        logger.warning("[JWT] token 格式错误: 段数 != 3")
        return None

    header_segment, payload_segment, signature_segment = parts

    # 1. 验证签名
    signing_input = f"{header_segment}.{payload_segment}"
    expected_signature = _sign(signing_input)
    if not hmac.compare_digest(expected_signature, signature_segment):
        logger.warning("[JWT] 签名验证失败")
        return None

    # 2. 解析 payload
    try:
        payload = json.loads(_b64url_decode(payload_segment).decode("utf-8"))
    except Exception as e:
        logger.warning("[JWT] payload 解析失败: %s", e)
        return None

    # 3. 验证有效期
    exp = payload.get("exp")
    if exp is None:
        logger.warning("[JWT] payload 缺少 exp 字段")
        return None

    if int(time.time()) > int(exp):
        logger.warning("[JWT] token 已过期: exp=%s", exp)
        return None

    return payload


def extract_user_id(token: str) -> Optional[str]:
    """从 token 中提取 user_id"""
    payload = verify_token(token)
    if payload is None:
        return None
    return payload.get("user_id")


def extract_role(token: str) -> Optional[str]:
    """从 token 中提取 role"""
    payload = verify_token(token)
    if payload is None:
        return None
    return payload.get("role")


def extract_payload(token: str) -> Optional[Dict[str, Any]]:
    """从 token 中提取完整 payload（验证签名与有效期）"""
    return verify_token(token)
