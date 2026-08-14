"""
用户存储模块 — ACE-RAG Agent 平台

基于 SQLite 的固定账号用户存储。

特点:
  - 6 个固定账号，初始化时写入数据库（IF NOT EXISTS）
  - 管理员: ZHY, MQ（密码 123456）
  - 普通用户: ML, CSW, XJY, LRF（密码 123456）
  - 密码使用 hashlib.sha256 哈希（不依赖 bcrypt）
  - 数据库路径: /app/data/auth.db（容器内路径）

表结构:
  users(
      id            TEXT PRIMARY KEY,
      username      TEXT UNIQUE,
      password_hash TEXT,
      role          TEXT,
      display_name  TEXT,
      created_at    TIMESTAMP
  )
"""

import hashlib
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
# 数据库路径：优先使用环境变量，默认容器内路径 /app/data/auth.db
# 本地开发时若 /app/data 不可写，回退到本地 ./data 目录
_DB_PATH = os.environ.get("AUTH_DB_PATH", "/app/data/auth.db")


def _resolve_db_path() -> str:
    """解析数据库路径，确保父目录存在"""
    db_path = _DB_PATH
    db_dir = os.path.dirname(db_path)
    try:
        os.makedirs(db_dir, exist_ok=True)
    except (OSError, PermissionError):
        # 容器路径不可用时回退到本地 data 目录
        local_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(local_dir, exist_ok=True)
        db_path = os.path.join(local_dir, "auth.db")
        logger.warning("[Auth] 容器路径不可写，回退到本地: %s", db_path)
    return db_path


DB_PATH = _resolve_db_path()

# ============================================================
# 固定账号列表
# ============================================================
_FIXED_USERS: List[Dict[str, str]] = [
    {"username": "ZHY", "password": "123456", "role": "admin",   "display_name": "ZHY（管理员）"},
    {"username": "MQ",  "password": "123456", "role": "admin",   "display_name": "MQ（管理员）"},
    {"username": "ML",  "password": "123456", "role": "user",    "display_name": "ML"},
    {"username": "CSW", "password": "123456", "role": "user",    "display_name": "CSW"},
    {"username": "XJY", "password": "123456", "role": "user",    "display_name": "XJY"},
    {"username": "LRF", "password": "123456", "role": "user",    "display_name": "LRF"},
]


# ============================================================
# 密码哈希（hashlib.sha256）
# ============================================================
def hash_password(password: str) -> str:
    """使用 sha256 对密码进行哈希"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """验证密码是否与哈希匹配"""
    return hash_password(password) == password_hash


# ============================================================
# 数据库连接
# ============================================================
def _get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接（启用外键，行结果按字典返回）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ============================================================
# 初始化
# ============================================================
def init_db() -> None:
    """
    初始化数据库:
      1. 创建 users 表（IF NOT EXISTS）
      2. 写入 6 个固定账号（已存在则跳过）
    """
    print("[Auth] 初始化用户数据库...", DB_PATH)
    logger.info("[Auth] 初始化用户数据库: %s", DB_PATH)

    with _get_conn() as conn:
        # 1. 建表
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL,
                display_name  TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # 2. 写入固定账号
        for user in _FIXED_USERS:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO users (id, username, password_hash, role, display_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        user["username"],
                        hash_password(user["password"]),
                        user["role"],
                        user["display_name"],
                        datetime.utcnow().isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                # 用户已存在，跳过
                pass
        conn.commit()

    # 统计初始化结果
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    print(f"[Auth] 用户数据库初始化完成，共 {count} 个账号")
    logger.info("[Auth] 用户数据库初始化完成，共 %d 个账号", count)


# ============================================================
# 用户验证
# ============================================================
def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    验证用户名密码

    Args:
        username: 用户名
        password: 明文密码

    Returns:
        验证成功返回用户信息字典，失败返回 None
        字典字段: id, username, role, display_name, created_at
    """
    if not username or not password:
        return None

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, display_name, created_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        logger.info("[Auth] 用户不存在: %s", username)
        return None

    if not verify_password(password, row["password_hash"]):
        logger.info("[Auth] 密码错误: %s", username)
        return None

    logger.info("[Auth] 用户验证成功: %s (role=%s)", username, row["role"])
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """根据 user_id 查询用户信息"""
    if not user_id:
        return None

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, role, display_name, created_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def list_users() -> List[Dict[str, Any]]:
    """列出所有用户（不含密码哈希）"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, role, display_name, created_at "
            "FROM users ORDER BY created_at ASC"
        ).fetchall()

    return [
        {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "display_name": row["display_name"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


# ============================================================
# 模块加载时自动初始化
# ============================================================
try:
    init_db()
except Exception as e:  # pragma: no cover
    logger.error("[Auth] 用户数据库初始化失败: %s", e, exc_info=True)
    print(f"[Auth] 用户数据库初始化失败: {e}")
