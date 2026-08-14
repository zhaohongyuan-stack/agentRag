"""
执行追踪存储 — Phase 6

将 TraceCollector.finalize() 的 trace 字典持久化到 SQLite，支持按会话查询。

表结构:
    execution_traces (
        id          TEXT PRIMARY KEY,    -- trace ID（UUID）
        session_id  TEXT NOT NULL,        -- 会话 ID
        trace_json  TEXT NOT NULL,        -- trace 完整 JSON
        created_at  TEXT NOT NULL         -- 创建时间（UTC ISO）
    )

索引:
    idx_traces_session  ON execution_traces(session_id)
    idx_traces_created  ON execution_traces(created_at DESC)
"""

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据库路径解析（参考 conversation_store 模式）
# ============================================================
_DB_PATH = os.environ.get("TRACE_DB_PATH", "/app/data/traces.db")


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
        db_path = os.path.join(local_dir, "traces.db")
        logger.warning("[Trace] 容器路径不可写，回退到本地: %s", db_path)
    return db_path


DB_PATH = _resolve_db_path()


# ============================================================
# 数据库连接
# ============================================================
def _get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接（启用外键，行结果按字典返回）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _now_iso() -> str:
    """当前 UTC 时间 ISO 字符串"""
    return datetime.utcnow().isoformat()


def _json_dumps(value: Any) -> str:
    """JSON 序列化（ensure_ascii=False）"""
    return json.dumps(value, ensure_ascii=False, default=str)


def init_db() -> None:
    """初始化数据库表（幂等）"""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_traces (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                trace_json  TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_session ON execution_traces(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_created ON execution_traces(created_at DESC)"
        )
        conn.commit()
    logger.info("[Trace] 执行追踪数据库初始化: %s", DB_PATH)


# ============================================================
# TraceStore
# ============================================================
class TraceStore:
    """
    执行追踪的 SQLite 存储与查询

    线程安全：每个方法独立连接，SQLite 默认串行化保证写安全。
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or DB_PATH
        self._init_lock = threading.Lock()
        self._initialized = False

    def _ensure_init(self) -> None:
        """惰性初始化表结构（首次写入时）"""
        if self._initialized:
            return
        with self._init_lock:
            if not self._initialized:
                self._init_table()
                self._initialized = True

    def _init_table(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_traces (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    trace_json  TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_session ON execution_traces(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_created ON execution_traces(created_at DESC)"
            )
            conn.commit()
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────
    # 写入
    # ──────────────────────────────────────────────────────

    def save(self, trace: Dict[str, Any]) -> str:
        """
        存储一条 trace

        Args:
            trace: TraceCollector.finalize() 返回的字典

        Returns:
            trace_id（UUID）
        """
        self._ensure_init()
        trace_id = str(uuid.uuid4())
        session_id = trace.get("session_id", "")
        trace_json = _json_dumps(trace)
        created_at = _now_iso()

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO execution_traces (id, session_id, trace_json, created_at) VALUES (?, ?, ?, ?)",
                (trace_id, session_id, trace_json, created_at),
            )
            conn.commit()
        finally:
            conn.close()
        logger.debug("[Trace] 已存储 trace: id=%s, session=%s", trace_id, session_id)
        return trace_id

    # ──────────────────────────────────────────────────────
    # 查询
    # ──────────────────────────────────────────────────────

    def get_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """按 trace ID 查询单条记录"""
        self._ensure_init()
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM execution_traces WHERE id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)
        finally:
            conn.close()

    def get_by_session(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        按会话 ID 查询所有 trace（按创建时间倒序）

        Args:
            session_id: 会话 ID
            limit: 最大返回数

        Returns:
            trace 字典列表（trace_json 已反序列化）
        """
        self._ensure_init()
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM execution_traces WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def list_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出最近的 trace（按创建时间倒序）

        Args:
            limit: 最大返回数

        Returns:
            trace 字典列表（trace_json 已反序列化）
        """
        self._ensure_init()
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM execution_traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def count(self) -> int:
        """返回 trace 总数"""
        self._ensure_init()
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM execution_traces").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def delete_by_session(self, session_id: str) -> int:
        """
        删除指定会话的所有 trace

        Returns:
            删除的记录数
        """
        self._ensure_init()
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute(
                "DELETE FROM execution_traces WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转为字典（trace_json 反序列化）"""
        result = dict(row)
        try:
            result["trace"] = json.loads(result.pop("trace_json"))
        except (json.JSONDecodeError, KeyError):
            result["trace"] = {}
        return result


# ============================================================
# 全局单例
# ============================================================
_singleton: Optional[TraceStore] = None
_singleton_lock = threading.Lock()


def get_trace_store() -> TraceStore:
    """获取 TraceStore 全局单例"""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TraceStore()
    return _singleton
