"""
对话持久化模块 — ACE-RAG Agent 平台

基于 SQLite 的对话存储，支持:
  - 对话记录的增删查改
  - 用户审查（correct / incorrect / needs_improvement）
  - 改进建议
  - 未回答标记
  - 管理员质检标注
  - 管理员维度的统计与查询

数据库路径: /app/data/conversations.db（容器内路径）

表结构:
  conversations            - 对话主表
  conversation_reviews     - 用户审查记录
  improvement_suggestions  - 改进建议
  unanswered_marks         - 未回答标记
  admin_qa_marks           - 管理员质检标注

所有 TEXT 字段中的 JSON 数据使用 json.dumps(ensure_ascii=False) 序列化。
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_user_map() -> Dict[str, Dict[str, str]]:
    """从 auth.db 构建 user_id → {username, display_name, role} 映射"""
    try:
        from .auth.user_store import list_users
        users = list_users()
        return {u["id"]: u for u in users}
    except Exception as e:
        logger.warning("[Conversation] 加载用户映射失败: %s", e)
        return {}

# ============================================================
# 配置
# ============================================================
_DB_PATH = os.environ.get("CONVERSATION_DB_PATH", "/app/data/conversations.db")


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
        db_path = os.path.join(local_dir, "conversations.db")
        logger.warning("[Conversation] 容器路径不可写，回退到本地: %s", db_path)
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
    """JSON 序列化（ensure_ascii=False），None/空值返回空字符串"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    """JSON 反序列化，空值返回 default"""
    if not value:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# ============================================================
# 初始化
# ============================================================
def init_db() -> None:
    """
    初始化所有表（IF NOT EXISTS）
    """
    print("[Conversation] 初始化对话数据库...", DB_PATH)
    logger.info("[Conversation] 初始化对话数据库: %s", DB_PATH)

    with _get_conn() as conn:
        # 1. 对话主表
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id                   TEXT PRIMARY KEY,
                user_id              TEXT NOT NULL,
                session_id           TEXT,
                query                TEXT,
                answer               TEXT,
                full_response        TEXT,
                intent               TEXT,
                complexity           TEXT,
                is_refusal           BOOLEAN DEFAULT 0,
                refusal_reason       TEXT,
                confidence           REAL DEFAULT 0.0,
                state_trace          TEXT,
                evidence_count       INT DEFAULT 0,
                sufficiency_score    REAL DEFAULT 0.0,
                latency_ms           REAL DEFAULT 0.0,
                citations            TEXT,
                claims_with_evidence TEXT,
                ambiguities          TEXT,
                created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_intent ON conversations(intent)"
        )

        # 2. 用户审查记录
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_reviews (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                review_status   TEXT CHECK(review_status IN ('correct','incorrect','needs_improvement')),
                review_comment  TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_conversation_id ON conversation_reviews(conversation_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON conversation_reviews(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_status ON conversation_reviews(review_status)"
        )

        # 3. 改进建议
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS improvement_suggestions (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                suggestion_text TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestions_conversation_id ON improvement_suggestions(conversation_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestions_user_id ON improvement_suggestions(user_id)"
        )

        # 4. 未回答标记
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unanswered_marks (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                is_marked       BOOLEAN DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unanswered_conversation_id ON unanswered_marks(conversation_id)"
        )

        # 5. 管理员质检标注
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_qa_marks (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                admin_id        TEXT NOT NULL,
                qa_result       TEXT CHECK(qa_result IN ('pass','fail','rework')),
                qa_comment      TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_conversation_id ON admin_qa_marks(conversation_id)"
        )

        conn.commit()

    print("[Conversation] 对话数据库初始化完成")
    logger.info("[Conversation] 对话数据库初始化完成")


# ============================================================
# 行解析
# ============================================================
def _row_to_conversation(row: sqlite3.Row) -> Dict[str, Any]:
    """将 conversations 表行转换为字典（反序列化 JSON 字段）"""
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "session_id": row["session_id"],
        "query": row["query"],
        "answer": row["answer"],
        "full_response": _json_loads(row["full_response"], default={}),
        "intent": row["intent"],
        "complexity": row["complexity"],
        "is_refusal": bool(row["is_refusal"]),
        "refusal_reason": row["refusal_reason"],
        "confidence": row["confidence"],
        "state_trace": _json_loads(row["state_trace"], default=[]),
        "evidence_count": row["evidence_count"],
        "sufficiency_score": row["sufficiency_score"],
        "latency_ms": row["latency_ms"],
        "citations": _json_loads(row["citations"], default=[]),
        "claims_with_evidence": _json_loads(row["claims_with_evidence"], default=[]),
        "ambiguities": _json_loads(row["ambiguities"], default=[]),
        "created_at": row["created_at"],
    }


def _enrich_conversation(conv: Dict[str, Any]) -> Dict[str, Any]:
    """为对话字典附加审查状态、未回答标记、建议数、质检结果等字段"""
    conv_id = conv["id"]
    with _get_conn() as conn:
        # 审查状态
        rev = conn.execute(
            "SELECT review_status, review_comment FROM conversation_reviews "
            "WHERE conversation_id = ? ORDER BY updated_at DESC LIMIT 1",
            (conv_id,),
        ).fetchone()
        conv["review_status"] = rev["review_status"] if rev else None
        conv["review_comment"] = rev["review_comment"] if rev else None

        # 未回答标记
        un = conn.execute(
            "SELECT is_marked FROM unanswered_marks "
            "WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
            (conv_id,),
        ).fetchone()
        conv["is_unanswered"] = bool(un["is_marked"]) if un else False

        # 建议数
        sug_count = conn.execute(
            "SELECT COUNT(*) AS c FROM improvement_suggestions WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()["c"]
        conv["suggestion_count"] = sug_count
        conv["has_suggestion"] = sug_count > 0

        # 质检结果 + 质检评语
        qa = conn.execute(
            "SELECT qa_result, qa_comment FROM admin_qa_marks "
            "WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
            (conv_id,),
        ).fetchone()
        conv["qa_result"] = qa["qa_result"] if qa else None
        conv["qa_comment"] = qa["qa_comment"] if qa else None

    return conv


# ============================================================
# 对话 CRUD
# ============================================================
def save_conversation(
    user_id: str,
    session_id: str,
    query: str,
    response_dict: Dict[str, Any],
) -> str:
    """
    保存对话记录

    Args:
        user_id:      用户 ID
        session_id:   会话 ID
        query:        用户原始问题
        response_dict: QueryResponse 的字典形式

    Returns:
        conversation_id
    """
    conversation_id = str(uuid.uuid4())
    created_at = _now_iso()

    # 提取字段（兼容 response_dict 可能的键名）
    answer = response_dict.get("answer", "")
    intent = response_dict.get("intent", "")
    complexity = response_dict.get("complexity", "")
    is_refusal = 1 if response_dict.get("is_refusal", False) else 0
    refusal_reason = response_dict.get("refusal_reason")
    confidence = float(response_dict.get("confidence", 0.0) or 0.0)
    state_trace = response_dict.get("state_trace", [])
    evidence_count = int(response_dict.get("evidence_count", 0) or 0)
    sufficiency_score = float(response_dict.get("sufficiency_score", 0.0) or 0.0)
    latency_ms = float(response_dict.get("latency_ms", 0.0) or 0.0)
    citations = response_dict.get("citations", [])
    claims_with_evidence = response_dict.get("claims_with_evidence", [])
    ambiguities = response_dict.get("ambiguities", [])

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversations (
                id, user_id, session_id, query, answer, full_response,
                intent, complexity, is_refusal, refusal_reason, confidence,
                state_trace, evidence_count, sufficiency_score, latency_ms,
                citations, claims_with_evidence, ambiguities, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                user_id,
                session_id,
                query,
                answer,
                _json_dumps(response_dict),
                intent,
                complexity,
                is_refusal,
                refusal_reason,
                confidence,
                _json_dumps(state_trace),
                evidence_count,
                sufficiency_score,
                latency_ms,
                _json_dumps(citations),
                _json_dumps(claims_with_evidence),
                _json_dumps(ambiguities),
                created_at,
            ),
        )
        conn.commit()

    logger.info(
        "[Conversation] 保存对话: id=%s, user=%s, intent=%s",
        conversation_id, user_id, intent,
    )
    return conversation_id


def get_conversations(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    分页查询用户对话列表

    Args:
        user_id:    用户 ID
        page:       页码（从 1 开始）
        page_size:  每页条数
        start_date: 起始日期（ISO 字符串，可选）
        end_date:   结束日期（ISO 字符串，可选）

    Returns:
        {items, total, page, page_size, total_pages}
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20
    offset = (page - 1) * page_size

    where_clauses = ["user_id = ?"]
    params: List[Any] = [user_id]
    if start_date:
        where_clauses.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("created_at <= ?")
        params.append(end_date)
    where_sql = " AND ".join(where_clauses)

    with _get_conn() as conn:
        # 总数
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM conversations WHERE {where_sql}", params
        ).fetchone()["c"]

        # 分页数据
        rows = conn.execute(
            f"""
            SELECT * FROM conversations
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ).fetchall()

    user_map = _get_user_map()
    items = []
    for row in rows:
        conv = _enrich_conversation(_row_to_conversation(row))
        uinfo = user_map.get(conv.get("user_id", ""), {})
        conv["username"] = uinfo.get("username", "未知")
        conv["display_name"] = uinfo.get("display_name", "未知")
        items.append(conv)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """查询单条对话详情（包含审查状态、未回答标记、建议数、质检结果）"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()

    if row is None:
        return None
    return _enrich_conversation(_row_to_conversation(row))


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """
    删除对话（仅对话所有者可删除）

    Returns:
        是否删除成功
    """
    with _get_conn() as conn:
        # 校验所有权
        row = conn.execute(
            "SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return False
        if row["user_id"] != user_id:
            return False

        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()

    logger.info("[Conversation] 删除对话: id=%s, user=%s", conversation_id, user_id)
    return True


# ============================================================
# 用户审查
# ============================================================
def save_review(
    conversation_id: str,
    user_id: str,
    review_status: str,
    review_comment: str,
) -> str:
    """
    保存/更新用户审查记录（同一用户对同一对话只保留最新一条）

    Args:
        conversation_id: 对话 ID
        user_id:         用户 ID
        review_status:   审查状态（correct / incorrect / needs_improvement）
        review_comment:  审查评论

    Returns:
        review_id
    """
    if review_status not in ("correct", "incorrect", "needs_improvement"):
        raise ValueError(f"无效的 review_status: {review_status}")

    now = _now_iso()
    review_id = str(uuid.uuid4())

    with _get_conn() as conn:
        # 查找是否已有审查记录
        existing = conn.execute(
            "SELECT id FROM conversation_reviews WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()

        if existing:
            review_id = existing["id"]
            conn.execute(
                """
                UPDATE conversation_reviews
                SET review_status = ?, review_comment = ?, updated_at = ?
                WHERE id = ?
                """,
                (review_status, review_comment, now, review_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO conversation_reviews (id, conversation_id, user_id, review_status, review_comment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (review_id, conversation_id, user_id, review_status, review_comment, now, now),
            )
        conn.commit()

    logger.info(
        "[Conversation] 保存审查: conv=%s, user=%s, status=%s",
        conversation_id, user_id, review_status,
    )
    return review_id


def get_review(conversation_id: str) -> Optional[Dict[str, Any]]:
    """查询对话的审查记录"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversation_reviews WHERE conversation_id = ? ORDER BY updated_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()

    if row is None:
        return None
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "user_id": row["user_id"],
        "review_status": row["review_status"],
        "review_comment": row["review_comment"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ============================================================
# 改进建议
# ============================================================
def add_suggestion(
    conversation_id: str,
    user_id: str,
    suggestion_text: str,
) -> str:
    """添加改进建议"""
    suggestion_id = str(uuid.uuid4())
    now = _now_iso()

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO improvement_suggestions (id, conversation_id, user_id, suggestion_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (suggestion_id, conversation_id, user_id, suggestion_text, now),
        )
        conn.commit()

    logger.info(
        "[Conversation] 添加建议: conv=%s, user=%s, id=%s",
        conversation_id, user_id, suggestion_id,
    )
    return suggestion_id


def get_suggestions(conversation_id: str) -> List[Dict[str, Any]]:
    """查询对话的所有改进建议"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM improvement_suggestions WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "user_id": row["user_id"],
            "suggestion_text": row["suggestion_text"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_suggestion(suggestion_id: str, user_id: str) -> bool:
    """删除改进建议（仅建议所有者可删除）"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT user_id FROM improvement_suggestions WHERE id = ?", (suggestion_id,)
        ).fetchone()
        if row is None:
            return False
        if row["user_id"] != user_id:
            return False

        conn.execute("DELETE FROM improvement_suggestions WHERE id = ?", (suggestion_id,))
        conn.commit()

    return True


# ============================================================
# 未回答标记
# ============================================================
def toggle_unanswered(
    conversation_id: str,
    user_id: str,
    is_marked: bool,
) -> bool:
    """
    标记/取消标记对话为未回答（同一用户对同一对话只保留一条记录）

    Returns:
        是否操作成功
    """
    now = _now_iso()
    marked_int = 1 if is_marked else 0
    mark_id = str(uuid.uuid4())

    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM unanswered_marks WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE unanswered_marks SET is_marked = ?, created_at = ? WHERE id = ?",
                (marked_int, now, existing["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO unanswered_marks (id, conversation_id, user_id, is_marked, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (mark_id, conversation_id, user_id, marked_int, now),
            )
        conn.commit()

    return True


def get_unanswered(conversation_id: str) -> Dict[str, Any]:
    """查询对话的未回答标记状态"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM unanswered_marks WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()

    if row is None:
        return {"is_marked": False, "marked_by": None, "created_at": None}
    return {
        "is_marked": bool(row["is_marked"]),
        "marked_by": row["user_id"],
        "created_at": row["created_at"],
    }


# ============================================================
# 管理员接口
# ============================================================
def admin_get_conversations(
    user_id: Optional[str] = None,
    review_status: Optional[str] = None,
    is_unanswered: Optional[bool] = None,
    has_suggestion: Optional[bool] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    intent: Optional[str] = None,
    is_refusal: Optional[bool] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    管理员查询所有对话（支持多维筛选）

    Args:
        user_id:       筛选特定用户
        review_status: 筛选审查状态
        is_unanswered: 筛选未回答标记
        has_suggestion:筛选是否有改进建议
        start_date:    起始日期
        end_date:      结束日期
        intent:        意图筛选
        is_refusal:    是否拒答
        page:          页码
        page_size:     每页条数

    Returns:
        {items, total, page, page_size, total_pages}
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20
    offset = (page - 1) * page_size

    where_clauses: List[str] = []
    params: List[Any] = []

    if user_id:
        where_clauses.append("c.user_id = ?")
        params.append(user_id)
    if start_date:
        where_clauses.append("c.created_at >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("c.created_at <= ?")
        params.append(end_date)
    if intent:
        where_clauses.append("c.intent = ?")
        params.append(intent)
    if is_refusal is not None:
        where_clauses.append("c.is_refusal = ?")
        params.append(1 if is_refusal else 0)
    if review_status:
        where_clauses.append(
            "EXISTS (SELECT 1 FROM conversation_reviews r WHERE r.conversation_id = c.id AND r.review_status = ?)"
        )
        params.append(review_status)
    if is_unanswered is not None:
        if is_unanswered:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM unanswered_marks u WHERE u.conversation_id = c.id AND u.is_marked = 1)"
            )
        else:
            where_clauses.append(
                "NOT EXISTS (SELECT 1 FROM unanswered_marks u WHERE u.conversation_id = c.id AND u.is_marked = 1)"
            )
    if has_suggestion is not None:
        if has_suggestion:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM improvement_suggestions s WHERE s.conversation_id = c.id)"
            )
        else:
            where_clauses.append(
                "NOT EXISTS (SELECT 1 FROM improvement_suggestions s WHERE s.conversation_id = c.id)"
            )

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    with _get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM conversations c WHERE {where_sql}", params
        ).fetchone()["c"]

        rows = conn.execute(
            f"""
            SELECT c.* FROM conversations c
            WHERE {where_sql}
            ORDER BY c.created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ).fetchall()

    user_map = _get_user_map()
    items = []
    for row in rows:
        conv = _enrich_conversation(_row_to_conversation(row))
        uinfo = user_map.get(conv.get("user_id", ""), {})
        conv["username"] = uinfo.get("username", "未知")
        conv["display_name"] = uinfo.get("display_name", "未知")
        items.append(conv)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def admin_get_users() -> List[Dict[str, Any]]:
    """
    管理员查看用户统计信息

    Returns:
        用户列表，每个用户包含对话数、拒答数、审查数、建议数等统计
    """
    user_map = _get_user_map()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                c.user_id,
                COUNT(c.id) AS conversation_count,
                SUM(CASE WHEN c.is_refusal = 1 THEN 1 ELSE 0 END) AS refusal_count,
                AVG(c.confidence) AS avg_confidence,
                AVG(c.latency_ms) AS avg_latency_ms,
                MIN(c.created_at) AS first_conversation_at,
                MAX(c.created_at) AS last_conversation_at
            FROM conversations c
            GROUP BY c.user_id
            ORDER BY conversation_count DESC
            """
        ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            user_id = row["user_id"]
            uinfo = user_map.get(user_id, {})
            review_count = conn.execute(
                "SELECT COUNT(*) AS c FROM conversation_reviews WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
            suggestion_count = conn.execute(
                "SELECT COUNT(*) AS c FROM improvement_suggestions WHERE user_id = ?",
                (user_id,),
            ).fetchone()["c"]
            unanswered_count = conn.execute(
                "SELECT COUNT(*) AS c FROM unanswered_marks WHERE user_id = ? AND is_marked = 1",
                (user_id,),
            ).fetchone()["c"]

            result.append(
                {
                    "user_id": user_id,
                    "username": uinfo.get("username", "未知"),
                    "display_name": uinfo.get("display_name", "未知"),
                    "role": uinfo.get("role", "user"),
                    "conversation_count": row["conversation_count"],
                    "refusal_count": row["refusal_count"] or 0,
                    "review_count": review_count,
                    "suggestion_count": suggestion_count,
                    "unanswered_count": unanswered_count,
                    "avg_confidence": round(row["avg_confidence"] or 0.0, 4),
                    "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 2),
                    "first_conversation_at": row["first_conversation_at"],
                    "last_conversation_at": row["last_conversation_at"],
                }
            )

    # 加入有账号但还没有对话记录的用户
    for uid, uinfo in user_map.items():
        if not any(r["user_id"] == uid for r in result):
            result.append({
                "user_id": uid,
                "username": uinfo.get("username", "未知"),
                "display_name": uinfo.get("display_name", "未知"),
                "role": uinfo.get("role", "user"),
                "conversation_count": 0,
                "refusal_count": 0,
                "review_count": 0,
                "suggestion_count": 0,
                "unanswered_count": 0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "first_conversation_at": None,
                "last_conversation_at": None,
            })

    return result


def admin_get_reviews(
    review_status: Optional[str] = None,
    user_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    管理员查看审查记录

    Args:
        review_status: 筛选审查状态
        user_id:       筛选特定用户
        page:          页码
        page_size:     每页条数

    Returns:
        {items, total, page, page_size, total_pages}
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20
    offset = (page - 1) * page_size

    where_clauses: List[str] = []
    params: List[Any] = []
    if review_status:
        where_clauses.append("r.review_status = ?")
        params.append(review_status)
    if user_id:
        where_clauses.append("r.user_id = ?")
        params.append(user_id)
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    with _get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM conversation_reviews r WHERE {where_sql}", params
        ).fetchone()["c"]

        rows = conn.execute(
            f"""
            SELECT r.*, c.query AS conversation_query, c.intent AS conversation_intent
            FROM conversation_reviews r
            LEFT JOIN conversations c ON r.conversation_id = c.id
            WHERE {where_sql}
            ORDER BY r.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ).fetchall()

    items = [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "user_id": row["user_id"],
            "review_status": row["review_status"],
            "review_comment": row["review_comment"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "conversation_query": row["conversation_query"],
            "conversation_intent": row["conversation_intent"],
        }
        for row in rows
    ]
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def admin_get_suggestions(
    user_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    管理员查看改进建议

    Args:
        user_id:   筛选特定用户
        page:      页码
        page_size: 每页条数

    Returns:
        {items, total, page, page_size, total_pages}
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20
    offset = (page - 1) * page_size

    where_clauses: List[str] = []
    params: List[Any] = []
    if user_id:
        where_clauses.append("s.user_id = ?")
        params.append(user_id)
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    with _get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM improvement_suggestions s WHERE {where_sql}", params
        ).fetchone()["c"]

        rows = conn.execute(
            f"""
            SELECT s.*, c.query AS conversation_query
            FROM improvement_suggestions s
            LEFT JOIN conversations c ON s.conversation_id = c.id
            WHERE {where_sql}
            ORDER BY s.created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ).fetchall()

    items = [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "user_id": row["user_id"],
            "suggestion_text": row["suggestion_text"],
            "created_at": row["created_at"],
            "conversation_query": row["conversation_query"],
        }
        for row in rows
    ]
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def admin_save_qa(
    conversation_id: str,
    admin_id: str,
    qa_result: str,
    qa_comment: str,
) -> str:
    """
    管理员保存质检标注

    Args:
        conversation_id: 对话 ID
        admin_id:        管理员 ID
        qa_result:       质检结果（pass / fail / rework）
        qa_comment:      质检评论

    Returns:
        qa_id
    """
    if qa_result not in ("pass", "fail", "rework"):
        raise ValueError(f"无效的 qa_result: {qa_result}")

    qa_id = str(uuid.uuid4())
    now = _now_iso()

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO admin_qa_marks (id, conversation_id, admin_id, qa_result, qa_comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (qa_id, conversation_id, admin_id, qa_result, qa_comment, now),
        )
        conn.commit()

    logger.info(
        "[Conversation] 管理员质检: conv=%s, admin=%s, result=%s",
        conversation_id, admin_id, qa_result,
    )
    return qa_id


# ============================================================
# 管理员删除对话（不验证所有权）
# ============================================================
def admin_delete_conversation(conversation_id: str) -> bool:
    """管理员删除任意对话（不需要验证所有权）"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    logger.info("[Conversation] 管理员删除对话: id=%s", conversation_id)
    return True


# ============================================================
# 质检中心统计
# ============================================================
def get_admin_stats() -> Dict[str, Any]:
    """获取质检中心准确统计数据"""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM conversations").fetchone()["c"]
        # 待质检 = 没有管理员质检标注的对话
        pending = conn.execute(
            "SELECT COUNT(*) AS c FROM conversations c WHERE NOT EXISTS "
            "(SELECT 1 FROM admin_qa_marks q WHERE q.conversation_id = c.id)"
        ).fetchone()["c"]
        # 已质检
        qa_done = conn.execute(
            "SELECT COUNT(*) AS c FROM conversations c WHERE EXISTS "
            "(SELECT 1 FROM admin_qa_marks q WHERE q.conversation_id = c.id)"
        ).fetchone()["c"]
        # 未回答
        unanswered = conn.execute(
            "SELECT COUNT(*) AS c FROM unanswered_marks WHERE is_marked = 1"
        ).fetchone()["c"]
        # 有建议
        has_suggestion = conn.execute(
            "SELECT COUNT(DISTINCT conversation_id) AS c FROM improvement_suggestions"
        ).fetchone()["c"]
        # 有用户审查
        has_review = conn.execute(
            "SELECT COUNT(DISTINCT conversation_id) AS c FROM conversation_reviews"
        ).fetchone()["c"]
        # 质检结果分布
        qa_pass = conn.execute(
            "SELECT COUNT(DISTINCT q.conversation_id) AS c FROM admin_qa_marks q WHERE q.qa_result = 'pass'"
        ).fetchone()["c"]
        qa_fail = conn.execute(
            "SELECT COUNT(DISTINCT q.conversation_id) AS c FROM admin_qa_marks q WHERE q.qa_result = 'fail'"
        ).fetchone()["c"]
        qa_rework = conn.execute(
            "SELECT COUNT(DISTINCT q.conversation_id) AS c FROM admin_qa_marks q WHERE q.qa_result = 'rework'"
        ).fetchone()["c"]
    return {
        "total": total,
        "pending": pending,
        "qa_done": qa_done,
        "unanswered": unanswered,
        "has_suggestion": has_suggestion,
        "has_review": has_review,
        "qa_pass": qa_pass,
        "qa_fail": qa_fail,
        "qa_rework": qa_rework,
    }


# ============================================================
# 定时清理无审查对话
# ============================================================
def cleanup_unreviewed_conversations() -> int:
    """
    清理没有用户审查记录的对话（每天0点执行）
    只保留有 review（correct/incorrect/needs_improvement）的对话
    Returns: 删除的对话数
    """
    with _get_conn() as conn:
        # 找出没有审查记录的对话ID
        rows = conn.execute(
            "SELECT c.id FROM conversations c "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM conversation_reviews r WHERE r.conversation_id = c.id"
            ") AND NOT EXISTS ("
            "  SELECT 1 FROM admin_qa_marks q WHERE q.conversation_id = c.id"
            ") AND NOT EXISTS ("
            "  SELECT 1 FROM unanswered_marks u WHERE u.conversation_id = c.id AND u.is_marked = 1"
            ")"
        ).fetchall()
        deleted = 0
        for row in rows:
            conn.execute("DELETE FROM conversations WHERE id = ?", (row["id"],))
            deleted += 1
        conn.commit()
    if deleted > 0:
        logger.info("[Conversation] 定时清理: 删除 %d 条无审查对话", deleted)
        print(f"[Conversation] 定时清理: 删除 {deleted} 条无审查对话")
    return deleted


# ============================================================
# 用户工作台数据
# ============================================================
def get_user_marked_conversations(user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """
    获取用户标记过的对话（有审查记录或管理员质检反馈的）
    用于普通用户的"我的工作台"
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20
    offset = (page - 1) * page_size

    with _get_conn() as conn:
        # 查询有审查记录或被管理员质检标注的对话
        total = conn.execute(
            "SELECT COUNT(DISTINCT c.id) AS c FROM conversations c "
            "WHERE c.user_id = ? AND ("
            "  EXISTS (SELECT 1 FROM conversation_reviews r WHERE r.conversation_id = c.id) OR "
            "  EXISTS (SELECT 1 FROM admin_qa_marks q WHERE q.conversation_id = c.id) OR "
            "  EXISTS (SELECT 1 FROM unanswered_marks u WHERE u.conversation_id = c.id AND u.is_marked = 1)"
            ")",
            (user_id,)
        ).fetchone()["c"]

        rows = conn.execute(
            "SELECT c.* FROM conversations c "
            "WHERE c.user_id = ? AND ("
            "  EXISTS (SELECT 1 FROM conversation_reviews r WHERE r.conversation_id = c.id) OR "
            "  EXISTS (SELECT 1 FROM admin_qa_marks q WHERE q.conversation_id = c.id) OR "
            "  EXISTS (SELECT 1 FROM unanswered_marks u WHERE u.conversation_id = c.id AND u.is_marked = 1)"
            ") "
            "ORDER BY c.created_at DESC "
            "LIMIT ? OFFSET ?",
            (user_id, page_size, offset)
        ).fetchall()

    items = []
    for row in rows:
        conv = _enrich_conversation(_row_to_conversation(row))
        items.append(conv)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


# ============================================================
# 模块加载时自动初始化
# ============================================================
try:
    init_db()
except Exception as e:  # pragma: no cover
    logger.error("[Conversation] 对话数据库初始化失败: %s", e, exc_info=True)
    print(f"[Conversation] 对话数据库初始化失败: {e}")
