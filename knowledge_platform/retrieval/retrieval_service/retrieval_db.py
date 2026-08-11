"""
关系数据库 — 文档与 Chunk 持久化存储

功能：
- SQLite 存储文档元数据和 chunk 数据
- 父子关系查询（parent_chunk_id → children）
- 前后关系查询（prev_chunk_id → prev, next_chunk_id → next）
- 版本查询（同一 doc_name 的多个版本）
- 多字段组合过滤查询
- 批量导入 / 自动链接 chunks

使用方式：
    db = RetrievalDB("retrieval.db")
    db.open()
    db.upsert_document(doc_dict)
    db.insert_chunks(chunk_list)
    db.auto_link_chunks()          # 自动补全 prev/next 链
    parent = db.get_parent(chunk_id)
    children = db.get_children(chunk_id)
    context = db.get_context(chunk_id)  # 父 + 子 + 前 + 后
    versions = db.get_document_versions("资本管理办法")
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager


# ============================================================
# 数据库 Schema
# ============================================================

SCHEMA_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT    UNIQUE NOT NULL,
    doc_name    TEXT    NOT NULL,
    doc_title   TEXT    DEFAULT '',
    parser_type TEXT    DEFAULT '',
    source_file TEXT    DEFAULT '',
    parse_timestamp TEXT DEFAULT '',
    attachment_no    TEXT DEFAULT '',
    applicable_scope TEXT DEFAULT '全部',
    metadata_json    TEXT DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT    UNIQUE NOT NULL,
    doc_id      TEXT    NOT NULL,
    chunk_type  TEXT    NOT NULL DEFAULT 'clause',
    content     TEXT    NOT NULL DEFAULT '',
    hierarchy_path TEXT DEFAULT '',
    parent_chunk_id TEXT DEFAULT '',
    prev_chunk_id   TEXT DEFAULT '',
    next_chunk_id   TEXT DEFAULT '',
    chapter_number  TEXT DEFAULT '',
    clause_number   TEXT DEFAULT '',
    subclause_number TEXT DEFAULT '',
    applicable_scope TEXT DEFAULT '全部',
    normative_level  TEXT DEFAULT 'neutral',
    capital_tool_level TEXT DEFAULT '',
    table_name       TEXT DEFAULT '',
    table_section_name TEXT DEFAULT '',
    sheet_name       TEXT DEFAULT '',
    glossary_term    TEXT DEFAULT '',
    keywords_json    TEXT DEFAULT '[]',
    evidence_snippet TEXT DEFAULT '',
    content_raw      TEXT DEFAULT '',
    sub_chunks_json  TEXT DEFAULT '[]',
    metadata_json    TEXT DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
"""

# 常用索引：加速过滤和关系查询
SCHEMA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id     ON chunks(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_type       ON chunks(chunk_type);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_parent     ON chunks(parent_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_prev       ON chunks(prev_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_next       ON chunks(next_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_table      ON chunks(table_name);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_clause     ON chunks(clause_number);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_chapter    ON chunks(chapter_number);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_glossary   ON chunks(glossary_term);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_scope      ON chunks(applicable_scope);",
    "CREATE INDEX IF NOT EXISTS idx_docs_name         ON documents(doc_name);",
    "CREATE INDEX IF NOT EXISTS idx_docs_attachment   ON documents(attachment_no);",
]

# FTS5 全文索引虚拟表（用于 ExactRetriever 的精确/子串/前缀匹配）
# unicode61 tokenizer 按字分词，支持中文
SCHEMA_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    doc_name UNINDEXED,
    tokenize = 'unicode61'
);
"""


class RetrievalDB:
    """SQLite 文档与 Chunk 关系存储"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._fts5_available: bool = False

    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器，确保并发安全）"""
        if self._conn is None:
            raise RuntimeError("数据库未初始化，请先调用 open()")
        yield self._conn

    # ============================================================
    # 生命周期
    # ============================================================
    def open(self) -> "RetrievalDB":
        """打开/创建数据库并初始化表结构，支持链式调用"""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute(SCHEMA_DOCUMENTS)
        self._conn.execute(SCHEMA_CHUNKS)
        for idx_sql in SCHEMA_INDEXES:
            self._conn.execute(idx_sql)
        # FTS5 全文索引（可用性已在环境检查中验证）
        try:
            self._conn.execute(SCHEMA_FTS5)
        except sqlite3.OperationalError as e:
            print(f"  [RetrievalDB] FTS5 不可用: {e}（ExactRetriever 将降级为 LIKE 查询）")
            self._fts5_available = False
        else:
            self._fts5_available = True
        self._conn.commit()
        print(f"  [RetrievalDB] 数据库已打开: {self.db_path}")
        return self

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    # ============================================================
    # 文档操作
    # ============================================================
    def upsert_document(self, doc: Dict[str, Any]):
        """
        插入或更新文档。
        doc 字段：doc_id, doc_name, doc_title, parser_type,
                  source_file, parse_timestamp, attachment_no,
                  applicable_scope, metadata(dict)
        """
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO documents (doc_id, doc_name, doc_title, parser_type,
                    source_file, parse_timestamp, attachment_no, applicable_scope,
                    metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    doc_name=excluded.doc_name,
                    doc_title=excluded.doc_title,
                    parser_type=excluded.parser_type,
                    source_file=excluded.source_file,
                    parse_timestamp=excluded.parse_timestamp,
                    attachment_no=excluded.attachment_no,
                    applicable_scope=excluded.applicable_scope,
                    metadata_json=excluded.metadata_json
            """, (
                doc.get("doc_id", ""),
                doc.get("doc_name", ""),
                doc.get("doc_title", ""),
                doc.get("parser_type", ""),
                doc.get("source_file", ""),
                doc.get("parse_timestamp", ""),
                doc.get("attachment_no", ""),
                doc.get("applicable_scope", "全部"),
                json.dumps(doc.get("metadata", {}), ensure_ascii=False),
            ))
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取单个文档信息"""
        return self._fetch_one("documents", "doc_id", doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档（摘要字段）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT doc_id, doc_name, doc_title, parser_type,
                          attachment_no, applicable_scope, parse_timestamp
                   FROM documents ORDER BY doc_name"""
            ).fetchall()
        return [
            {"doc_id": r[0], "doc_name": r[1], "doc_title": r[2],
             "parser_type": r[3], "attachment_no": r[4],
             "applicable_scope": r[5], "parse_timestamp": r[6]}
            for r in rows
        ]

    def get_document_versions(self, doc_name: str) -> List[Dict[str, Any]]:
        """
        获取同一文档名的所有版本（按解析时间降序排列）。
        用途：当法规文件更新时，查询同一文件的不同解析版本。
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT doc_id, doc_name, doc_title, parser_type,
                          parse_timestamp, source_file, created_at
                   FROM documents
                   WHERE doc_name = ?
                   ORDER BY parse_timestamp DESC, created_at DESC""",
                (doc_name,)
            ).fetchall()
        return [
            {"doc_id": r[0], "doc_name": r[1], "doc_title": r[2],
             "parser_type": r[3], "parse_timestamp": r[4],
             "source_file": r[5], "created_at": r[6]}
            for r in rows
        ]

    def get_latest_version(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """获取文档的最新解析版本"""
        versions = self.get_document_versions(doc_name)
        return versions[0] if versions else None

    # ============================================================
    # Chunk 批量操作
    # ============================================================
    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """批量插入 chunks（INSERT OR REPLACE，幂等）"""
        with self._get_conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, chunk_type, content, hierarchy_path,
                 parent_chunk_id, prev_chunk_id, next_chunk_id,
                 chapter_number, clause_number, subclause_number,
                 applicable_scope, normative_level, capital_tool_level,
                 table_name, table_section_name, sheet_name,
                 glossary_term, keywords_json,
                 evidence_snippet, content_raw, sub_chunks_json, metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                (
                    c.get("chunk_id", ""),
                    c.get("doc_id", ""),
                    c.get("chunk_type", "clause"),
                    c.get("content", ""),
                    c.get("hierarchy_path", ""),
                    c.get("parent_chunk_id", ""),
                    c.get("prev_chunk_id", ""),
                    c.get("next_chunk_id", ""),
                    c.get("chapter_number", ""),
                    c.get("clause_number", ""),
                    c.get("subclause_number", ""),
                    c.get("applicable_scope", "全部"),
                    c.get("normative_level", "neutral"),
                    c.get("capital_tool_level", ""),
                    c.get("table_name", ""),
                    c.get("table_section_name", ""),
                    c.get("sheet_name", ""),
                    c.get("glossary_term", ""),
                    json.dumps(c.get("keywords", []), ensure_ascii=False),
                    c.get("evidence_snippet", ""),
                    c.get("content_raw", ""),
                    json.dumps(c.get("sub_chunks", []), ensure_ascii=False),
                    json.dumps(c.get("metadata", {}), ensure_ascii=False),
                )
                for c in chunks
            ])
            conn.commit()
        print(f"  [RetrievalDB] 已写入 {len(chunks)} 条 chunks")

    def auto_link_chunks(self):
        """
        自动补全 prev_chunk_id / next_chunk_id 链。
        对于同一个 doc 内的 chunks，按 clause_number → subclause_number 自然排序后链接。
        如果原数据已有链接则跳过。
        """
        with self._get_conn() as conn:
            # 获取每个 doc_id 的 chunk 列表（按自然顺序）
            doc_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT doc_id FROM chunks").fetchall()]

            updated = 0
            for doc_id in doc_ids:
                chunks = conn.execute(
                    """SELECT chunk_id, prev_chunk_id, next_chunk_id,
                              clause_number, subclause_number
                       FROM chunks WHERE doc_id = ?
                       ORDER BY
                         CAST(clause_number AS INTEGER) ASC,
                         CAST(subclause_number AS INTEGER) ASC,
                         id ASC""",
                    (doc_id,)
                ).fetchall()

                for i in range(len(chunks)):
                    cid, prev, nxt, _, _ = chunks[i]
                    new_prev = chunks[i - 1][0] if i > 0 else ""
                    new_next = chunks[i + 1][0] if i < len(chunks) - 1 else ""

                    # 只在原字段为空时才自动填充
                    if not prev and new_prev:
                        conn.execute(
                            "UPDATE chunks SET prev_chunk_id = ? WHERE chunk_id = ?",
                            (new_prev, cid))
                        updated += 1
                    if not nxt and new_next:
                        conn.execute(
                            "UPDATE chunks SET next_chunk_id = ? WHERE chunk_id = ?",
                            (new_next, cid))
                        updated += 1

            conn.commit()
            if updated:
                print(f"  [RetrievalDB] 自动链接了 {updated} 条 prev/next 关系")

    def delete_document(self, doc_id: str):
        """级联删除文档及其所有 chunks"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

    # ============================================================
    # Chunk 单条查询
    # ============================================================
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取单个 chunk"""
        return self._fetch_one("chunks", "chunk_id", chunk_id)

    def get_parent(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取父 chunk（通过 parent_chunk_id）"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.get("parent_chunk_id"):
            return self.get_chunk(chunk["parent_chunk_id"])
        return None

    def get_children(self, chunk_id: str) -> List[Dict[str, Any]]:
        """获取直接子 chunks"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE parent_chunk_id = ? ORDER BY id",
                (chunk_id,)
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_prev(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """前一个 chunk（沿 prev_chunk_id 链）"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.get("prev_chunk_id"):
            return self.get_chunk(chunk["prev_chunk_id"])
        return None

    def get_next(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """后一个 chunk（沿 next_chunk_id 链）"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.get("next_chunk_id"):
            return self.get_chunk(chunk["next_chunk_id"])
        return None

    def get_siblings(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        获取同级 chunks（同一 parent 下的所有子 chunk，含自身）。
        适用于查看同一个条款下面的所有子条款，或同一个表格块下的所有分块。
        """
        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return []
        parent_id = chunk.get("parent_chunk_id", "")
        if not parent_id:
            # 没有父节点时，取同 doc 且同级（parent_chunk_id 也为空）的 chunk
            with self._get_conn() as conn:
                rows = conn.execute(
                    """SELECT * FROM chunks
                       WHERE doc_id = ? AND (parent_chunk_id = '' OR parent_chunk_id IS NULL)
                       ORDER BY id""",
                    (chunk.get("doc_id", ""),)
                ).fetchall()
        else:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM chunks WHERE parent_chunk_id = ? ORDER BY id",
                    (parent_id,)
                ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_surrounding(self, chunk_id: str, window: int = 2) -> List[Dict[str, Any]]:
        """
        获取当前 chunk 及前后各 window 个邻居。
        适用于给 LLM 提供上下文窗口。
        """
        results = []
        # 向前遍历
        curr_id = chunk_id
        for _ in range(window):
            chunk = self.get_chunk(curr_id)
            if not chunk or not chunk.get("prev_chunk_id"):
                break
            prev = self.get_chunk(chunk["prev_chunk_id"])
            if prev:
                results.insert(0, prev)
                curr_id = prev["chunk_id"]
            else:
                break
        # 当前
        curr = self.get_chunk(chunk_id)
        if curr:
            results.append(curr)
        # 向后遍历
        curr_id = chunk_id
        for _ in range(window):
            chunk = self.get_chunk(curr_id)
            if not chunk or not chunk.get("next_chunk_id"):
                break
            nxt = self.get_chunk(chunk["next_chunk_id"])
            if nxt:
                results.append(nxt)
                curr_id = nxt["chunk_id"]
            else:
                break
        return results

    def get_context(self, chunk_id: str) -> Dict[str, Any]:
        """
        获取完整关系上下文：
        - chunk: 自身
        - parent: 父 chunk
        - children: 子 chunks 列表
        - siblings: 同级 chunks 列表
        - prev / next: 前 / 后各 1 个邻居
        - doc: 所属文档信息
        """
        chunk = self.get_chunk(chunk_id)
        return {
            "chunk": chunk,
            "parent": self.get_parent(chunk_id) if chunk else None,
            "children": self.get_children(chunk_id),
            "siblings": self.get_siblings(chunk_id),
            "prev": self.get_prev(chunk_id),
            "next": self.get_next(chunk_id),
            "doc": self.get_document(chunk["doc_id"]) if chunk and chunk.get("doc_id") else None,
        }

    # ============================================================
    # Chunk 批量查询（字段过滤）
    # ============================================================
    def search_chunks(self,
                      doc_id: Optional[str] = None,
                      chunk_type: Optional[str] = None,
                      table_name: Optional[str] = None,
                      clause_number: Optional[str] = None,
                      chapter_number: Optional[str] = None,
                      glossary_term: Optional[str] = None,
                      applicable_scope: Optional[str] = None,
                      normative_level: Optional[str] = None,
                      parent_chunk_id: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> List[Dict[str, Any]]:
        """
        结构化多字段组合查询。所有字段可选，AND 组合。

        参数示例：
          search_chunks(doc_id="400", chunk_type="clause",
                        clause_number="12", applicable_scope="全部")
        """
        conditions = []
        params = []

        _eq = lambda col, val: (conditions.append(f"{col} = ?"), params.append(val))
        _like = lambda col, val: (conditions.append(f"{col} LIKE ?"), params.append(f"%{val}%"))

        if doc_id:             _eq("doc_id", doc_id)
        if chunk_type:         _eq("chunk_type", chunk_type)
        if table_name:         _eq("table_name", table_name)
        if clause_number:      _eq("clause_number", clause_number)
        if chapter_number:     _eq("chapter_number", chapter_number)
        if applicable_scope:   _eq("applicable_scope", applicable_scope)
        if normative_level:    _eq("normative_level", normative_level)
        if parent_chunk_id:    _eq("parent_chunk_id", parent_chunk_id)
        if glossary_term:      _like("glossary_term", glossary_term)

        where = " AND ".join(conditions) if conditions else "1=1"

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE {where} ORDER BY id LIMIT ? OFFSET ?",
                params + [limit, offset]
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def count_chunks(self, doc_id: Optional[str] = None) -> int:
        """统计 chunk 总数（可按 doc_id 过滤）"""
        with self._get_conn() as conn:
            if doc_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunk_ids_by_doc(self, doc_id: str) -> List[str]:
        """获取某文档下所有 chunk_id"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY id",
                (doc_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def get_existing_chunk_ids(self) -> set:
        """获取数据库中所有已存在的 chunk_id 集合（用于增量写入判断）"""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks").fetchall()
        return {r[0] for r in rows}

    def get_existing_doc_ids(self) -> set:
        """获取数据库中所有已存在的 doc_id 集合（用于增量写入判断）"""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT doc_id FROM documents").fetchall()
        return {r[0] for r in rows}

    # ============================================================
    # 灵活元数据过滤（MetadataRetriever DB 驱动模式）
    # ============================================================

    # 字段 → DB 列映射
    _CHUNK_FILTER_COLS: Dict[str, str] = {
        "chunk_type":         "chunk_type",
        "doc_id":             "doc_id",
        "content":            "content",
        "hierarchy_path":     "hierarchy_path",
        "parent_chunk_id":    "parent_chunk_id",
        "chapter_number":     "chapter_number",
        "clause_number":      "clause_number",
        "subclause_number":   "subclause_number",
        "applicable_scope":   "applicable_scope",
        "normative_level":    "normative_level",
        "capital_tool_level": "capital_tool_level",
        "table_name":         "table_name",
        "table_section_name": "table_section_name",
        "sheet_name":         "sheet_name",
        "glossary_term":      "glossary_term",
    }
    _DOC_FILTER_COLS: Dict[str, str] = {
        "doc_name":      "doc_name",
        "doc_title":     "doc_title",
        "parser_type":   "parser_type",
        "attachment_no": "attachment_no",
    }
    # doc_name/doc_title 使用子串匹配（与旧版 MetadataRetriever._match_eq 一致）
    _SUBSTRING_FIELDS = {"doc_name", "doc_title"}
    # keywords 存储为 JSON 数组文本
    _KEYWORDS_FIELD = "keywords"

    def search_by_filters(self, filters: Dict[str, Any],
                          limit: int = 100) -> List[str]:
        """
        灵活的元数据过滤查询（SQL WHERE 驱动）。

        支持的操作符：
          eq (默认): 精确匹配（doc_name/doc_title 做子串匹配，keywords 做 JSON 包含）
          in:        列表包含（OR 语义）
          contains:  子串包含 (LIKE %value%)
          regex:     正则匹配（LIKE 粗筛 + Python re 精排）
          gt/gte/lt/lte: 数值比较 (CAST AS REAL)
          prefix:    前缀匹配 (LIKE value%)
          suffix:    后缀匹配 (LIKE %value)

        参数：
          filters: 过滤条件字典
                   简单格式: {"chunk_type": "clause"}
                   扩展格式: {"clause_number": {"value": ["12","13"], "op": "in"}}
          limit:   最多返回条数

        返回：
          符合所有条件的 chunk_id 列表
        """
        import re as re_module

        conditions: List[str] = []
        params: List[Any] = []
        needs_join = False
        # 收集需要 post-filter 的正则条件
        regex_post_filters: List[tuple] = []  # [(field, pattern)]

        for field, condition in filters.items():
            if isinstance(condition, dict):
                value = condition.get("value")
                op = condition.get("op", "eq")
            else:
                value = condition
                op = "eq"

            if value is None or value == "":
                continue

            # 确定 DB 列
            if field in self._CHUNK_FILTER_COLS:
                col = f"c.{self._CHUNK_FILTER_COLS[field]}"
            elif field in self._DOC_FILTER_COLS:
                col = f"d.{self._DOC_FILTER_COLS[field]}"
                needs_join = True
            elif field == self._KEYWORDS_FIELD:
                col = "c.keywords_json"
            else:
                continue  # 未知字段跳过

            # 按操作符构建 SQL 条件
            if op == "eq":
                if field in self._SUBSTRING_FIELDS:
                    conditions.append(f"{col} LIKE ?")
                    params.append(f"%{value}%")
                elif field == self._KEYWORDS_FIELD:
                    conditions.append(f"{col} LIKE ?")
                    params.append(f'%"{value}"%')
                else:
                    conditions.append(f"{col} = ?")
                    params.append(str(value))

            elif op == "in":
                if not isinstance(value, list):
                    value = [value]
                if field in self._SUBSTRING_FIELDS or field == self._KEYWORDS_FIELD:
                    or_parts = []
                    for v in value:
                        or_parts.append(f"{col} LIKE ?")
                        if field == self._KEYWORDS_FIELD:
                            params.append(f'%"{v}"%')
                        else:
                            params.append(f"%{v}%")
                    conditions.append(f"({' OR '.join(or_parts)})")
                else:
                    placeholders = ",".join("?" * len(value))
                    conditions.append(f"{col} IN ({placeholders})")
                    params.extend(str(v) for v in value)

            elif op == "contains":
                if field == self._KEYWORDS_FIELD:
                    conditions.append(f"{col} LIKE ?")
                    params.append(f'%"{value}"%')
                else:
                    conditions.append(f"{col} LIKE ?")
                    params.append(f"%{value}%")

            elif op == "prefix":
                conditions.append(f"{col} LIKE ?")
                params.append(f"{value}%")

            elif op == "suffix":
                conditions.append(f"{col} LIKE ?")
                params.append(f"%{value}")

            elif op in ("gt", "gte", "lt", "lte"):
                op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
                conditions.append(f"CAST({col} AS REAL) {op_map[op]} ?")
                params.append(float(value))

            elif op == "regex":
                # SQL 不支持正则，用 LIKE 粗筛 + Python re 精排
                try:
                    regex = re_module.compile(str(value), re_module.IGNORECASE)
                except re_module.error:
                    continue
                regex_post_filters.append((field, regex))
                # 提取字面字符做 LIKE 预筛
                literal_parts = re_module.findall(r'[a-zA-Z0-9\u4e00-\u9fff]+', str(value))
                if literal_parts:
                    like_term = max(literal_parts, key=len)
                    conditions.append(f"{col} LIKE ?")
                    params.append(f"%{like_term}%")

        # 构建 SQL
        join_clause = " LEFT JOIN documents d ON c.doc_id = d.doc_id" if needs_join else ""
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        if regex_post_filters:
            # 需要取回更多数据做正则 post-filter
            query = f"""
                SELECT c.chunk_id, c.content, c.keywords_json, c.hierarchy_path,
                       c.chunk_type, c.doc_id, c.parent_chunk_id, c.chapter_number,
                       c.clause_number, c.subclause_number, c.applicable_scope,
                       c.normative_level, c.capital_tool_level, c.table_name,
                       c.table_section_name, c.sheet_name, c.glossary_term,
                       d.doc_name, d.doc_title, d.parser_type, d.attachment_no
                FROM chunks c{join_clause}
                WHERE {where_clause}
                LIMIT ?
            """
            with self._get_conn() as conn:
                rows = conn.execute(query, params + [limit * 5]).fetchall()

            # Post-filter with regex
            result_ids = []
            for row in rows:
                row_dict = dict(zip([
                    "chunk_id", "content", "keywords_json", "hierarchy_path",
                    "chunk_type", "doc_id", "parent_chunk_id", "chapter_number",
                    "clause_number", "subclause_number", "applicable_scope",
                    "normative_level", "capital_tool_level", "table_name",
                    "table_section_name", "sheet_name", "glossary_term",
                    "doc_name", "doc_title", "parser_type", "attachment_no",
                ], row))

                match_all = True
                for field, regex in regex_post_filters:
                    actual = str(row_dict.get(field, ""))
                    if not regex.search(actual):
                        match_all = False
                        break

                if match_all:
                    result_ids.append(row_dict["chunk_id"])
                    if len(result_ids) >= limit:
                        break
            return result_ids
        else:
            query = f"""
                SELECT c.chunk_id
                FROM chunks c{join_clause}
                WHERE {where_clause}
                ORDER BY c.id
                LIMIT ?
            """
            with self._get_conn() as conn:
                rows = conn.execute(query, params + [limit]).fetchall()
            return [r[0] for r in rows]

    def list_field_values_db(self, field: str) -> List[str]:
        """
        列出某个字段的所有唯一值（DB 驱动，用于构建过滤选项 UI）。
        """
        if field in self._CHUNK_FILTER_COLS:
            col = self._CHUNK_FILTER_COLS[field]
            with self._get_conn() as conn:
                rows = conn.execute(
                    f"SELECT DISTINCT {col} FROM chunks WHERE {col} != '' ORDER BY {col}"
                ).fetchall()
            return [str(r[0]) for r in rows if r[0]]
        elif field in self._DOC_FILTER_COLS:
            col = self._DOC_FILTER_COLS[field]
            with self._get_conn() as conn:
                rows = conn.execute(
                    f"""SELECT DISTINCT d.{col} FROM documents d
                        WHERE d.{col} != '' ORDER BY d.{col}"""
                ).fetchall()
            return [str(r[0]) for r in rows if r[0]]
        return []

    def count_by_field_db(self, field: str) -> Dict[str, int]:
        """
        按字段值分组计数（DB 驱动，返回 {value: count} 字典）。
        对 chunk 字段统计 chunk 数，对 doc 字段通过 JOIN 统计 chunk 数。
        """
        if field in self._CHUNK_FILTER_COLS:
            col = self._CHUNK_FILTER_COLS[field]
            with self._get_conn() as conn:
                rows = conn.execute(
                    f"SELECT {col}, COUNT(*) FROM chunks GROUP BY {col} ORDER BY COUNT(*) DESC"
                ).fetchall()
            return {(str(r[0]) if r[0] else "(空)"): r[1] for r in rows}
        elif field in self._DOC_FILTER_COLS:
            col = self._DOC_FILTER_COLS[field]
            with self._get_conn() as conn:
                rows = conn.execute(
                    f"""SELECT d.{col}, COUNT(*) FROM chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        GROUP BY d.{col} ORDER BY COUNT(*) DESC"""
                ).fetchall()
            return {(str(r[0]) if r[0] else "(空)"): r[1] for r in rows}
        return {}

    # ============================================================
    # FTS5 全文索引
    # ============================================================
    @staticmethod
    def _tokenize_cjk(text: str) -> str:
        """
        将 CJK 字符之间插入空格，使 FTS5 unicode61 tokenizer 按字分词。

        unicode61 默认将 CJK 连续字符视为单个 token（如 "总资产" → 1 个 token），
        导致按字查询（"资"）无法命中。插入空格后每个字符成为独立 token。

        非 CJK 字符（ASCII、标点等）保持不变。
        """
        if not text:
            return ""
        result = []
        for ch in text:
            # CJK 统一表意文字范围：U+4E00 ~ U+9FFF
            # CJK 扩展 A：U+3400 ~ U+4DBF
            # CJK 兼容：U+F900 ~ U+FAFF
            if ('\u4e00' <= ch <= '\u9fff' or
                '\u3400' <= ch <= '\u4dbf' or
                '\uf900' <= ch <= '\ufaff'):
                result.append(f" {ch} ")
            else:
                result.append(ch)
        return "".join(result)

    def populate_fts5_index(self):
        """
        从 chunks 表填充 FTS5 索引。
        先清空旧索引，再全量插入（幂等）。
        在 insert_chunks 后调用。

        ⚠️ CJK 预分词：content 和 doc_name 在写入 FTS5 前经过 _tokenize_cjk，
        使每个中文字符成为独立 token，支持按字 MATCH 查询。
        """
        if not self._fts5_available:
            return

        with self._get_conn() as conn:
            # 清空旧索引
            conn.execute("DELETE FROM chunks_fts")
            # 从 chunks 表批量插入（CJK 预分词）
            rows = conn.execute("""
                SELECT c.chunk_id, c.content,
                       COALESCE(d.doc_name, '')
                FROM chunks c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """).fetchall()

            conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, content, doc_name) VALUES (?, ?, ?)",
                [(r[0], self._tokenize_cjk(r[1]), self._tokenize_cjk(r[2])) for r in rows]
            )
            conn.commit()
            count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            print(f"  [RetrievalDB] FTS5 索引已填充: {count} 条（CJK 预分词）")

    def fts5_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        FTS5 MATCH 全文搜索（用于 contains 模式）。

        参数：
          query:  搜索文本（自动按字拆分为 FTS5 查询表达式）
          top_k:  返回条数

        返回：
          [{"chunk_id": str, "score": float, "match_pos": int}, ...]
          按 BM25 分数降序
        """
        if not self._fts5_available:
            return self._like_search(query, top_k)

        # 构造 FTS5 查询表达式：将每个非空字符用引号包裹，用 OR 连接
        # FTS5 unicode61 对中文按字分词，所以查询也要按字拆分
        terms = []
        for ch in query:
            ch = ch.strip()
            if ch and ch not in ('"', '*', '(', ')', ':'):
                terms.append(f'"{ch}"')
        if not terms:
            return []

        fts_query = " OR ".join(terms)

        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """SELECT chunk_id, bm25(chunks_fts) as score
                       FROM chunks_fts
                       WHERE chunks_fts MATCH ?
                       ORDER BY score
                       LIMIT ?""",
                    (fts_query, top_k)
                ).fetchall()

            results = []
            for r in rows:
                # bm25() 返回负值（越小越相关），转换为正分
                score = -float(r[1]) if r[1] else 0.0
                results.append({
                    "chunk_id": r[0],
                    "score": round(score, 4),
                    "match_pos": 0,
                })
            return results
        except sqlite3.OperationalError:
            # FTS5 查询语法错误时降级为 LIKE
            return self._like_search(query, top_k)

    def _like_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        LIKE 子串搜索（FTS5 不可用时的降级方案）。

        返回：
          [{"chunk_id": str, "score": float, "match_pos": int}, ...]
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT chunk_id, content FROM chunks
                   WHERE content LIKE ?
                   LIMIT ?""",
                (f"%{query}%", top_k * 2)
            ).fetchall()

        results = []
        query_lower = query.lower()
        for r in rows:
            chunk_id, content = r[0], r[1] or ""
            count = content.lower().count(query_lower)
            score = min(count, 10) / 10.0
            pos = content.lower().find(query_lower)
            results.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "match_pos": pos if pos >= 0 else 0,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def regex_search(self, pattern: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        正则搜索：SQL LIKE 粗筛 + Python re 精排。
        FTS5 不支持正则，所以分两步走。

        返回：
          [{"chunk_id": str, "score": float, "match_pos": int}, ...]
        """
        import re as re_module
        try:
            regex = re_module.compile(pattern, re_module.IGNORECASE)
        except re_module.error:
            return []

        # LIKE 粗筛：提取正则中的字面字符做 LIKE 查询
        # 简化策略：取 pattern 中最长的连续字母数字子串做 LIKE
        literal_parts = re_module.findall(r'[a-zA-Z0-9\u4e00-\u9fff]+', pattern)
        like_term = max(literal_parts, key=len) if literal_parts else ""

        with self._get_conn() as conn:
            if like_term:
                rows = conn.execute(
                    """SELECT chunk_id, content FROM chunks
                       WHERE content LIKE ?
                       LIMIT ?""",
                    (f"%{like_term}%", top_k * 5)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT chunk_id, content FROM chunks LIMIT ?", (top_k * 5,)
                ).fetchall()

        results = []
        for r in rows:
            chunk_id, content = r[0], r[1] or ""
            matches = regex.findall(content)
            if matches:
                score = min(len(matches), 10) / 10.0
                m = regex.search(content)
                results.append({
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "match_pos": m.start() if m else 0,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def prefix_search(self, prefix: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        前缀搜索：查找 content 中任意行以指定前缀开头的 chunk。

        返回：
          [{"chunk_id": str, "score": float, "match_pos": int}, ...]
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT chunk_id, content FROM chunks
                   WHERE content LIKE ?
                   LIMIT ?""",
                (f"{prefix}%", top_k * 3)
            ).fetchall()

        results = []
        prefix_lower = prefix.lower()
        for r in rows:
            chunk_id, content = r[0], r[1] or ""
            # 检查每一行是否以 prefix 开头
            for line in content.split("\n"):
                if line.strip().lower().startswith(prefix_lower):
                    results.append({
                        "chunk_id": chunk_id,
                        "score": 1.0,
                        "match_pos": 0,
                    })
                    break
        return results[:top_k]

    def exact_match(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        精确匹配：content 去除空白后与 query 完全相等。

        返回：
          [{"chunk_id": str, "score": float, "match_pos": int}, ...]
        """
        import re as re_module
        # 用 SQL 做初步筛选（content = query），去除首尾空白
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT chunk_id, content FROM chunks
                   WHERE TRIM(content) = TRIM(?)
                   LIMIT ?""",
                (query, top_k)
            ).fetchall()

        results = []
        for r in rows:
            results.append({
                "chunk_id": r[0],
                "score": 1.0,
                "match_pos": 0,
            })
        return results

    # ============================================================
    # 数据导出
    # ============================================================
    def export_documents(self) -> List[Dict[str, Any]]:
        """导出全部文档记录（含 metadata_json 反序列化）"""
        with self._get_conn() as conn:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(documents)")]
            rows = conn.execute("SELECT * FROM documents ORDER BY doc_name").fetchall()
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            if "metadata_json" in d and isinstance(d["metadata_json"], str):
                try:
                    d["metadata"] = json.loads(d["metadata_json"])
                except json.JSONDecodeError:
                    d["metadata"] = {}
            results.append(d)
        return results

    def export_chunks(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """导出 chunk 记录（可按 doc_id 过滤），JSON 字段全部反序列化"""
        with self._get_conn() as conn:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(chunks)")]
            if doc_id:
                rows = conn.execute(
                    "SELECT * FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def export_to_jsonl(self, output_dir: str):
        """
        将整个数据库导出为 JSONL + summary 格式（与 regulatory_docs 目录结构一致），
        方便备份、迁移或供其他工具使用。
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        doc_ids = [r[0] for r in
                   self._conn.execute("SELECT DISTINCT doc_id FROM documents ORDER BY doc_name").fetchall()]

        total_chunks = 0
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            chunks = self.export_chunks(doc_id=doc_id)

            # 写 chunks.jsonl
            chunk_path = out / f"{doc_id}_chunks.jsonl"
            with open(chunk_path, "w", encoding="utf-8") as f:
                for c in chunks:
                    # 将 DB 的扁平列还原为原始 JSON 结构
                    record = self._chunk_to_jsonl_record(c)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 写 summary.json
            summary_path = out / f"{doc_id}_summary.json"
            summary = {
                "doc_name": doc.get("doc_name", ""),
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "chunk_type_distribution": self._count_chunk_types(chunks),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            total_chunks += len(chunks)

        print(f"  [RetrievalDB] 已导出 {len(doc_ids)} 个文档、{total_chunks} 条 chunks → {output_dir}")

    @staticmethod
    def _chunk_to_jsonl_record(c: Dict[str, Any]) -> Dict[str, Any]:
        """将 DB 扁平 chunk 记录还原为原始 JSONL 格式"""
        meta = c.get("metadata", {})
        # 把顶层列合并回 metadata
        for key in ("doc_id", "chapter_number", "clause_number", "subclause_number",
                     "applicable_scope", "normative_level", "capital_tool_level",
                     "table_name", "table_section_name", "sheet_name",
                     "glossary_term", "keywords"):
            if key in c and c[key]:
                meta[key] = c[key]
        return {
            "chunk_id": c.get("chunk_id", ""),
            "chunk_type": c.get("chunk_type", "clause"),
            "hierarchy_path": c.get("hierarchy_path", ""),
            "content": c.get("content", ""),
            "content_raw": c.get("content_raw", ""),
            "evidence_snippet": c.get("evidence_snippet", ""),
            "parent_chunk_id": c.get("parent_chunk_id") or None,
            "sub_chunks": c.get("sub_chunks", []) if isinstance(c.get("sub_chunks"), list) else [],
            "metadata": meta,
        }

    @staticmethod
    def _count_chunk_types(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计 chunk_type 分布"""
        from collections import Counter
        return dict(Counter(c.get("chunk_type", "unknown") for c in chunks))

    def export_tables(self, output_dir: str):
        """
        将两张表直接导出为独立文件：
          - documents.json  — 全部文档（JSON 数组）
          - chunks.jsonl    — 全部 chunk（JSONL，每行一条）
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 表 1：documents
        docs = self.export_documents()
        docs_path = out / "documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print(f"  documents.json  → {len(docs)} 行")

        # 表 2：chunks
        chunks = self.export_chunks()
        chunks_path = out / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  chunks.jsonl    → {len(chunks)} 行")

        print(f"  [RetrievalDB] 两张表已导出 → {output_dir}")

    # ============================================================
    # 内部工具方法
    # ============================================================
    def _fetch_one(self, table: str, key_col: str, key_val: str) -> Optional[Dict[str, Any]]:
        """通用单条查询"""
        with self._get_conn() as conn:
            # PRAGMA table_info 返回: (cid, name, type, notnull, dflt_value, pk)
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
            row = conn.execute(
                f"SELECT * FROM {table} WHERE {key_col} = ?", (key_val,)
            ).fetchone()
            if row:
                d = dict(zip(cols, row))
                return self._deserialize_json_fields(d)
        return None

    def _row_to_chunk(self, row) -> Dict[str, Any]:
        """将 SQLite 行转为 chunk 字典（含 JSON 反序列化）"""
        if not hasattr(self, '_chunks_cols'):
            with self._get_conn() as conn:
                self._chunks_cols = [c[1] for c in conn.execute("PRAGMA table_info(chunks)")]
        d = dict(zip(self._chunks_cols, row))
        return self._deserialize_json_fields(d)

    @staticmethod
    def _deserialize_json_fields(d: Dict[str, Any]) -> Dict[str, Any]:
        """反序列化 JSON 文本字段"""
        for field in ["keywords_json", "sub_chunks_json", "metadata_json"]:
            if field in d and isinstance(d[field], str):
                try:
                    # 映射到更友好的 key 名
                    if field == "keywords_json":
                        d["keywords"] = json.loads(d[field])
                    elif field == "sub_chunks_json":
                        d["sub_chunks"] = json.loads(d[field])
                    elif field == "metadata_json":
                        d["metadata"] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        return d
