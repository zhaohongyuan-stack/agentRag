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


class RetrievalDB:
    """SQLite 文档与 Chunk 关系存储"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

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
