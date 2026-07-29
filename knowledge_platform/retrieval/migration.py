"""
数据库建表与数据导入模块 — migration

功能：
- 从 regulatory_docs/ 目录读取 JSON 文件（chunks + summary）
- 创建 SQLite 数据库表结构（documents + chunks）
- 批量导入文档元数据和 chunk 数据
- 自动补全 prev/next chunk 链接

使用方式：
    # 方式一：命令行直接运行
    python migration.py

    # 方式二：代码调用
    from migration import migrate
    db = migrate("regulatory_docs/", db_path="retrieval.db")

    # 方式三：自定义数据目录
    from migration import MigrationRunner
    runner = MigrationRunner(db_path="retrieval.db")
    runner.run("regulatory_docs/")
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .retrieval_db import RetrievalDB


# ============================================================
# 数据扫描：从 JSON 文件提取文档和 chunk 数据
# ============================================================

def scan_regulatory_docs(data_dir: str) -> List[Tuple[str, str, str]]:
    """
    扫描 regulatory_docs 目录，返回 (doc_id, summary_path, chunks_path) 列表。
    自动配对 *_summary.json 和 *_chunks.jsonl 文件。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 收集所有 summary 文件
    summaries = sorted(data_path.glob("*_summary.json"))
    doc_groups: List[Tuple[str, str, str]] = []

    for summary_file in summaries:
        # 从 "001_summary.json" 提取前缀 "001"
        stem = summary_file.name.replace("_summary.json", "")
        chunks_file = data_path / f"{stem}_chunks.jsonl"

        if not chunks_file.exists():
            print(f"  [跳过] {stem}: 缺少对应的 chunks 文件")
            continue

        doc_id = stem  # 如 "001", "002", ...
        doc_groups.append((doc_id, str(summary_file), str(chunks_file)))

    return doc_groups


def load_document_from_summary(summary_path: str) -> Dict[str, Any]:
    """从 *_summary.json 文件加载文档元数据"""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    doc_id = summary.get("doc_id", "")
    doc_name = summary.get("doc_name", "")
    # 从文件名中提取文档标题（去掉 doc_id 前缀和扩展名等）
    doc_title = doc_name

    stats = summary.get("stats", {})

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "doc_title": doc_title,
        "parser_type": stats.get("file_type", ""),
        "source_file": doc_name,
        "parse_timestamp": "",  # summary 中通常没有，chunk metadata 里有
        "attachment_no": "",
        "applicable_scope": "全部",
        "metadata": {
            "chunk_count": summary.get("chunk_count", 0),
            "chunk_type_distribution": summary.get("chunk_type_distribution", {}),
            "error_count": summary.get("error_count", 0),
            "warn_count": summary.get("warn_count", 0),
            "stats": stats,
        },
    }


def load_chunks_from_jsonl(chunks_path: str) -> List[Dict[str, Any]]:
    """
    从 *_chunks.jsonl 文件加载所有 chunk 记录，
    并转换为 RetrievalDB.insert_chunks() 所需的格式。
    """
    chunks_for_db: List[Dict[str, Any]] = []

    # 定义需要从 metadata 中提取到顶层列的字段（与 DB schema 对齐）
    TOP_META_KEYS = {
        "chapter_number", "clause_number", "subclause_number",
        "applicable_scope", "normative_level", "capital_tool_level",
        "table_name", "table_section_name", "sheet_name",
        "glossary_term",
    }

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [警告] JSON 解析失败，跳过行: {line[:80]}...")
                continue

            meta = raw.get("metadata", {})

            # 从 metadata 提取顶层字段
            top_fields = {}
            for key in TOP_META_KEYS:
                if key in meta:
                    top_fields[key] = meta[key] if meta[key] is not None else ""

            # 构建 insert_chunks 所需的字典
            chunk_dict = {
                "chunk_id": raw.get("chunk_id", ""),
                "doc_id": meta.get("doc_id", ""),
                "chunk_type": raw.get("chunk_type", "clause"),
                "content": raw.get("content", ""),
                "hierarchy_path": raw.get("hierarchy_path", ""),
                "parent_chunk_id": raw.get("parent_chunk_id") or "",
                "prev_chunk_id": "",
                "next_chunk_id": "",
                "chapter_number": top_fields.get("chapter_number", ""),
                "clause_number": top_fields.get("clause_number", ""),
                "subclause_number": top_fields.get("subclause_number", ""),
                "applicable_scope": top_fields.get("applicable_scope", "全部"),
                "normative_level": top_fields.get("normative_level", "neutral"),
                "capital_tool_level": top_fields.get("capital_tool_level", ""),
                "table_name": top_fields.get("table_name", ""),
                "table_section_name": top_fields.get("table_section_name", ""),
                "sheet_name": top_fields.get("sheet_name", ""),
                "glossary_term": top_fields.get("glossary_term", ""),
                "keywords": meta.get("keywords", []),
                "evidence_snippet": raw.get("evidence_snippet", ""),
                "content_raw": raw.get("content_raw", ""),
                "sub_chunks": raw.get("sub_chunks", []),
                "metadata": {k: v for k, v in meta.items() if k not in TOP_META_KEYS},
            }
            chunks_for_db.append(chunk_dict)

    return chunks_for_db


# ============================================================
# 导入运行器
# ============================================================

class MigrationRunner:
    """数据迁移运行器 — 从 JSON 文件导入到 SQLite 数据库"""

    def __init__(self, db_path: str = "retrieval.db"):
        self.db_path = db_path
        self.db: Optional[RetrievalDB] = None

    def run(self, data_dir: str = "regulatory_docs") -> RetrievalDB:
        """
        执行完整迁移流程：
        1. 创建/打开数据库（含建表）
        2. 扫描数据目录，配对 summary + chunks 文件
        3. 导入文档元数据
        4. 批量导入 chunk 数据
        5. 自动补全 prev/next 链接
        """
        print("=" * 60)
        print("  Migration — 法规数据导入工具")
        print("=" * 60)

        # ── 第 1 步：初始化数据库 ──
        print(f"\n[1/4] 初始化数据库: {self.db_path}")
        self.db = RetrievalDB(self.db_path).open()

        # ── 第 2 步：扫描数据文件 ──
        print(f"\n[2/4] 扫描数据目录: {data_dir}")
        doc_groups = scan_regulatory_docs(data_dir)
        print(f"  发现 {len(doc_groups)} 个文档（含 summary + chunks）")

        if not doc_groups:
            print("  [错误] 未发现任何可导入的文档数据")
            self.db.close()
            sys.exit(1)

        # ── 第 3 步：导入文档元数据 ──
        print(f"\n[3/4] 导入文档元数据 ...")
        total_chunks = 0

        for i, (doc_id, summary_path, chunks_path) in enumerate(doc_groups, 1):
            # 导入文档
            doc = load_document_from_summary(summary_path)
            # 从 chunks 文件中补充 parse_timestamp
            doc = self._enrich_document_from_chunks(doc, chunks_path)
            self.db.upsert_document(doc)

            # 导入 chunks
            chunks = load_chunks_from_jsonl(chunks_path)
            if chunks:
                self.db.insert_chunks(chunks)
                total_chunks += len(chunks)

            if i % 10 == 0 or i == len(doc_groups):
                print(f"  [{i}/{len(doc_groups)}] 已导入 {total_chunks} 条 chunks")

        # ── 第 4 步：自动补全链接 ──
        print(f"\n[4/4] 自动补全 prev/next 链接 ...")
        self.db.auto_link_chunks()

        # ── 完成统计 ──
        doc_count = len(self.db.list_documents())
        chunk_count = self.db.count_chunks()
        print()
        print("=" * 60)
        print(f"  迁移完成！")
        print(f"  文档数: {doc_count}")
        print(f"  Chunk 数: {chunk_count}")
        print(f"  数据库: {self.db_path}")
        print("=" * 60)

        return self.db

    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()

    @staticmethod
    def _enrich_document_from_chunks(doc: Dict[str, Any], chunks_path: str) -> Dict[str, Any]:
        """
        从 chunks 文件第一行中提取 parse_timestamp 等补充信息，
        补充到文档元数据中。
        """
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if first_line:
                first_chunk = json.loads(first_line)
                meta = first_chunk.get("metadata", {})
                if not doc.get("parse_timestamp"):
                    doc["parse_timestamp"] = meta.get("parse_timestamp", "")
        except (OSError, json.JSONDecodeError):
            pass
        return doc


# ============================================================
# 便捷入口函数
# ============================================================

def migrate(data_dir: str = "regulatory_docs",
            db_path: str = "retrieval.db") -> RetrievalDB:
    """
    一站式迁移入口：从 JSON 数据目录导入到 SQLite 数据库。

    参数:
        data_dir: JSON 数据目录路径
        db_path:  目标 SQLite 数据库文件路径

    返回:
        已初始化并填充数据的 RetrievalDB 实例

    示例:
        db = migrate("regulatory_docs/", db_path="retrieval.db")
        print(db.count_chunks())
        db.close()
    """
    runner = MigrationRunner(db_path=db_path)
    return runner.run(data_dir=data_dir)


def export_db(db_path: str = "retrieval.db",
              output_dir: str = "regulatory_docs_export") -> RetrievalDB:
    """
    将 SQLite 数据库导出为 JSONL + summary 格式。
    导出的目录结构与 regulatory_docs 一致，可重新导入。

    参数:
        db_path:    SQLite 数据库文件路径
        output_dir: 导出目标目录

    返回:
        RetrievalDB 实例

    示例:
        db = export_db("retrieval.db", output_dir="backup/")
        db.close()
    """
    print("=" * 60)
    print("  Migration — 数据库导出工具")
    print("=" * 60)

    db = RetrievalDB(db_path).open()
    doc_count = len(db.list_documents())
    chunk_count = db.count_chunks()
    print(f"  数据库: {db_path}")
    print(f"  文档数: {doc_count}, Chunk 数: {chunk_count}")

    print(f"\n  导出到: {output_dir}")
    db.export_to_jsonl(output_dir)

    print()
    print("=" * 60)
    print(f"  导出完成！")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    return db


def export_tables(db_path: str = "retrieval.db",
                  output_dir: str = "tables_export") -> RetrievalDB:
    """
    将两张表直接导出为两个文件：
      - documents.json  — documents 表全部数据（JSON 数组）
      - chunks.jsonl    — chunks 表全部数据（JSONL）

    示例:
        db = export_tables("retrieval.db", output_dir="tables/")
        db.close()
    """
    print("=" * 60)
    print("  Migration — 两张表导出")
    print("=" * 60)

    db = RetrievalDB(db_path).open()
    doc_count = len(db.list_documents())
    chunk_count = db.count_chunks()
    print(f"  数据库: {db_path}")
    print(f"  documents: {doc_count} 行, chunks: {chunk_count} 行")

    print(f"\n  导出到: {output_dir}/")
    db.export_tables(output_dir)

    print()
    print("=" * 60)
    print(f"  导出完成：documents.json + chunks.jsonl")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    return db


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migration — 法规数据 导入/导出 工具"
    )
    sub = parser.add_subparsers(dest="command", help="操作命令")

    # ── import 子命令 ──
    p_import = sub.add_parser("import", help="从 JSON 文件导入到数据库")
    p_import.add_argument("--data-dir", "-d", default="regulatory_docs",
                          help="JSON 数据目录（默认: regulatory_docs）")
    p_import.add_argument("--db-path", "-o", default="retrieval.db",
                          help="数据库文件路径（默认: retrieval.db）")

    # ── export 子命令（原始格式，按文档拆分） ──
    p_export = sub.add_parser("export", help="导出为原始 JSON 格式（按文档拆分）")
    p_export.add_argument("--db-path", "-i", default="retrieval.db",
                          help="数据库文件路径（默认: retrieval.db）")
    p_export.add_argument("--output-dir", "-o", default="regulatory_docs_export",
                          help="导出目录（默认: regulatory_docs_export）")

    # ── tables 子命令（两张表直接导出） ──
    p_tables = sub.add_parser("tables", help="导出两张表：documents.json + chunks.jsonl")
    p_tables.add_argument("--db-path", "-i", default="retrieval.db",
                          help="数据库文件路径（默认: retrieval.db）")
    p_tables.add_argument("--output-dir", "-o", default="tables_export",
                          help="导出目录（默认: tables_export）")

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.command == "export":
        db_path = str(script_dir / args.db_path) if not Path(args.db_path).is_absolute() else args.db_path
        output_dir = str(script_dir / args.output_dir) if not Path(args.output_dir).is_absolute() else args.output_dir
        db = export_db(db_path=db_path, output_dir=output_dir)
        db.close()
    elif args.command == "tables":
        db_path = str(script_dir / args.db_path) if not Path(args.db_path).is_absolute() else args.db_path
        output_dir = str(script_dir / args.output_dir) if not Path(args.output_dir).is_absolute() else args.output_dir
        db = export_tables(db_path=db_path, output_dir=output_dir)
        db.close()
    else:
        # 默认走 import（兼容旧用法：python migration.py --data-dir ...）
        data_dir = getattr(args, "data_dir", "regulatory_docs")
        db_path = getattr(args, "db_path", "retrieval.db")
        data_dir = str(script_dir / data_dir) if not Path(data_dir).is_absolute() else data_dir
        db_path = str(script_dir / db_path) if not Path(db_path).is_absolute() else db_path

        db = migrate(data_dir=data_dir, db_path=db_path)
        db.close()
