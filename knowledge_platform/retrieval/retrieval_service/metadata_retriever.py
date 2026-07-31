"""
Metadata 检索器 — 基于元数据字段的过滤与查询

职责：对 Chunk/Document 的元数据字段做等值匹配、子串匹配、关键词匹配，
      不涉及文本内容搜索（内容搜索由 lexical / dense / exact 负责）。

⚠️ 第三阶段改造（2026-07-31）：
  - DB 驱动模式：传入 RetrievalDB 后，所有过滤查询走 SQL WHERE，不在内存中保存 _records
  - 内存占用：O(1)（仅持有 DB 引用 + chunk_ids 列表）
  - 降级模式：未传入 DB 时，自动回退到内存展平过滤（向后兼容测试场景）
  - search() 返回 chunk_id 列表（List[str]）
  - get_allowed_ids() 返回 Set[str]（chunk_id 集合）
  - 原文统一由 ChunkStore 管理

支持过滤字段（来自 chunk_json 约定 v1.2）：
  Chunk 直接属性:  chunk_type, doc_id, doc_name, doc_title
  metadata 字段:   parser_type, attachment_no, applicable_scope,
                   normative_level, table_name, sheet_name,
                   table_section_name, parent_chunk_id,
                   chapter_number, clause_number, subclause_number,
                   capital_tool_level, glossary_term, keywords

支持操作符（op 参数）:
  - "eq" (默认): 精确匹配      chunk_type = "clause"
  - "in":        列表包含      chunk_type in ["clause", "subclause"]
  - "contains":  子串包含      doc_name contains "资本"
  - "regex":     正则匹配      clause_number ~ "12|13|14"
  - "gt"/"gte"/"lt"/"lte": 数值比较  chapter_number > 3

使用方式：
    # DB 驱动模式（推荐）
    retriever = MetadataRetriever(db)
    chunk_ids = retriever.search({
        "chunk_type": {"value": "clause"},
        "applicable_scope": {"value": "全部"},
        "clause_number": {"value": ["12", "13", "14"], "op": "in"},
    })
    # → ["chunk-001", "chunk-002", ...]  — 返回 chunk_id 列表

    # 降级模式（无 DB，用于测试）
    retriever = MetadataRetriever()
    retriever.index(chunks)
    chunk_ids = retriever.search({"chunk_type": "clause"})
"""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any, Set


class MetadataRetriever:
    """元数据字段过滤检索器（DB 驱动 / 内存降级双模式）"""

    # 字段映射：外部过滤名 → (是否从 Chunk 属性直接取, metadata 中的 key)
    _FIELD_MAP: Dict[str, tuple] = {
        # === Chunk 直接属性 ===
        "chunk_type":  (True,  "chunk_type"),
        "doc_id":      (True,  "doc_id"),
        "doc_name":    (True,  "doc_name"),
        "doc_title":   (True,  "doc_title"),
        # === metadata 字典字段 ===
        "parser_type":       (False, "parser_type"),
        "attachment_no":     (False, "attachment_no"),
        "applicable_scope":  (False, "applicable_scope"),
        "normative_level":   (False, "normative_level"),
        "table_name":        (False, "table_name"),
        "sheet_name":        (False, "sheet_name"),
        "table_section_name": (False, "table_section_name"),
        "parent_chunk_id":   (False, "parent_chunk_id"),
        "chapter_number":    (False, "chapter_number"),
        "clause_number":     (False, "clause_number"),
        "subclause_number":  (False, "subclause_number"),
        "capital_tool_level": (False, "capital_tool_level"),
        "glossary_term":     (False, "glossary_term"),
        "keywords":          (False, "keywords"),
        # 额外通用字段
        "content":             (True,  "content"),
        "hierarchy_path":   (True,  "hierarchy_path"),
    }

    def __init__(self, db=None):
        """
        参数：
          db: RetrievalDB 实例。
              传入时使用 DB 驱动模式（SQL WHERE，不在内存保存 _records）。
              不传时使用内存降级模式（向后兼容测试场景）。
        """
        self._db = db
        self._db_mode = db is not None
        # DB 模式：仅存 chunk_ids（下标兼容），不存 _records
        # 降级模式：存 _records（展平后的元数据列表）
        self._records: List[Dict[str, Any]] = []
        self._chunk_ids: List[str] = []

    # ============================================================
    # 索引
    # ============================================================
    def index(self, chunks: List[Any]):
        """
        索引 chunk 列表。

        DB 模式：仅记录 chunk_ids（数据已在 insert_chunks 时写入 SQLite）。
        降级模式：展平到 _records 列表（内存过滤用）。

        参数：
          chunks: Chunk 对象列表 或字典列表
        """
        if self._db_mode:
            # DB 驱动模式：数据已在 DB 中，仅记录 chunk_ids 用于下标兼容
            self._chunk_ids = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    self._chunk_ids.append(chunk.get("chunk_id", ""))
                else:
                    self._chunk_ids.append(getattr(chunk, "chunk_id", ""))
            print(f"  [MetadataRetriever] DB 驱动模式，已记录 {len(self._chunk_ids)} 个 chunk_ids")
        else:
            # 降级模式：展平到内存
            self._records = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    self._records.append(self._flatten_dict(chunk))
                else:
                    self._records.append(self._flatten_chunk(chunk))
            print(f"  [MetadataRetriever] 降级模式（内存），已索引 {len(self._records)} 条记录")

    def _flatten_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """将 dict 形式的 chunk 展平（降级模式用）"""
        meta = d.get("metadata", {})
        return {
            "chunk_id": d.get("chunk_id", ""),
            "chunk_type": d.get("chunk_type", ""),
            "doc_id": str(d.get("doc_id", "")),
            "doc_name": d.get("doc_name", ""),
            "doc_title": d.get("doc_title", ""),
            "content": d.get("content", d.get("text", "")),
            "hierarchy_path": d.get("hierarchy_path", ""),
            **{k: meta.get(k, "") for _, (_, k) in self._FIELD_MAP.items()
               if not self._FIELD_MAP[k[0] if isinstance(k, tuple) else k][0]},
        }

    def _flatten_chunk(self, chunk) -> Dict[str, Any]:
        """将 Chunk 对象展平（降级模式用）"""
        meta = getattr(chunk, "metadata", {})
        return {
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "chunk_type": getattr(chunk, "chunk_type", ""),
            "doc_id": str(getattr(chunk, "doc_id", "")),
            "doc_name": getattr(chunk, "doc_name", ""),
            "doc_title": getattr(chunk, "doc_title", ""),
            "content": getattr(chunk, "content", ""),
            "hierarchy_path": getattr(chunk, "hierarchy_path", ""),
            **{field: meta.get(meta_key, "") for field, (is_attr, meta_key) in self._FIELD_MAP.items() if not is_attr},
        }

    # ============================================================
    # 检索 — 主入口
    # ============================================================
    def search(self, filters: Dict[str, Any],
               limit: int = 100) -> List[str]:
        """
        按元数据字段过滤。

        参数：
          filters: 过滤条件字典，支持两种格式：
                   简单格式:  {"chunk_type": "clause", "doc_id": "400"}
                   扩展格式:  {"chunk_type": {"value": "clause", "op": "eq"},
                               "clause_number": {"value": ["12","13"], "op": "in"}}
          limit:   最多返回条数

        返回：
          符合所有条件的 chunk_id 列表（List[str]）
        """
        if self._db_mode:
            # DB 驱动模式：委托给 SQL WHERE
            return self._db.search_by_filters(filters, limit=limit)

        # 降级模式：内存过滤
        allowed = set(range(len(self._records)))

        for field, condition in filters.items():
            if field not in self._FIELD_MAP:
                continue  # 未知字段不过滤

            # 解析条件格式
            if isinstance(condition, dict):
                value = condition.get("value")
                op = condition.get("op", "eq")
            else:
                value = condition
                op = "eq"

            if value is None or value == "":
                continue

            filtered = set()
            for idx in allowed:
                record = self._records[idx]
                if self._match(record, field, value, op):
                    filtered.add(idx)

            allowed = filtered
            if not allowed:
                break

        # 返回 chunk_id 列表
        return [self._records[i].get("chunk_id", "") for i in sorted(allowed)[:limit]]

    def get_allowed_ids(self, filters: Dict[str, Any]) -> Set[str]:
        """
        返回符合过滤条件的 chunk_id 集合。
        用于传给 lexical / dense 检索器做前置过滤。
        """
        return set(self.search(filters, limit=999999))

    def get_allowed_indices(self, filters: Dict[str, Any],
                            chunk_id_list: Optional[List[str]] = None) -> Set[int]:
        """
        返回符合过滤条件的文档下标集合。
        需要传入 chunk_id_list（来自 Lexical/Dense 的 _chunk_ids）做映射。

        参数：
          filters:       过滤条件
          chunk_id_list: chunk_id 有序列表（与检索器内部下标对齐）
        """
        if chunk_id_list is None:
            return set()
        allowed_ids = self.get_allowed_ids(filters)
        return {i for i, cid in enumerate(chunk_id_list) if cid in allowed_ids}

    # ============================================================
    # 单条匹配（降级模式用）
    # ============================================================
    def _match(self, record: Dict[str, Any],
               field: str, value: Any, op: str) -> bool:
        """判断单条记录是否匹配过滤条件（降级模式内存过滤用）"""
        is_attr, meta_key = self._FIELD_MAP.get(field, (False, field))
        actual = record.get(field, "")

        if op == "eq":
            return self._match_eq(actual, value, field)
        elif op == "in":
            if not isinstance(value, list):
                value = [value]
            return any(self._match_eq(actual, v, field) for v in value)
        elif op == "contains":
            if field == "keywords":
                kw_text = " ".join(actual) if isinstance(actual, list) else str(actual)
                if isinstance(value, list):
                    return any(k in kw_text for k in value)
                return str(value) in kw_text
            return str(value).lower() in str(actual).lower()
        elif op == "regex":
            try:
                return bool(re.search(str(value), str(actual)))
            except re.error:
                return False
        elif op in ("gt", "gte", "lt", "lte"):
            return self._match_numeric(actual, value, op)
        elif op == "prefix":
            return str(actual).lower().startswith(str(value).lower())
        elif op == "suffix":
            return str(actual).lower().endswith(str(value).lower())
        return False

    def _match_eq(self, actual: Any, value: Any, field: str) -> bool:
        """等值匹配（对 doc_name/doc_title 做子串匹配，其余精确）"""
        if field in ("doc_name", "doc_title"):
            return str(value).lower() in str(actual).lower()
        if field == "keywords":
            if isinstance(actual, list):
                return str(value) in " ".join(actual)
            return str(value) in str(actual)
        return str(actual) == str(value)

    @staticmethod
    def _match_numeric(actual: Any, value: Any, op: str) -> bool:
        """数值比较"""
        try:
            a = float(actual)
            v = float(value)
            if op == "gt":  return a > v
            if op == "gte": return a >= v
            if op == "lt":  return a < v
            if op == "lte": return a <= v
        except (ValueError, TypeError):
            pass
        return False

    # ============================================================
    # 便捷方法
    # ============================================================
    def list_field_values(self, field: str) -> List[str]:
        """列出某个字段的所有唯一值（用于构建过滤选项 UI）"""
        if self._db_mode:
            return self._db.list_field_values_db(field)

        values = set()
        for record in self._records:
            val = record.get(field, "")
            if val:
                values.add(str(val))
        return sorted(values)

    def count_by_field(self, field: str) -> Dict[str, int]:
        """按字段值分组计数"""
        if self._db_mode:
            return self._db.count_by_field_db(field)

        counts: Dict[str, int] = {}
        for record in self._records:
            val = str(record.get(field, "(空)"))
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @property
    def record_count(self) -> int:
        if self._db_mode:
            return len(self._chunk_ids)
        return len(self._records)

    @property
    def is_db_mode(self) -> bool:
        """是否为 DB 驱动模式"""
        return self._db_mode

    # ============================================================
    # 持久化
    # ============================================================
    def save_index(self, path: str) -> None:
        """
        持久化到 pickle 文件。

        DB 模式：仅保存 chunk_ids（轻量，不含 _records）。
        降级模式：保存 _records（含展平后的元数据）。
        """
        if self._db_mode:
            data = {
                "_db_mode": True,
                "_chunk_ids": self._chunk_ids,
            }
        else:
            data = {
                "_db_mode": False,
                "_records": self._records,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, path: str) -> bool:
        """
        从 pickle 文件加载，成功返回 True。

        注意：加载后如果当前实例是 DB 模式但缓存是降级模式，
        会自动切换到降级模式（保持数据一致性）。
        """
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)

        # 兼容旧格式（无 _db_mode 字段时按降级模式处理）
        self._db_mode = data.get("_db_mode", False)
        if self._db_mode:
            self._chunk_ids = data.get("_chunk_ids", [])
        else:
            self._records = data.get("_records", data.get("records", []))
            # 旧格式兼容
            if not self._records:
                self._records = []

        return True
