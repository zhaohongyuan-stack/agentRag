"""
Metadata 检索器 — 基于元数据字段的过滤与查询

职责：对 Chunk/Document 的元数据字段做等值匹配、子串匹配、关键词匹配，
      不涉及文本内容搜索（内容搜索由 lexical / dense / exact 负责）。

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
    retriever = MetadataRetriever()
    retriever.index(chunks)
    results = retriever.search({
        "chunk_type": {"value": "clause"},
        "applicable_scope": {"value": "全部"},
        "clause_number": {"value": ["12", "13", "14"], "op": "in"},
    })
"""

import re
from typing import List, Dict, Optional, Any, Set


class MetadataRetriever:
    """元数据字段过滤检索器"""

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

    def __init__(self):
        self._records: List[Dict[str, Any]] = []  # 展平后的记录列表

    # ============================================================
    # 索引
    # ============================================================
    def index(self, chunks: List[Any]):
        """
        索引 chunk 列表。
        支持 Chunk 对象和 dict 两种输入。

        参数：
          chunks: Chunk 对象列表 或字典列表
        """
        self._records = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                self._records.append(self._flatten_dict(chunk))
            else:
                self._records.append(self._flatten_chunk(chunk))
        print(f"  [MetadataRetriever] 已索引 {len(self._records)} 条记录")

    def _flatten_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """将 dict 形式的 chunk 展平"""
        meta = d.get("metadata", {})
        return {
            "chunk_type": d.get("chunk_type", ""),
            "doc_id": str(d.get("doc_id", "")),
            "doc_name": d.get("doc_name", ""),
            "doc_title": d.get("doc_title", ""),
            "content": d.get("content", d.get("text", "")),
            "hierarchy_path": d.get("hierarchy_path", ""),
            **{k: meta.get(k, "") for _, (_, k) in self._FIELD_MAP.items()
               if not self._FIELD_MAP[k[0] if isinstance(k, tuple) else k][0]},
            "_raw": d,
        }

    def _flatten_chunk(self, chunk) -> Dict[str, Any]:
        """将 Chunk 对象展平"""
        meta = getattr(chunk, "metadata", {})
        return {
            "chunk_type": getattr(chunk, "chunk_type", ""),
            "doc_id": str(getattr(chunk, "doc_id", "")),
            "doc_name": getattr(chunk, "doc_name", ""),
            "doc_title": getattr(chunk, "doc_title", ""),
            "content": getattr(chunk, "content", ""),
            "hierarchy_path": getattr(chunk, "hierarchy_path", ""),
            **{field: meta.get(meta_key, "") for field, (is_attr, meta_key) in self._FIELD_MAP.items() if not is_attr},
            "_raw": chunk,
        }

    # ============================================================
    # 检索 — 主入口
    # ============================================================
    def search(self, filters: Dict[str, Any],
               limit: int = 100) -> List[Dict[str, Any]]:
        """
        按元数据字段过滤。

        参数：
          filters: 过滤条件字典，支持两种格式：
                   简单格式:  {"chunk_type": "clause", "doc_id": "400"}
                   扩展格式:  {"chunk_type": {"value": "clause", "op": "eq"},
                               "clause_number": {"value": ["12","13"], "op": "in"}}
          limit:   最多返回条数

        返回：
          符合所有条件的原始记录列表
        """
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

        return [self._records[i]["_raw"] for i in sorted(allowed)[:limit]]

    def get_allowed_indices(self, filters: Dict[str, Any]) -> Set[int]:
        """
        返回符合过滤条件的记录索引集合。
        用于传给 lexical / dense / exact 检索器做前置过滤。
        """
        results = self.search(filters, limit=len(self._records))
        return {self._records.index(r["_raw"]) if isinstance(r, dict) and "_raw" in r
                else i for i, r in enumerate(self._records)
                if r.get("_raw") in results}

    # ============================================================
    # 单条匹配
    # ============================================================
    def _match(self, record: Dict[str, Any],
               field: str, value: Any, op: str) -> bool:
        """判断单条记录是否匹配过滤条件"""
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
        values = set()
        for record in self._records:
            val = record.get(field, "")
            if val:
                values.add(str(val))
        return sorted(values)

    def count_by_field(self, field: str) -> Dict[str, int]:
        """按字段值分组计数"""
        counts: Dict[str, int] = {}
        for record in self._records:
            val = str(record.get(field, "(空)"))
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @property
    def record_count(self) -> int:
        return len(self._records)
