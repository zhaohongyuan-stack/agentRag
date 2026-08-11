"""
引用格式化器 — 按用户确认的规范格式化引用来源

支持的引用格式:
  - 条款引用: 《文件名》第X条   例: 《商业银行资本管理办法》第43条
  - 页码引用: 《文件名》第X页   例: 《商业银行资本管理办法》第12页
  - 表格引用: 附件X 表Y 第Z行  例: 附件1 表2 第3行

引用类型判定依据:
  - chunk_type 字段（table / clause / definition / text 等）
  - metadata 中的定位信息（clause_number / page_number / attachment_number /
    table_number / row_number）
  - 已有 citation 字段的兜底解析
"""

import re
from typing import Any, Dict, List


# 表格类 chunk_type 集合
_TABLE_CHUNK_TYPES = {"table", "attachment_table", "form"}

# 条款/文本类 chunk_type 集合（默认按条款/页码处理）
_CLAUSE_CHUNK_TYPES = {
    "clause", "article", "definition", "text", "section", "",
}


class CitationFormatter:
    """
    引用格式化器

    将 EvidenceItem（对象或字典）转换为标准引用字符串，
    并支持生成编号引用列表与行内引用标记。
    """

    def format_citation(self, evidence_item: Any) -> str:
        """
        格式化单条引用

        Args:
            evidence_item: EvidenceItem 对象或字典

        Returns:
            引用字符串，如 《商业银行资本管理办法》第43条 或 附件1 表2 第3行
        """
        item = self._as_dict(evidence_item)
        chunk_type = str(item.get("chunk_type") or "").lower()
        metadata = item.get("metadata") or {}
        source_doc = item.get("source_doc") or item.get("doc_name") or ""

        # 表格类证据 → 附件X 表Y 第Z行
        if chunk_type in _TABLE_CHUNK_TYPES:
            return self._format_table_citation(item, metadata, source_doc)

        # 其余默认按条款 / 页码处理
        return self._format_clause_citation(item, metadata, source_doc)

    def format_citation_list(
        self, evidence_items: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        生成编号引用列表

        对证据项去重后按出现顺序编号。

        Args:
            evidence_items: 证据项列表（EvidenceItem 对象或字典）

        Returns:
            编号引用列表，例如:
            [{"index": 1, "citation": "《...》第43条", "source_doc": "...", "chunk_id": "..."}]
        """
        result: List[Dict[str, Any]] = []
        seen: set = set()
        index = 0

        for evidence_item in evidence_items:
            item = self._as_dict(evidence_item)
            citation = self.format_citation(evidence_item)

            # 去重：相同引用字符串只保留首次出现
            if citation in seen:
                continue
            seen.add(citation)
            index += 1

            result.append({
                "index": index,
                "citation": citation,
                "source_doc": item.get("source_doc") or item.get("doc_name") or "",
                "chunk_id": item.get("chunk_id", ""),
            })

        return result

    def format_inline_citation(self, index: int) -> str:
        """
        生成行内引用标记

        Args:
            index: 引用编号（从 1 开始）

        Returns:
            行内引用标记，如 [1]、[2]
        """
        return f"[{index}]"

    # ============================================================
    # 内部方法
    # ============================================================

    def _format_clause_citation(
        self,
        item: Dict[str, Any],
        metadata: Dict[str, Any],
        source_doc: str,
    ) -> str:
        """格式化条款 / 页码引用: 《文件名》第X条 或 《文件名》第X页"""
        # 优先从 metadata 提取条款号 / 页码
        clause_num = (
            metadata.get("clause_number")
            or metadata.get("article_number")
            or item.get("clause_number")
            or ""
        )
        page_num = (
            metadata.get("page_number")
            or item.get("page_number")
            or ""
        )

        doc_name = source_doc or metadata.get("doc_name") or "相关法规"
        existing_citation = str(item.get("citation") or "")

        # 1. 有明确条款号 → 《文件名》第X条
        if clause_num:
            return f"《{doc_name}》第{clause_num}条"

        # 2. 有明确页码 → 《文件名》第X页
        if page_num:
            return f"《{doc_name}》第{page_num}页"

        # 3. 尝试从已有 citation 字段提取条款号
        if existing_citation:
            extracted = self._extract_clause_number(existing_citation)
            if extracted:
                return f"《{doc_name}》第{extracted}条"

        # 4. 已有 citation 已包含规范格式 → 直接返回
        if existing_citation and (
            "《" in existing_citation
            or "第" in existing_citation
            or "附件" in existing_citation
        ):
            return existing_citation

        # 5. 已有 citation 为纯文本 → 补充书名号
        if existing_citation:
            return f"《{doc_name}》{existing_citation}"

        # 6. 兜底
        return f"《{doc_name}》"

    def _format_table_citation(
        self,
        item: Dict[str, Any],
        metadata: Dict[str, Any],
        source_doc: str,
    ) -> str:
        """格式化表格引用: 附件X 表Y 第Z行"""
        attachment_num = (
            metadata.get("attachment_number")
            or metadata.get("attachment_num")
            or item.get("attachment_number")
            or ""
        )
        table_num = (
            metadata.get("table_number")
            or metadata.get("table_num")
            or item.get("table_number")
            or ""
        )
        row_num = (
            metadata.get("row_number")
            or metadata.get("row_num")
            or item.get("row_number")
            or ""
        )

        existing_citation = str(item.get("citation") or "")

        # 缺失字段时尝试从已有 citation 字段解析补全
        if not attachment_num or not table_num or not row_num:
            extracted = self._extract_table_location(existing_citation)
            attachment_num = attachment_num or extracted.get("attachment", "")
            table_num = table_num or extracted.get("table", "")
            row_num = row_num or extracted.get("row", "")

        parts: List[str] = []
        if attachment_num:
            parts.append(f"附件{attachment_num}")
        if table_num:
            parts.append(f"表{table_num}")
        if row_num:
            parts.append(f"第{row_num}行")

        if parts:
            return " ".join(parts)

        # 兜底：使用已有 citation 或文档名
        if existing_citation:
            return existing_citation
        return source_doc or "表格数据"

    def _extract_clause_number(self, citation: str) -> str:
        """从引用字符串中提取条款号"""
        match = re.search(r"第\s*([一二三四五六七八九十百千\d]+)\s*条", citation)
        if match:
            return match.group(1)
        return ""

    def _extract_table_location(self, citation: str) -> Dict[str, str]:
        """从引用字符串中提取表格定位信息（附件号 / 表号 / 行号）"""
        result: Dict[str, str] = {}

        m = re.search(r"附件\s*(\d+)", citation)
        if m:
            result["attachment"] = m.group(1)

        m = re.search(r"表\s*(\d+)", citation)
        if m:
            result["table"] = m.group(1)

        m = re.search(r"第\s*(\d+)\s*行", citation)
        if m:
            result["row"] = m.group(1)

        return result

    def _as_dict(self, item: Any) -> Dict[str, Any]:
        """将证据项统一为字典（兼容对象与字典两种输入）"""
        if isinstance(item, dict):
            return item
        # dataclass / 普通对象：取 __dict__ 以包含全部字段（含 metadata）
        if hasattr(item, "__dict__"):
            return dict(item.__dict__)
        return {}
