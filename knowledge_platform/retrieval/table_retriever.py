"""
表格行列检索 — 从 Markdown 表格 Chunk 中提取结构化数据

功能：
- 解析 Markdown 表格文本，建立行列索引
- 按行号 / 列名取值（get_row, get_column, get_cell）
- 按关键词模糊搜索行（find_rows）
- 表格转字典列表（as_dict_list，便于 LLM 直接使用）
- 表名 → 所有数据映射（list_tables + 按名查询）

使用方式：
    tr = TableRetriever()
    tr.index(chunks)
    rows = tr.find_rows("KM1", "核心一级资本")
    data = tr.as_dict_list("KM1")
    cell = tr.get_cell("KM1", row=2, col=3)
"""

import re
from typing import List, Dict, Optional, Any


class TableRetriever:
    """表格行列检索器（Markdown 表格解析 + 行列查询）"""

    def __init__(self):
        self._tables: Dict[str, Dict[str, Any]] = {}  # table_name → {headers, rows, chunks}

    # ============================================================
    # 索引构建
    # ============================================================
    def index(self, chunks: List[Any]):
        """
        从 chunk 列表构建表格索引。
        只处理 chunk_type 为 table / table_section / excel_sheet / excel_range 的 chunk，
        或 metadata 中包含 table_name 的 chunk。
        """
        count = 0
        for chunk in chunks:
            # 统一获取属性（兼容 Chunk 对象和 dict）
            if isinstance(chunk, dict):
                ctype = chunk.get("chunk_type", "")
                # table_name 可能在顶层，也可能在 metadata 里
                table_name = chunk.get("table_name", "") or chunk.get("metadata", {}).get("table_name", "")
                text = chunk.get("content", chunk.get("text", ""))
                chunk_id = chunk.get("chunk_id", "")
            else:
                ctype = getattr(chunk, "chunk_type", "")
                table_name = getattr(chunk, "metadata", {}).get("table_name", "")
                text = getattr(chunk, "content", "")
                chunk_id = getattr(chunk, "chunk_id", "")

            if not table_name:
                continue

            if table_name not in self._tables:
                self._tables[table_name] = {"headers": [], "rows": [], "chunks": []}

            parsed = self._parse_table_text(text)
            if parsed:
                # 以第一个成功的解析为准来确定 headers
                if not self._tables[table_name]["headers"] and parsed["headers"]:
                    self._tables[table_name]["headers"] = parsed["headers"]
                # 追加行数据
                self._tables[table_name]["rows"].extend(parsed["rows"])

            self._tables[table_name]["chunks"].append(chunk_id)
            count += 1

        print(f"  [TableRetriever] 已索引 {count} 个表格 chunk（{len(self._tables)} 个独立表）")
        for name, entry in self._tables.items():
            print(f"    · {name}: {len(entry['headers'])} 列 × {len(entry['rows'])} 行")

    # ============================================================
    # Markdown 表格解析
    # ============================================================
    def _parse_table_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        解析 Markdown 表格文本。
        支持标准格式:
          | 列A | 列B | 列C |
          |-----|-----|-----|
          | v1  | v2  | v3  |
        也支持无分隔线的简单管道符格式。
        """
        lines = [l for l in text.strip().split("\n") if l.strip()]

        # 找到所有以 | 开头的行
        pipe_lines = []
        separator_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|"):
                pipe_lines.append((i, stripped))
                if re.match(r'^[\|\s\-:]+$', stripped):
                    separator_idx = i

        if len(pipe_lines) < 2:
            return None

        # 确定 header 行（分隔线上方的第一行）
        if separator_idx > 0:
            # 标准 Markdown 格式：header | separator | data
            header_line = None
            for j, (orig_idx, line) in enumerate(pipe_lines):
                if orig_idx == separator_idx and j > 0:
                    header_line = pipe_lines[j - 1][1]
                    data_start = j + 1
                    break
            if header_line is None:
                header_line = pipe_lines[0][1]
                data_start = 1
        else:
            # 无分隔线，取第一行做 header
            header_line = pipe_lines[0][1]
            data_start = 1

        headers = self._parse_cells(header_line)
        rows = [self._parse_cells(line) for _, line in pipe_lines[data_start:]]

        # 过滤掉分隔线行
        rows = [r for r in rows if not all(re.match(r'^[\-:]+$', c) for c in r if c)]

        return {"headers": headers, "rows": rows}

    @staticmethod
    def _parse_cells(line: str) -> List[str]:
        """解析单行管道符分隔的单元格"""
        cells = line.strip().split("|")
        # 去掉首尾空元素（| 开头和结尾会产生空字符串）
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        return [c.strip() for c in cells]

    # ============================================================
    # 行列检索
    # ============================================================
    def list_tables(self) -> List[str]:
        """列出所有已索引的表名"""
        return sorted(self._tables.keys())

    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """获取表的元信息（列数、行数）"""
        entry = self._tables.get(table_name)
        if not entry:
            return None
        return {
            "table_name": table_name,
            "columns": len(entry["headers"]),
            "rows": len(entry["rows"]),
            "headers": entry["headers"],
            "chunk_count": len(entry["chunks"]),
        }

    def get_headers(self, table_name: str) -> List[str]:
        """获取表头列名列表"""
        entry = self._tables.get(table_name)
        return entry["headers"][:] if entry else []

    def get_row(self, table_name: str, row_index: int) -> Optional[Dict[str, str]]:
        """
        按行号取一行，返回 {列名: 值} 字典。
        row_index 从 0 开始。
        """
        entry = self._tables.get(table_name)
        if not entry or row_index < 0 or row_index >= len(entry["rows"]):
            return None
        return self._zip_row(entry["headers"], entry["rows"][row_index])

    def get_rows(self, table_name: str, start: int = 0, end: Optional[int] = None) -> List[Dict[str, str]]:
        """取行范围 [start, end)，end 为 None 表示到最后"""
        entry = self._tables.get(table_name)
        if not entry:
            return []
        end = end or len(entry["rows"])
        return [self._zip_row(entry["headers"], r) for r in entry["rows"][start:end]]

    def get_column(self, table_name: str, col_name: str) -> List[str]:
        """按列名（模糊匹配）取整列数据"""
        entry = self._tables.get(table_name)
        if not entry:
            return []

        col_idx = self._find_col_index(entry["headers"], col_name)
        if col_idx is None:
            return []

        return [row[col_idx] if col_idx < len(row) else "" for row in entry["rows"]]

    def get_cell(self, table_name: str, row_index: int,
                 col_spec: Any) -> Optional[str]:
        """
        取单个单元格。
        col_spec 可以是列名（str，模糊匹配）或列索引（int）。
        """
        entry = self._tables.get(table_name)
        if not entry or row_index < 0 or row_index >= len(entry["rows"]):
            return None

        if isinstance(col_spec, str):
            col_idx = self._find_col_index(entry["headers"], col_spec)
        elif isinstance(col_spec, int):
            col_idx = col_spec
        else:
            return None

        if col_idx is None or col_idx >= len(entry["rows"][row_index]):
            return None
        return entry["rows"][row_index][col_idx]

    def find_rows(self, table_name: str, pattern: str,
                  col_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        按关键词搜索表格行，返回匹配行的行号和完整字典。

        参数：
          table_name: 表名
          pattern:    搜索关键词（大小写不敏感）
          col_name:   限定只搜索该列，为 None 则搜索全行

        返回：
          [{"row_index": int, "row": {header: value}, ...}, ...]
        """
        entry = self._tables.get(table_name)
        if not entry:
            return []

        col_idx = None
        if col_name:
            col_idx = self._find_col_index(entry["headers"], col_name)
            if col_idx is None:
                return []

        results = []
        pattern_lower = pattern.lower()
        for i, row in enumerate(entry["rows"]):
            if col_idx is not None:
                target = row[col_idx] if col_idx < len(row) else ""
                if pattern_lower in target.lower():
                    results.append({
                        "row_index": i,
                        "row": self._zip_row(entry["headers"], row),
                    })
            else:
                row_text = " ".join(row)
                if pattern_lower in row_text.lower():
                    results.append({
                        "row_index": i,
                        "row": self._zip_row(entry["headers"], row),
                    })
        return results

    def as_dict_list(self, table_name: str) -> List[Dict[str, str]]:
        """
        将整个表格转为字典列表（每行一个 dict）。
        这是最常用的输出格式，可直接喂给 LLM。
        """
        entry = self._tables.get(table_name)
        if not entry:
            return []
        return [self._zip_row(entry["headers"], row) for row in entry["rows"]]

    # ============================================================
    # 内部工具
    # ============================================================
    @staticmethod
    def _zip_row(headers: List[str], row: List[str]) -> Dict[str, str]:
        """将 headers + row 列表拼成字典（缺失值补空）"""
        result = {}
        for j, h in enumerate(headers):
            result[h] = row[j] if j < len(row) else ""
        return result

    @staticmethod
    def _find_col_index(headers: List[str], col_name: str) -> Optional[int]:
        """按列名模糊查找列索引（忽略大小写和前后空格）"""
        col_lower = col_name.strip().lower()
        for i, h in enumerate(headers):
            if col_lower in h.lower():
                return i
        return None
