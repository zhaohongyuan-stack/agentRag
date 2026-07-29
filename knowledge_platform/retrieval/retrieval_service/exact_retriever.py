"""
Exact 检索器 — 精确匹配 / 子串包含 / 正则表达式检索

职责：对文档文本做精确/模糊匹配，不依赖模型也不依赖数据库。

匹配模式：
  - "exact":    完整字符串相等（忽略空白差异可选）
  - "contains": 子串包含（默认忽略大小写）
  - "regex":    正则表达式搜索
  - "prefix":   前缀匹配（查条款编号等）

使用方式：
    retriever = ExactRetriever()
    retriever.index(documents)
    results = retriever.search("第十二条", top_k=5, mode="contains")
    results = retriever.search(r"第[一二三四五六七八九十]+条", top_k=5, mode="regex")
"""

import re
from typing import List, Optional, Dict, Any


class ExactRetriever:
    """精确匹配检索器"""

    def __init__(self):
        self.documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._indexed = False

    # ============================================================
    # 索引
    # ============================================================
    def index(self, documents: List[str],
              metadatas: Optional[List[Dict[str, Any]]] = None):
        """构建索引（仅保存引用，不做预处理）"""
        self.documents = documents
        self._metadatas = metadatas or [{} for _ in documents]
        self._indexed = True

    # ============================================================
    # 检索
    # ============================================================
    def search(self, query: str, top_k: int = 10,
               mode: str = "contains",
               case_sensitive: bool = False,
               allowed_indices: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        精确/模糊匹配检索。

        参数：
          query:           查询文本
          top_k:           返回条数
          mode:            匹配模式 "exact" | "contains" | "regex" | "prefix"
          case_sensitive:  是否区分大小写
          allowed_indices: 允许匹配的文档索引集合

        返回：
          [{"index": int, "text": str, "score": 1.0, "match_pos": int}, ...]
        """
        if not self._indexed:
            return []

        if mode == "exact":
            return self._search_exact(query, top_k, case_sensitive, allowed_indices)
        elif mode == "contains":
            return self._search_contains(query, top_k, case_sensitive, allowed_indices)
        elif mode == "regex":
            return self._search_regex(query, top_k, case_sensitive, allowed_indices)
        elif mode == "prefix":
            return self._search_prefix(query, top_k, case_sensitive, allowed_indices)
        else:
            raise ValueError(f"不支持的模式: {mode}，可选: exact/contains/regex/prefix")

    # ============================================================
    # 各模式实现
    # ============================================================
    def _search_exact(self, query: str, top_k: int,
                      case_sensitive: bool,
                      allowed_indices: Optional[set]) -> List[Dict[str, Any]]:
        results = []
        q = query if case_sensitive else query.lower()
        for i, doc in enumerate(self.documents):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            d = doc if case_sensitive else doc.lower()
            if d.strip() == q.strip():
                results.append({
                    "index": i, "text": doc[:500],
                    "score": 1.0, "match_pos": 0,
                })
        return results[:top_k]

    def _search_contains(self, query: str, top_k: int,
                         case_sensitive: bool,
                         allowed_indices: Optional[set]) -> List[Dict[str, Any]]:
        results = []
        q = query if case_sensitive else query.lower()
        for i, doc in enumerate(self.documents):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            d = doc if case_sensitive else doc.lower()
            pos = d.find(q)
            if pos >= 0:
                # 多次出现的 query 得分更高
                count = d.count(q)
                score = min(count, 10) / 10.0  # 0.1 ~ 1.0
                results.append({
                    "index": i, "text": doc[:500],
                    "score": round(score, 4), "match_pos": pos,
                })
        # 按分数降序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _search_regex(self, query: str, top_k: int,
                      case_sensitive: bool,
                      allowed_indices: Optional[set]) -> List[Dict[str, Any]]:
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            return [{"index": -1, "text": f"正则错误: {e}", "score": 0}]

        for i, doc in enumerate(self.documents):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            matches = pattern.findall(doc)
            if matches:
                score = min(len(matches), 10) / 10.0
                results.append({
                    "index": i, "text": doc[:500],
                    "score": round(score, 4),
                    "match_pos": pattern.search(doc).start() if pattern.search(doc) else 0,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _search_prefix(self, query: str, top_k: int,
                       case_sensitive: bool,
                       allowed_indices: Optional[set]) -> List[Dict[str, Any]]:
        results = []
        q = query if case_sensitive else query.lower()
        for i, doc in enumerate(self.documents):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            # 检查每一行是否以 query 开头
            for line in doc.split("\n"):
                d = line.strip() if case_sensitive else line.strip().lower()
                if d.startswith(q):
                    results.append({
                        "index": i, "text": doc[:500],
                        "score": 1.0, "match_pos": 0,
                    })
                    break
        return results[:top_k]

    @property
    def metadatas(self) -> List[Dict[str, Any]]:
        return self._metadatas

    @property
    def doc_count(self) -> int:
        return len(self.documents)
