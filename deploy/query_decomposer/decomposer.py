"""
查询分解器 — 复合查询拆分

识别多选题等复合型查询，自动拆分为独立子问题。
每个子问题可独立检索，避免多主题语义稀释和误过滤。

支持的复合查询模式：
  - 多选题：含"下列哪项表述正确"+ A/B/C/D 选项
  - 多问句：含多个独立问号的问题
  - 并列查询：用"另外"、"还有"等连接的多个查询
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubQuery:
    """分解后的子问题"""

    sub_query_id: str
    text: str
    option_label: Optional[str] = None  # A/B/C/D
    source_span: Optional[str] = None  # 原文中的对应片段

    def to_dict(self) -> dict:
        return {
            "sub_query_id": self.sub_query_id,
            "text": self.text,
            "option_label": self.option_label,
            "source_span": self.source_span,
        }


class QueryDecomposer:
    """
    查询分解器

    识别复合型问题（如多选题），自动拆分为独立子问题。
    每个子问题可独立检索、独立验证证据充分性。
    """

    # 多选题识别模式
    MULTI_CHOICE_PATTERNS = [
        re.compile(r"下列哪[项些]表述?正确"),
        re.compile(r"下列哪[项些]"),
        re.compile(r"以下哪[项些]"),
        re.compile(r"哪些表述?正确"),
        re.compile(r"哪[项些]是(?:正确|错误)的"),
    ]

    # 选项分割：A、 B、 C、 D、 或 A. B. C. D. 或 A） B）
    OPTION_PATTERN = re.compile(
        r"([A-D])[、.．）)]\s*([^A-D]+?)(?=[A-D][、.．）)]|$)"
    )

    # 多问句分割
    MULTI_QUESTION_SPLIT = re.compile(r"[？?。；;]\s*(?=[^A-D]*[？?])")

    # 连接词分割
    CONJUNCTION_PATTERN = re.compile(r"(?:另外|还有|此外|同时|以及|并且)[，,]?\s*")

    def is_decomposable(self, query: str) -> bool:
        """判断是否为可分解的复合查询"""
        if not query or len(query) < 20:
            return False

        # 检查多选题模式
        has_multi_choice = any(p.search(query) for p in self.MULTI_CHOICE_PATTERNS)
        options = self.OPTION_PATTERN.findall(query)
        has_multiple_options = len(options) >= 2

        if has_multi_choice and has_multiple_options:
            return True

        # 检查多问句模式（至少2个问号且距离较远）
        question_marks = [m.start() for m in re.finditer(r"[？?]", query)]
        if len(question_marks) >= 2:
            # 问号间距大于15字符，认为是独立问题
            for i in range(1, len(question_marks)):
                if question_marks[i] - question_marks[i - 1] > 15:
                    return True

        return False

    def decompose(self, query: str) -> List[SubQuery]:
        """
        将复合查询分解为子问题列表

        Args:
            query: 用户原始问题

        Returns:
            子问题列表，空列表表示无需分解
        """
        if not self.is_decomposable(query):
            return []

        # 优先尝试多选题分解
        sub_queries = self._decompose_multi_choice(query)
        if sub_queries:
            return sub_queries

        # 尝试多问句分解
        sub_queries = self._decompose_multi_question(query)
        if sub_queries:
            return sub_queries

        return []

    def _decompose_multi_choice(self, query: str) -> List[SubQuery]:
        """多选题分解：按 A/B/C/D 选项拆分"""
        options = self.OPTION_PATTERN.findall(query)
        if len(options) < 2:
            return []

        # 提取题干（选项之前的部分）
        first_option_match = re.search(
            r"[A-D][、.．）)]", query
        )
        preamble = ""
        if first_option_match:
            preamble = query[: first_option_match.start()].strip()
            # 去掉"下列哪项表述正确？"等引导语，保留文档引用等上下文
            preamble = re.sub(
                r"下列哪[项些]表述?正确[？?]?", "", preamble
            )
            preamble = re.sub(r"[，,。；;]$", "", preamble).strip()

        sub_queries = []
        for label, text in options:
            text = text.strip().rstrip("。.；;？?")
            if not text or len(text) < 5:
                continue

            # 将题干上下文拼接到每个子问题前面
            # 例如题干提到《寿险合同负债评估折现率曲线》，每个选项都应带上
            full_text = text
            if preamble:
                # 如果题干有书名号内容，拼接到子问题前
                doc_refs = re.findall(r"《[^》]+》", preamble)
                if doc_refs:
                    full_text = " ".join(doc_refs) + " " + text

            sub_queries.append(
                SubQuery(
                    sub_query_id=f"sq_{label}",
                    text=full_text,
                    option_label=label,
                    source_span=text,
                )
            )

        return sub_queries

    def _decompose_multi_question(self, query: str) -> List[SubQuery]:
        """多问句分解：按问号分割"""
        # 按问号分割
        parts = re.split(r"[？?]", query)
        parts = [p.strip().rstrip("。.；;") for p in parts if p.strip()]

        if len(parts) < 2:
            return []

        # 用连接词进一步分割
        expanded_parts = []
        for part in parts:
            sub_parts = self.CONJUNCTION_PATTERN.split(part)
            expanded_parts.extend(
                [sp.strip().rstrip("。.；;") for sp in sub_parts if sp.strip()]
            )

        sub_queries = []
        for i, text in enumerate(expanded_parts):
            if len(text) < 5:
                continue
            sub_queries.append(
                SubQuery(
                    sub_query_id=f"sq_{i+1}",
                    text=text,
                    option_label=None,
                    source_span=text,
                )
            )

        return sub_queries if len(sub_queries) >= 2 else []
