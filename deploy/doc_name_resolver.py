"""
文档名称解析器 — 别名映射 + 归一化

从知识库文档表自动构建别名注册表，将用户简写/不标准的文档名
解析为库内标准全称。同时在 Agent 层对 doc_name 做归一化处理。

别名生成策略（从每个标准文档名自动派生）：
  1. 去文件后缀 (.pdf/.docx/.xlsx)
  2. 去编号前缀 (460_xxx → xxx)
  3. 提取附件名 (含"附件"的 → 提取"附件N：xxx"部分)
  4. 去发文机关前缀 (xxx_关于印发yyy → yyy)
  5. 去下划线连接的冗余重复部分
"""

import re
from typing import Dict, List, Optional, Tuple


class DocNameResolver:
    """文档名称解析器"""

    # 文件后缀
    FILE_EXTENSIONS = re.compile(r"\.(pdf|docx|xlsx|xls|doc|pptx)$", re.IGNORECASE)

    # 编号前缀：460_xxx 或 001_xxx
    NUMERIC_PREFIX = re.compile(r"^\d{3}_")

    # 附件提取：匹配 "附件N：xxx" 或 "附件N:xxx"
    ATTACHMENT_PATTERN = re.compile(r"附件\s*([一二三四五六七八九十\d]+)\s*[：:]\s*(.+)")

    # 发文机关前缀（常见）
    ISSUER_PREFIX = re.compile(
        r"^(?:中国银保监会|中国银保监会办公厅|银保监会|银保监会办公厅|"
        r"国家金融监督管理总局|金融监管总局|财政部|财政部办公厅|"
        r"中国人民银行|人民银行)"
        r"(?:办公厅)?(?:关于)?(?:印发)?_?"
    )

    def __init__(self):
        # 别名 → 标准文档名 的映射
        self._alias_map: Dict[str, str] = {}
        # 所有标准文档名
        self._doc_names: List[str] = []

    def build_from_doc_names(self, doc_names: List[str]) -> None:
        """从文档名列表构建别名注册表"""
        self._doc_names = list(doc_names)
        self._alias_map = {}

        for doc_name in doc_names:
            aliases = self._generate_aliases(doc_name)
            for alias in aliases:
                alias_normalized = self._normalize(alias)
                # 不覆盖已有映射（先到先得，避免歧义）
                if alias_normalized not in self._alias_map:
                    self._alias_map[alias_normalized] = doc_name

    def resolve(self, user_doc_name: str) -> Optional[str]:
        """
        将用户提供的文档名解析为标准文档名

        Args:
            user_doc_name: 用户提供的文档名（可能不完整/不标准）

        Returns:
            匹配到的标准文档名，未匹配返回 None
        """
        if not user_doc_name:
            return None

        # 1. 精确匹配
        if user_doc_name in self._doc_names:
            return user_doc_name

        # 2. 归一化后查别名表
        normalized = self._normalize(user_doc_name)
        if normalized in self._alias_map:
            return self._alias_map[normalized]

        # 3. 子串匹配：用户名称是某个标准名称的子串
        for doc_name in self._doc_names:
            if normalized in self._normalize(doc_name):
                return doc_name

        # 4. 反向子串匹配：标准名称的尾部包含用户名称
        for doc_name in self._doc_names:
            doc_normalized = self._normalize(doc_name)
            # 取标准名称最后 N 个字符做匹配
            user_len = len(normalized)
            if user_len > 4 and doc_normalized.endswith(normalized):
                return doc_name

        # 5. 关键词重合匹配：用户名称的核心词都在某个标准名称中
        user_keywords = self._extract_keywords(user_doc_name)
        if user_keywords:
            best_match = None
            best_score = 0
            for doc_name in self._doc_names:
                doc_normalized = self._normalize(doc_name)
                score = sum(1 for kw in user_keywords if kw in doc_normalized)
                # 要求所有关键词都命中
                if score == len(user_keywords) and score >= 2:
                    match_ratio = score / max(len(user_keywords), 1)
                    if match_ratio > best_score:
                        best_score = match_ratio
                        best_match = doc_name
            if best_match:
                return best_match

        return None

    def _normalize(self, name: str) -> str:
        """归一化文档名：去后缀、统一标点、去空格"""
        # 去文件后缀
        name = self.FILE_EXTENSIONS.sub("", name)
        # 统一标点：全角→半角
        name = name.replace("：", ":").replace("（", "(").replace("）", ")")
        # 去空格
        name = name.replace(" ", "").strip()
        return name

    def _generate_aliases(self, doc_name: str) -> List[str]:
        """从标准文档名生成别名列表"""
        aliases = [doc_name]
        normalized = self._normalize(doc_name)

        # 1. 归一化版本
        if normalized != doc_name:
            aliases.append(normalized)

        # 2. 去编号前缀
        no_prefix = self.NUMERIC_PREFIX.sub("", normalized)
        if no_prefix != normalized:
            aliases.append(no_prefix)

        # 3. 提取附件名
        att_match = self.ATTACHMENT_PATTERN.search(doc_name)
        if att_match:
            att_name = f"附件{att_match.group(1)}：{att_match.group(2)}"
            aliases.append(att_name)
            aliases.append(self._normalize(att_name))
            # 也把附件内容名单独加入
            aliases.append(att_match.group(2))
            aliases.append(self._normalize(att_match.group(2)))

        # 4. 去发文机关前缀
        no_issuer = self.ISSUER_PREFIX.sub("", no_prefix)
        if no_issuer != no_prefix:
            aliases.append(no_issuer)

        # 5. 处理下划线分隔：取最后一个有意义的部分
        parts = no_prefix.split("_")
        if len(parts) >= 2:
            # 最后一个部分通常是附件名或主题名
            last_part = parts[-1]
            aliases.append(last_part)
            aliases.append(self._normalize(last_part))

        # 去重
        seen = set()
        unique = []
        for a in aliases:
            if a and a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    def _extract_keywords(self, name: str) -> List[str]:
        """从文档名提取关键词（用于模糊匹配）"""
        normalized = self._normalize(name)
        # 去编号前缀
        normalized = self.NUMERIC_PREFIX.sub("", normalized)
        # 按下划线分割
        parts = normalized.split("_")
        keywords = []
        for part in parts:
            part = part.strip()
            if len(part) >= 3:
                keywords.append(part)
        return keywords
