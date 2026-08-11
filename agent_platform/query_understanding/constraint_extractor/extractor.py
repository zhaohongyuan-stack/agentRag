"""
约束条件抽取器

从用户问题中抽取查询约束：
  - time_range: 时间范围（如 2026年Q1）
  - version_status: 版本状态（如 现行有效、已废止）
  - applicable_scope: 适用范围（如 大型商业银行）
  - normative_level: 规范强度（如 不得→prohibitive, 应当→obligatory）
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryConstraints:
    """查询约束条件"""

    time_range: Optional[str] = None
    version_status: List[str] = field(default_factory=list)
    applicable_scope: Optional[str] = None
    normative_level: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "time_range": self.time_range,
            "version_status": self.version_status,
            "applicable_scope": self.applicable_scope,
            "normative_level": self.normative_level,
        }

    def is_empty(self) -> bool:
        return (
            self.time_range is None
            and not self.version_status
            and self.applicable_scope is None
            and self.normative_level is None
        )


# ============================================================
# 约束抽取规则
# ============================================================

# 时间范围
TIME_PATTERNS = [
    re.compile(r"(\d{4}年(?:\d{1,2}月?|Q[1-4]|第[一二三四]季度))"),
    re.compile(r"(\d{4}-\d{2})"),
    re.compile(r"((?:今年|去年|前年|上半年|下半年))"),
]

# 版本状态关键词映射
VERSION_STATUS_MAP = {
    "现行有效": "active",
    "现行": "active",
    "有效": "active",
    "最新": "active",
    "已废止": "repealed",
    "废止": "repealed",
    "已失效": "expired",
    "失效": "expired",
    "已修订": "superseded",
    "修订": "superseded",
    "草案": "draft",
    "征求意见稿": "draft",
}

# 适用范围关键词
SCOPE_MAP = {
    "系统重要性银行": "系统重要性银行",
    "非系统重要性银行": "非系统重要性银行",
    "国内系统重要性银行": "国内系统重要性银行",
    "全球系统重要性银行": "全球系统重要性银行",
    "大型商业银行": "大型商业银行",
    "中型银行": "中型银行",
    "小型银行": "小型银行",
    "城市商业银行": "城市商业银行",
    "农村商业银行": "农村商业银行",
    "民营银行": "民营银行",
    "外资银行": "外资银行",
    "政策性银行": "政策性银行",
    "全部": "全部",
    "所有银行": "全部",
}

# 规范强度关键词映射
NORMATIVE_LEVEL_MAP = {
    "不得": "prohibitive",
    "禁止": "prohibitive",
    "严禁": "prohibitive",
    "应当": "obligatory",
    "必须": "obligatory",
    "应": "obligatory",
    "须": "obligatory",
    "可以": "permissive",
    "可": "permissive",
    "宜": "advisory",
    "建议": "advisory",
    "鼓励": "advisory",
    "是指": "definitional",
    "定义为": "definitional",
}


class ConstraintExtractor:
    """
    约束条件抽取器

    从用户问题中抽取时间、版本、范围、规范强度等约束条件。
    """

    def __init__(self):
        pass

    def extract(self, query: str) -> QueryConstraints:
        """
        抽取查询约束

        Args:
            query: 用户原始问题

        Returns:
            QueryConstraints 对象
        """
        if not query or not query.strip():
            return QueryConstraints()

        query_stripped = query.strip()

        # 时间范围
        time_range = self._extract_time_range(query_stripped)

        # 版本状态
        version_status = self._extract_version_status(query_stripped)

        # 适用范围
        applicable_scope = self._extract_scope(query_stripped)

        # 规范强度
        normative_level = self._extract_normative_level(query_stripped)

        return QueryConstraints(
            time_range=time_range,
            version_status=version_status,
            applicable_scope=applicable_scope,
            normative_level=normative_level,
        )

    def _extract_time_range(self, query: str) -> Optional[str]:
        """抽取时间范围"""
        for pattern in TIME_PATTERNS:
            match = pattern.search(query)
            if match:
                return match.group(1)
        return None

    def _extract_version_status(self, query: str) -> List[str]:
        """抽取版本状态"""
        statuses = []
        for keyword, status in VERSION_STATUS_MAP.items():
            if keyword in query:
                if status not in statuses:
                    statuses.append(status)
        return statuses

    def _extract_scope(self, query: str) -> Optional[str]:
        """抽取适用范围"""
        for keyword, scope in SCOPE_MAP.items():
            if keyword in query:
                return scope
        return None

    def _extract_normative_level(self, query: str) -> Optional[str]:
        """抽取规范强度"""
        for keyword, level in NORMATIVE_LEVEL_MAP.items():
            if keyword in query:
                return level
        return None
