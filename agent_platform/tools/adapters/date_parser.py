"""
日期解析器适配器 — M5.3 工具模块

解析和比较日期，支持多种日期格式。
"""

import logging
from datetime import datetime
from typing import Any, Dict

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y年%m月%d日",
    "%Y%m%d",
    "%d/%m/%Y",
]


def date_parser_handler(input_data: Dict[str, Any]) -> Any:
    """
    日期解析处理函数

    支持两种模式:
      1. parse: 解析日期字符串
      2. compare: 比较两个日期

    Args:
        input_data:
            - date_str: str — 日期字符串
            - 或 date1: str, date2: str — 比较模式

    Returns:
        解析结果或比较结果
    """
    if "date_str" in input_data:
        date_str = input_data["date_str"]
        parsed = None
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

        if parsed is None:
            raise ValueError(f"无法解析日期: {date_str}")

        return {
            "parsed_date": parsed.strftime("%Y-%m-%d"),
            "original": date_str,
            "format": fmt,
        }

    elif "date1" in input_data and "date2" in input_data:
        d1 = _parse_date(input_data["date1"])
        d2 = _parse_date(input_data["date2"])

        if d1 is None or d2 is None:
            raise ValueError("日期解析失败")

        return {
            "date1": d1.strftime("%Y-%m-%d"),
            "date2": d2.strftime("%Y-%m-%d"),
            "comparison": "before" if d1 < d2 else "after" if d1 > d2 else "equal",
            "days_diff": abs((d1 - d2).days),
        }

    else:
        raise ValueError("需要 'date_str' 或 'date1'+'date2' 参数")


def _parse_date(date_str: str):
    """尝试多种格式解析日期"""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


DATE_PARSER_MANIFEST = ToolManifest(
    name="date_parser",
    version="1.0.0",
    description="日期解析和比较工具",
    input_schema={
        "type": "object",
        "properties": {
            "date_str": {"type": "string", "description": "日期字符串"},
            "date1": {"type": "string", "description": "第一个日期"},
            "date2": {"type": "string", "description": "第二个日期"},
        },
    },
    capabilities=["read_only"],
    permission_level="public",
    is_read_only=True,
    timeout_ms=1000,
    idempotent=True,
    cost_level="low",
    result_trust_level="verified",
)
