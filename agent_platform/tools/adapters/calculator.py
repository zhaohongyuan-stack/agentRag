"""
计算器适配器 — M5.3 工具模块

提供数值计算能力，用于合规判断中的比例计算。
支持基本算术运算和百分比比较。
"""

import logging
import re
from typing import Any, Dict

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)


def calculator_handler(input_data: Dict[str, Any]) -> Any:
    """
    计算器处理函数

    支持两种模式:
      1. expression: 数学表达式字符串（如 "8 * 1.25"）
      2. compare: 比较两个数值（actual vs threshold）

    Args:
        input_data: 输入数据
            - expression: str — 数学表达式
            - 或 actual: float, threshold: float — 比较模式

    Returns:
        计算结果（float）或比较结果（dict）
    """
    if "expression" in input_data:
        expr = input_data["expression"]
        # 安全检查：只允许数字和基本运算符
        if not re.match(r"^[\d\s+\-*/().%]+$", expr):
            raise ValueError(f"不安全的表达式: {expr}")
        try:
            result = eval(expr)  # noqa: S307 — 已做安全校验
            return {"result": float(result), "expression": expr}
        except Exception as e:
            raise ValueError(f"表达式计算失败: {e}")

    elif "actual" in input_data and "threshold" in input_data:
        actual = float(input_data["actual"])
        threshold = float(input_data["threshold"])
        compliant = actual >= threshold
        return {
            "actual": actual,
            "threshold": threshold,
            "compliant": compliant,
            "margin": actual - threshold,
        }

    else:
        raise ValueError("需要 'expression' 或 'actual'+'threshold' 参数")


CALCULATOR_MANIFEST = ToolManifest(
    name="calculator",
    version="1.0.0",
    description="数值计算器，支持算术运算和合规比较",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"},
            "actual": {"type": "number", "description": "实际值"},
            "threshold": {"type": "number", "description": "阈值"},
        },
    },
    capabilities=["compute"],
    permission_level="public",
    is_read_only=True,
    timeout_ms=1000,
    idempotent=True,
    cost_level="low",
    result_trust_level="verified",
)
