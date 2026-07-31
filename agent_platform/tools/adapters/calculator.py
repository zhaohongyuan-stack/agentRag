"""
计算器适配器 — M5.3 工具模块

提供数值计算能力，用于合规判断中的比例计算。
支持全部基本算术运算、百分比、求和、均值、合规比较、
表达式求值、变化率与比率计算。

通过 "operation" 参数指定运算类型；未提供 operation 时，
按 "expression" 或 "actual"+"threshold" 字段向后兼容。
"""

import logging
import re
from typing import Any, Callable, Dict, List

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)


# ============================================================
# 参数提取辅助
# ============================================================

def _require_number(input_data: Dict[str, Any], name: str) -> float:
    """从输入中提取必填数值，缺失或类型错误时抛出 ValueError。"""
    if name not in input_data:
        raise ValueError(f"缺少必填参数: '{name}'")
    try:
        return float(input_data[name])
    except (TypeError, ValueError) as e:
        raise ValueError(f"参数 '{name}' 必须是数值: {e}")


def _require_number_list(
    input_data: Dict[str, Any], name: str
) -> List[float]:
    """从输入中提取必填数值数组。"""
    if name not in input_data:
        raise ValueError(f"缺少必填参数: '{name}'")
    raw = input_data[name]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"参数 '{name}' 必须是数值数组")
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError) as e:
        raise ValueError(f"参数 '{name}' 中包含非数值元素: {e}")


def _require_string(input_data: Dict[str, Any], name: str) -> str:
    """从输入中提取必填字符串。"""
    if name not in input_data:
        raise ValueError(f"缺少必填参数: '{name}'")
    val = input_data[name]
    if not isinstance(val, str):
        raise ValueError(f"参数 '{name}' 必须是字符串")
    return val


# ============================================================
# 各运算实现
# ============================================================

def _op_add(input_data: Dict[str, Any]) -> Dict[str, Any]:
    a = _require_number(input_data, "a")
    b = _require_number(input_data, "b")
    return {
        "success": True,
        "operation": "add",
        "result": a + b,
        "operands": {"a": a, "b": b},
    }


def _op_subtract(input_data: Dict[str, Any]) -> Dict[str, Any]:
    a = _require_number(input_data, "a")
    b = _require_number(input_data, "b")
    return {
        "success": True,
        "operation": "subtract",
        "result": a - b,
        "operands": {"a": a, "b": b},
    }


def _op_multiply(input_data: Dict[str, Any]) -> Dict[str, Any]:
    a = _require_number(input_data, "a")
    b = _require_number(input_data, "b")
    return {
        "success": True,
        "operation": "multiply",
        "result": a * b,
        "operands": {"a": a, "b": b},
    }


def _op_divide(input_data: Dict[str, Any]) -> Dict[str, Any]:
    a = _require_number(input_data, "a")
    b = _require_number(input_data, "b")
    if b == 0:
        return {
            "success": False,
            "operation": "divide",
            "error": "除数不能为零 (b=0)",
            "operands": {"a": a, "b": b},
        }
    return {
        "success": True,
        "operation": "divide",
        "result": a / b,
        "operands": {"a": a, "b": b},
    }


def _op_percentage(input_data: Dict[str, Any]) -> Dict[str, Any]:
    part = _require_number(input_data, "part")
    total = _require_number(input_data, "total")
    if total == 0:
        return {
            "success": False,
            "operation": "percentage",
            "error": "总值不能为零 (total=0)",
            "part": part,
            "total": total,
        }
    return {
        "success": True,
        "operation": "percentage",
        "result": part / total * 100,
        "part": part,
        "total": total,
    }


def _op_sum(input_data: Dict[str, Any]) -> Dict[str, Any]:
    numbers = _require_number_list(input_data, "numbers")
    return {
        "success": True,
        "operation": "sum",
        "result": sum(numbers),
        "count": len(numbers),
    }


def _op_average(input_data: Dict[str, Any]) -> Dict[str, Any]:
    numbers = _require_number_list(input_data, "numbers")
    if len(numbers) == 0:
        return {
            "success": False,
            "operation": "average",
            "error": "数值数组不能为空",
            "count": 0,
        }
    return {
        "success": True,
        "operation": "average",
        "result": sum(numbers) / len(numbers),
        "count": len(numbers),
    }


def _op_compare(input_data: Dict[str, Any]) -> Dict[str, Any]:
    actual = _require_number(input_data, "actual")
    threshold = _require_number(input_data, "threshold")
    compliant = actual >= threshold
    return {
        "success": True,
        "operation": "compare",
        "actual": actual,
        "threshold": threshold,
        "compliant": compliant,
        "margin": actual - threshold,
    }


# 表达式安全校验：仅允许数字、空白与基本运算符
_EXPR_PATTERN = re.compile(r"^[\d\s+\-*/().%]+$")


def _op_expression(input_data: Dict[str, Any]) -> Dict[str, Any]:
    expr = _require_string(input_data, "expression")
    if not _EXPR_PATTERN.match(expr):
        raise ValueError(f"不安全的表达式: {expr}")
    try:
        result = eval(expr)  # noqa: S307 — 已做安全校验
    except Exception as e:
        raise ValueError(f"表达式计算失败: {e}")
    return {
        "success": True,
        "operation": "expression",
        "result": float(result),
        "expression": expr,
    }


def _op_percentage_change(input_data: Dict[str, Any]) -> Dict[str, Any]:
    old_value = _require_number(input_data, "old_value")
    new_value = _require_number(input_data, "new_value")
    if old_value == 0:
        return {
            "success": False,
            "operation": "percentage_change",
            "error": "旧值不能为零 (old_value=0)",
            "old_value": old_value,
            "new_value": new_value,
        }
    return {
        "success": True,
        "operation": "percentage_change",
        "result": (new_value - old_value) / old_value * 100,
        "old_value": old_value,
        "new_value": new_value,
    }


def _op_ratio(input_data: Dict[str, Any]) -> Dict[str, Any]:
    a = _require_number(input_data, "a")
    b = _require_number(input_data, "b")
    if b == 0:
        return {
            "success": False,
            "operation": "ratio",
            "error": "除数不能为零 (b=0)",
        }
    return {
        "success": True,
        "operation": "ratio",
        "result": a / b,
    }


# 运算分发表
_OPERATIONS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "add": _op_add,
    "subtract": _op_subtract,
    "multiply": _op_multiply,
    "divide": _op_divide,
    "percentage": _op_percentage,
    "sum": _op_sum,
    "average": _op_average,
    "compare": _op_compare,
    "expression": _op_expression,
    "percentage_change": _op_percentage_change,
    "ratio": _op_ratio,
}


def calculator_handler(input_data: Dict[str, Any]) -> Any:
    """
    计算器处理函数（统一入口）

    通过 "operation" 参数指定运算类型，支持:
      - add: 加法 (a, b)
      - subtract: 减法 (a, b)
      - multiply: 乘法 (a, b)
      - divide: 除法 (a, b)，b=0 时返回 success=False
      - percentage: 百分比 (part, total) → part/total*100
      - sum: 求和 (numbers)
      - average: 均值 (numbers)
      - compare: 合规比较 (actual, threshold)
      - expression: 表达式求值 (expression)，带安全校验
      - percentage_change: 变化率 (old_value, new_value) → (new-old)/old*100
      - ratio: 比率 (a, b) → a/b

    向后兼容:
      - 未提供 operation 但提供 expression → expression 模式
      - 未提供 operation 但提供 actual+threshold → compare 模式

    Args:
        input_data: 输入数据，需包含 operation 及对应参数

    Returns:
        统一 dict，包含 success 字段；
        成功时含 result 及运算相关字段，失败时含 error 字段。
    """
    operation = input_data.get("operation")

    # 向后兼容：未指定 operation 时按已有字段推断
    if not operation:
        if "expression" in input_data:
            operation = "expression"
        elif "actual" in input_data and "threshold" in input_data:
            operation = "compare"
        else:
            raise ValueError(
                "需要 'operation' 参数，或提供 'expression' / "
                "'actual'+'threshold' 以向后兼容"
            )

    handler_fn = _OPERATIONS.get(operation)
    if handler_fn is None:
        raise ValueError(
            f"不支持的 operation: '{operation}'，"
            f"可选值: {sorted(_OPERATIONS.keys())}"
        )

    return handler_fn(input_data)


CALCULATOR_MANIFEST = ToolManifest(
    name="calculator",
    version="1.1.0",
    description=(
        "数值计算器，支持全部基本运算、百分比、求和、均值、"
        "合规比较、表达式求值、变化率与比率"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": (
                    "运算类型；未提供时按 expression 或 "
                    "actual+threshold 向后兼容"
                ),
                "enum": [
                    "add",
                    "subtract",
                    "multiply",
                    "divide",
                    "percentage",
                    "sum",
                    "average",
                    "compare",
                    "expression",
                    "percentage_change",
                    "ratio",
                ],
            },
            "a": {
                "type": "number",
                "description": "运算数 a（add/subtract/multiply/divide/ratio）",
            },
            "b": {
                "type": "number",
                "description": "运算数 b（add/subtract/multiply/divide/ratio）",
            },
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "数值数组（sum/average）",
            },
            "part": {"type": "number", "description": "部分值（percentage）"},
            "total": {"type": "number", "description": "总值（percentage）"},
            "actual": {"type": "number", "description": "实际值（compare）"},
            "threshold": {"type": "number", "description": "阈值（compare）"},
            "expression": {
                "type": "string",
                "description": "数学表达式（expression）",
            },
            "old_value": {
                "type": "number",
                "description": "旧值（percentage_change）",
            },
            "new_value": {
                "type": "number",
                "description": "新值（percentage_change）",
            },
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
