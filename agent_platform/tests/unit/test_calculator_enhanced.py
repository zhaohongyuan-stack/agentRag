"""
增强计算器单元测试 — M5.4

测试覆盖:
  - 11 种运算（add/subtract/multiply/divide/percentage/sum/average/compare/expression/percentage_change/ratio）
  - 错误处理（除零、空数组、total=0 等）
  - 向后兼容（无 operation 字段时的 expression/compare 模式）
  - Manifest 属性验证
"""

import pytest

from agent_platform.tools.adapters.calculator import (
    CALCULATOR_MANIFEST,
    calculator_handler,
)


# ============================================================
# 基本运算测试
# ============================================================

class TestBasicArithmetic:
    """基本算术运算"""

    def test_add(self):
        r = calculator_handler({"operation": "add", "a": 10, "b": 20})
        assert r["success"] is True
        assert r["result"] == 30.0
        assert r["operation"] == "add"

    def test_subtract(self):
        r = calculator_handler({"operation": "subtract", "a": 50, "b": 18})
        assert r["success"] is True
        assert r["result"] == 32.0

    def test_multiply(self):
        r = calculator_handler({"operation": "multiply", "a": 7, "b": 8})
        assert r["success"] is True
        assert r["result"] == 56.0

    def test_divide(self):
        r = calculator_handler({"operation": "divide", "a": 100, "b": 4})
        assert r["success"] is True
        assert r["result"] == 25.0

    def test_divide_by_zero(self):
        r = calculator_handler({"operation": "divide", "a": 10, "b": 0})
        assert r["success"] is False
        assert "error" in r

    def test_add_with_floats(self):
        r = calculator_handler({"operation": "add", "a": 3.14, "b": 2.86})
        assert r["success"] is True
        assert abs(r["result"] - 6.0) < 0.001


# ============================================================
# 百分比和比率
# ============================================================

class TestPercentageAndRatio:
    """百分比和比率运算"""

    def test_percentage(self):
        r = calculator_handler({"operation": "percentage", "part": 25, "total": 200})
        assert r["success"] is True
        assert r["result"] == 12.5

    def test_percentage_zero_total(self):
        r = calculator_handler({"operation": "percentage", "part": 10, "total": 0})
        assert r["success"] is False

    def test_percentage_change(self):
        r = calculator_handler({"operation": "percentage_change", "old_value": 100, "new_value": 120})
        assert r["success"] is True
        assert r["result"] == 20.0

    def test_percentage_change_negative(self):
        r = calculator_handler({"operation": "percentage_change", "old_value": 100, "new_value": 80})
        assert r["success"] is True
        assert r["result"] == -20.0

    def test_ratio(self):
        r = calculator_handler({"operation": "ratio", "a": 3, "b": 4})
        assert r["success"] is True
        assert r["result"] == 0.75

    def test_ratio_zero_denominator(self):
        r = calculator_handler({"operation": "ratio", "a": 5, "b": 0})
        assert r["success"] is False


# ============================================================
# 聚合运算
# ============================================================

class TestAggregation:
    """求和与均值"""

    def test_sum(self):
        r = calculator_handler({"operation": "sum", "numbers": [1, 2, 3, 4, 5]})
        assert r["success"] is True
        assert r["result"] == 15.0
        assert r["count"] == 5

    def test_sum_empty(self):
        r = calculator_handler({"operation": "sum", "numbers": []})
        assert r["success"] is True
        assert r["result"] == 0.0

    def test_average(self):
        r = calculator_handler({"operation": "average", "numbers": [10, 20, 30]})
        assert r["success"] is True
        assert r["result"] == 20.0

    def test_average_empty(self):
        r = calculator_handler({"operation": "average", "numbers": []})
        assert r["success"] is False


# ============================================================
# 比较与表达式
# ============================================================

class TestCompareAndExpression:
    """合规比较与表达式求值"""

    def test_compare_compliant(self):
        r = calculator_handler({"operation": "compare", "actual": 12.5, "threshold": 10.0})
        assert r["compliant"] is True
        assert r["margin"] == 2.5

    def test_compare_non_compliant(self):
        r = calculator_handler({"operation": "compare", "actual": 8.0, "threshold": 10.0})
        assert r["compliant"] is False
        assert r["margin"] == -2.0

    def test_expression(self):
        r = calculator_handler({"operation": "expression", "expression": "2 * (3 + 4)"})
        assert r["success"] is True
        assert r["result"] == 14.0

    def test_expression_unsafe(self):
        with pytest.raises(ValueError):
            calculator_handler({"operation": "expression", "expression": "__import__('os').system('ls')"})


# ============================================================
# 向后兼容
# ============================================================

class TestBackwardCompatibility:
    """无 operation 字段时的向后兼容"""

    def test_compat_expression(self):
        """无 operation + 有 expression → expression 模式"""
        r = calculator_handler({"expression": "8 * 1.25"})
        assert r["success"] is True
        assert r["result"] == 10.0

    def test_compat_compare(self):
        """无 operation + 有 actual+threshold → compare 模式"""
        r = calculator_handler({"actual": 12.0, "threshold": 10.0})
        assert r["compliant"] is True


# ============================================================
# Manifest 验证
# ============================================================

class TestManifest:
    """工具清单验证"""

    def test_manifest_name(self):
        assert CALCULATOR_MANIFEST.name == "calculator"

    def test_manifest_capabilities(self):
        assert "compute" in CALCULATOR_MANIFEST.capabilities

    def test_manifest_read_only(self):
        assert CALCULATOR_MANIFEST.is_read_only is True

    def test_manifest_has_operation_in_schema(self):
        props = CALCULATOR_MANIFEST.input_schema.get("properties", {})
        assert "operation" in props
        enum_values = props["operation"].get("enum", [])
        assert "add" in enum_values
        assert "divide" in enum_values
        assert "percentage" in enum_values
        assert len(enum_values) == 11
