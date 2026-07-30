"""
工具平台单元测试 — M5.3 工具模块

测试用例覆盖:
  - 注册工具（注册表有记录）
  - Schema 校验（输入缺字段 → 校验失败）
  - 权限检查（无权限调用 → 拒绝）
  - 幂等检查（相同输入 → 返回缓存）
  - 超时/重试（工具执行失败 → 触发重试）
  - 降级（主工具失败 → 调用 fallback）
  - 事件日志（调用完成 → 日志有记录）
  - 内置工具（calculator, version_checker, date_parser）
"""

import pytest

from agent_platform.tools import (
    PermissionChecker,
    RetryPolicy,
    ToolExecutor,
    ToolManifest,
    ToolRegistry,
    ToolResult,
    create_default_platform,
)


# ============================================================
# 工具注册表测试
# ============================================================


class TestToolRegistry:
    """工具注册表测试"""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_and_get(self, registry):
        """注册工具 → 可查询"""
        manifest = ToolManifest(name="test_tool")
        registry.register(manifest, lambda x: x)

        assert registry.exists("test_tool")
        assert registry.get("test_tool").name == "test_tool"

    def test_register_duplicate_raises(self, registry):
        """重复注册 → 报错"""
        manifest = ToolManifest(name="test_tool")
        registry.register(manifest, lambda x: x)

        with pytest.raises(ValueError, match="已存在"):
            registry.register(manifest, lambda x: x)

    def test_unregister(self, registry):
        """注销工具"""
        manifest = ToolManifest(name="test_tool")
        registry.register(manifest, lambda x: x)

        assert registry.unregister("test_tool") is True
        assert not registry.exists("test_tool")

    def test_unregister_nonexistent(self, registry):
        """注销不存在的工具返回 False"""
        assert registry.unregister("nonexistent") is False

    def test_list_tools(self, registry):
        """列出所有工具"""
        registry.register(ToolManifest(name="tool_a"), lambda x: x)
        registry.register(ToolManifest(name="tool_b"), lambda x: x)

        names = registry.list_names()
        assert "tool_a" in names
        assert "tool_b" in names
        assert len(names) == 2

    def test_validate_input_success(self, registry):
        """Schema 校验通过"""
        manifest = ToolManifest(
            name="test_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "name": {"type": "string"},
                },
                "required": ["value"],
            },
        )
        registry.register(manifest, lambda x: x)

        valid, error = registry.validate_input(
            "test_tool", {"value": 42, "name": "test"}
        )
        assert valid is True
        assert error == ""

    def test_validate_input_missing_required(self, registry):
        """Schema 校验: 缺少必填字段 → 失败"""
        manifest = ToolManifest(
            name="test_tool",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        )
        registry.register(manifest, lambda x: x)

        valid, error = registry.validate_input("test_tool", {})
        assert valid is False
        assert "value" in error

    def test_validate_input_wrong_type(self, registry):
        """Schema 校验: 类型不匹配 → 失败"""
        manifest = ToolManifest(
            name="test_tool",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        )
        registry.register(manifest, lambda x: x)

        valid, error = registry.validate_input(
            "test_tool", {"value": "not_a_number"}
        )
        assert valid is False
        assert "类型错误" in error

    def test_validate_input_nonexistent_tool(self, registry):
        """校验不存在的工具"""
        valid, error = registry.validate_input("nonexistent", {})
        assert valid is False


# ============================================================
# 权限检查测试
# ============================================================


class TestPermissionChecker:
    """权限检查测试"""

    @pytest.fixture
    def checker(self):
        return PermissionChecker()

    def test_public_tool_all_roles(self, checker):
        """public 工具 → 所有角色可用"""
        manifest = ToolManifest(name="public_tool", permission_level="public")
        for role in ["anonymous", "authenticated", "admin"]:
            allowed, _ = checker.check(manifest, role)
            assert allowed is True

    def test_internal_tool_requires_auth(self, checker):
        """internal 工具 → 需认证"""
        manifest = ToolManifest(name="internal_tool", permission_level="internal")
        allowed, _ = checker.check(manifest, "anonymous")
        assert allowed is False

        allowed, _ = checker.check(manifest, "authenticated")
        assert allowed is True

    def test_restricted_tool_requires_admin(self, checker):
        """restricted 工具 → 仅 admin"""
        manifest = ToolManifest(name="restricted_tool", permission_level="restricted")
        allowed, _ = checker.check(manifest, "authenticated")
        assert allowed is False

        allowed, _ = checker.check(manifest, "admin")
        assert allowed is True

    def test_denied_includes_reason(self, checker):
        """拒绝时包含原因"""
        manifest = ToolManifest(name="restricted_tool", permission_level="restricted")
        allowed, reason = checker.check(manifest, "anonymous")
        assert allowed is False
        assert "权限不足" in reason


# ============================================================
# 执行器测试
# ============================================================


class TestToolExecutor:
    """工具调用执行器测试"""

    @pytest.fixture
    def setup(self):
        """创建注册表 + 执行器"""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        return registry, executor

    def test_successful_invoke(self, setup):
        """成功调用工具"""
        registry, executor = setup
        registry.register(
            ToolManifest(name="echo"),
            lambda x: {"echo": x.get("msg", "")},
        )

        result = executor.invoke("echo", {"msg": "hello"})

        assert result.success is True
        assert result.data["echo"] == "hello"
        assert result.tool_name == "echo"

    def test_invoke_nonexistent_tool(self, setup):
        """调用不存在的工具"""
        _, executor = setup
        result = executor.invoke("nonexistent", {})

        assert result.success is False
        assert "不存在" in result.error

    def test_invoke_schema_validation_failure(self, setup):
        """Schema 校验失败"""
        registry, executor = setup
        registry.register(
            ToolManifest(
                name="required_tool",
                input_schema={
                    "properties": {"val": {"type": "number"}},
                    "required": ["val"],
                },
            ),
            lambda x: x,
        )

        result = executor.invoke("required_tool", {})

        assert result.success is False
        assert "校验失败" in result.error

    def test_permission_denied(self, setup):
        """权限不足 → 拒绝"""
        registry, executor = setup
        registry.register(
            ToolManifest(name="restricted", permission_level="restricted"),
            lambda x: {"ok": True},
        )

        result = executor.invoke("restricted", {}, caller_role="anonymous")

        assert result.success is False
        assert "权限不足" in result.error

    def test_idempotent_cache(self, setup):
        """幂等检查: 相同输入 → 返回缓存"""
        registry, executor = setup
        call_count = [0]

        def handler(x):
            call_count[0] += 1
            return {"result": call_count[0]}

        registry.register(
            ToolManifest(name="idempotent_tool", idempotent=True),
            handler,
        )

        # 第一次调用
        result1 = executor.invoke("idempotent_tool", {"key": "value"})
        assert result1.success is True
        assert result1.data["result"] == 1

        # 第二次相同输入 → 返回缓存
        result2 = executor.invoke("idempotent_tool", {"key": "value"})
        assert result2.success is True
        assert result2.data["result"] == 1  # 缓存值

        # handler 只被调用一次
        assert call_count[0] == 1

    def test_retry_on_failure(self, setup):
        """工具失败 → 触发重试"""
        registry, executor = setup
        call_count = [0]

        def flaky_handler(x):
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError("模拟超时")
            return {"ok": True}

        registry.register(
            ToolManifest(
                name="flaky_tool",
                retry_policy=RetryPolicy(
                    max_retries=3,
                    retryable_errors=["timeout"],
                ),
            ),
            flaky_handler,
        )

        result = executor.invoke("flaky_tool", {})

        assert result.success is True
        assert result.retries == 2
        assert call_count[0] == 3

    def test_fallback_on_failure(self, setup):
        """主工具失败 → 调用 fallback"""
        registry, executor = setup

        # 主工具总是失败
        registry.register(
            ToolManifest(
                name="primary",
                fallback_tool="backup",
                retry_policy=RetryPolicy(max_retries=0),
            ),
            lambda x: (_ for _ in ()).throw(ValueError("总是失败")),
        )

        # 降级工具成功
        registry.register(
            ToolManifest(name="backup"),
            lambda x: {"fallback_result": True},
        )

        result = executor.invoke("primary", {})

        assert result.success is True
        assert result.fallback_used is True
        assert result.tool_name == "backup"

    def test_fallback_nonexistent(self, setup):
        """降级工具不存在 → 失败"""
        registry, executor = setup
        registry.register(
            ToolManifest(
                name="primary",
                fallback_tool="nonexistent_fallback",
                retry_policy=RetryPolicy(max_retries=0),
            ),
            lambda x: (_ for _ in ()).throw(ValueError("失败")),
        )

        result = executor.invoke("primary", {})
        assert result.success is False

    def test_event_log_recorded(self, setup):
        """调用完成 → 事件日志有记录"""
        registry, executor = setup
        registry.register(
            ToolManifest(name="logged_tool"),
            lambda x: {"ok": True},
        )

        executor.invoke("logged_tool", {"input": "data"})

        events = executor.get_event_log()
        assert len(events) == 1
        assert events[0].tool_name == "logged_tool"
        assert events[0].success is True
        assert events[0].input_data == {"input": "data"}

    def test_event_log_failed_call(self, setup):
        """失败调用也记录事件日志"""
        registry, executor = setup
        registry.register(
            ToolManifest(
                name="failing",
                retry_policy=RetryPolicy(max_retries=0),
            ),
            lambda x: (_ for _ in ()).throw(RuntimeError("执行错误")),
        )

        executor.invoke("failing", {})

        events = executor.get_event_log()
        assert len(events) == 1
        assert events[0].success is False
        assert "执行错误" in events[0].error


# ============================================================
# 内置工具测试
# ============================================================


class TestBuiltinTools:
    """内置工具测试"""

    @pytest.fixture
    def platform(self):
        return create_default_platform()

    def test_calculator_expression(self, platform):
        """计算器: 表达式计算"""
        result = platform.invoke("calculator", {"expression": "8 * 1.25"})

        assert result.success is True
        assert result.data["result"] == 10.0

    def test_calculator_compare(self, platform):
        """计算器: 合规比较"""
        result = platform.invoke(
            "calculator",
            {"actual": 9.5, "threshold": 8.0},
        )

        assert result.success is True
        assert result.data["compliant"] is True
        assert result.data["margin"] == 1.5

    def test_calculator_unsafe_expression(self, platform):
        """计算器: 不安全表达式被拒绝"""
        result = platform.invoke(
            "calculator",
            {"expression": "__import__('os').system('ls')"},
        )

        assert result.success is False

    def test_version_checker_active(self, platform):
        """版本检查器: 有效版本"""
        result = platform.invoke(
            "version_checker",
            {
                "effective_date": "2024-01-01",
                "query_date": "2024-06-01",
            },
        )

        assert result.success is True
        assert result.data["is_effective"] is True
        assert result.data["status"] == "active"

    def test_version_checker_superseded(self, platform):
        """版本检查器: 已被替代"""
        result = platform.invoke(
            "version_checker",
            {
                "effective_date": "2020-01-01",
                "superseded_date": "2024-01-01",
                "query_date": "2024-06-01",
            },
        )

        assert result.success is True
        assert result.data["is_effective"] is False
        assert result.data["status"] == "superseded"

    def test_version_checker_not_yet_effective(self, platform):
        """版本检查器: 尚未生效"""
        result = platform.invoke(
            "version_checker",
            {
                "effective_date": "2025-01-01",
                "query_date": "2024-06-01",
            },
        )

        assert result.success is True
        assert result.data["status"] == "not_yet_effective"

    def test_date_parser_parse(self, platform):
        """日期解析器: 解析日期"""
        result = platform.invoke(
            "date_parser",
            {"date_str": "2024年6月1日"},
        )

        assert result.success is True
        assert result.data["parsed_date"] == "2024-06-01"

    def test_date_parser_compare(self, platform):
        """日期解析器: 比较日期"""
        result = platform.invoke(
            "date_parser",
            {"date1": "2024-01-01", "date2": "2024-06-01"},
        )

        assert result.success is True
        assert result.data["comparison"] == "before"
        assert result.data["days_diff"] == 152

    def test_date_parser_invalid_format(self, platform):
        """日期解析器: 无效格式"""
        result = platform.invoke(
            "date_parser",
            {"date_str": "invalid-date"},
        )

        assert result.success is False

    def test_all_tools_registered(self, platform):
        """所有内置工具已注册"""
        # 通过事件日志间接验证
        for tool_name in ["calculator", "version_checker", "date_parser"]:
            result = platform.invoke(
                tool_name,
                {"expression": "1+1"} if tool_name == "calculator"
                else {"effective_date": "2024-01-01", "query_date": "2024-01-01"}
                if tool_name == "version_checker"
                else {"date_str": "2024-01-01"},
            )
            assert result.success is True, f"工具 {tool_name} 调用失败"
