"""
预算控制器单元测试

测试用例（对应开发计划测试用例表）:
  - 分配预算: P2 路径 → max_retrieval_rounds=2
  - 消耗追踪: 1 次检索 → consumed.retrieval_rounds=1
  - 超预算: 超过 max_retrieval_rounds → 触发 STOP
  - Token 追踪: 消耗 5000 token → consumed.tokens=5000
  - 超时: 超过 total_timeout_ms → 触发 STOP（超时属硬限制，返回 STOP）

额外测试用例:
  - 各路径预算分配: P0-P4 的 max_retrieval_rounds 正确
  - 消耗改写/子任务/工具调用追踪
  - 剩余预算 get_remaining
  - 预算摘要 get_summary
  - BudgetEnforcer: CONTINUE/STOP/DOWNGRADE 信号
  - BudgetEnforcer: should_stop/should_downgrade 状态
  - 降级阈值: 80% 消耗触发 DOWNGRADE
"""

import pytest

from agent_platform.orchestration.budget_controller import (
    BUDGET_BY_PATH,
    BudgetAction,
    BudgetConsumed,
    BudgetController,
    BudgetEnforcer,
    ExecutionBudget,
)


# 各路径 max_retrieval_rounds 预期值（与 BUDGET_BY_PATH 定义对齐）
_EXPECTED_MAX_RETRIEVAL = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 5,
}


class TestBudgetModels:
    """预算数据模型测试"""

    def test_budget_action_values(self):
        """BudgetAction 枚举值正确"""
        assert BudgetAction.CONTINUE.value == "continue"
        assert BudgetAction.STOP.value == "stop"
        assert BudgetAction.DOWNGRADE.value == "downgrade"

    def test_budget_consumed_to_dict(self):
        """BudgetConsumed.to_dict 返回全部字段"""
        c = BudgetConsumed(
            retrieval_rounds=1,
            rewrites=2,
            subtasks=3,
            tool_calls=4,
            tokens=5000,
            elapsed_ms=1000,
        )
        d = c.to_dict()
        assert d == {
            "retrieval_rounds": 1,
            "rewrites": 2,
            "subtasks": 3,
            "tool_calls": 4,
            "tokens": 5000,
            "elapsed_ms": 1000,
        }

    def test_execution_budget_to_dict(self):
        """ExecutionBudget.to_dict 包含上限与消耗"""
        b = ExecutionBudget(max_retrieval_rounds=2, total_token_budget=20000)
        d = b.to_dict()
        assert d["max_retrieval_rounds"] == 2
        assert d["total_token_budget"] == 20000
        assert "consumed" in d
        assert d["consumed"]["retrieval_rounds"] == 0

    def test_budget_by_path_contains_all_paths(self):
        """BUDGET_BY_PATH 包含 P0-P4 全部路径"""
        assert set(BUDGET_BY_PATH.keys()) == {"P0", "P1", "P2", "P3", "P4"}


class TestBudgetAllocation:
    """预算分配测试"""

    def test_p2_allocation_max_retrieval_rounds(self):
        """P2 路径分配预算 max_retrieval_rounds=2"""
        bc = BudgetController("P2")
        assert bc.path_id == "P2"
        assert bc.budget.max_retrieval_rounds == 2

    @pytest.mark.parametrize(
        "path_id,expected", list(_EXPECTED_MAX_RETRIEVAL.items())
    )
    def test_all_paths_max_retrieval_rounds(self, path_id, expected):
        """各路径 P0-P4 的 max_retrieval_rounds 正确"""
        bc = BudgetController(path_id)
        assert bc.budget.max_retrieval_rounds == expected
        # 与全局预算表保持一致
        assert (
            bc.budget.max_retrieval_rounds
            == BUDGET_BY_PATH[path_id].max_retrieval_rounds
        )

    def test_p2_full_budget_fields(self):
        """P2 预算各维度上限完整正确"""
        bc = BudgetController("P2")
        b = bc.budget
        assert b.max_retrieval_rounds == 2
        assert b.max_rewrites == 2
        assert b.max_subtasks == 5
        assert b.max_tool_calls == 3
        assert b.total_timeout_ms == 5000
        assert b.total_token_budget == 20000

    def test_default_path_is_p2(self):
        """未指定路径时默认为 P2"""
        bc = BudgetController()
        assert bc.path_id == "P2"
        assert bc.budget.max_retrieval_rounds == 2

    def test_unknown_path_falls_back_to_p2(self):
        """未知路径回退到 P2 默认预算"""
        bc = BudgetController("PX_UNKNOWN")
        assert bc.path_id == "PX_UNKNOWN"
        assert (
            bc.budget.max_retrieval_rounds
            == BUDGET_BY_PATH["P2"].max_retrieval_rounds
        )

    def test_budget_consumed_initial_zero(self):
        """初始消耗记录全为零"""
        bc = BudgetController("P3")
        c = bc.budget.consumed
        assert c.retrieval_rounds == 0
        assert c.rewrites == 0
        assert c.subtasks == 0
        assert c.tool_calls == 0
        assert c.tokens == 0
        assert c.elapsed_ms == 0

    def test_allocate_resets_consumed(self):
        """allocate 重新分配时重置消耗记录"""
        bc = BudgetController("P4")
        bc.consume_retrieval_round()
        bc.consume_tokens(1000)
        assert bc.budget.consumed.retrieval_rounds == 1
        # 重新分配，消耗清零
        bc.allocate("P4")
        assert bc.budget.consumed.retrieval_rounds == 0
        assert bc.budget.consumed.tokens == 0

    def test_allocate_returns_fresh_instance(self):
        """allocate 返回全新 ExecutionBudget 实例，不与旧实例共享状态"""
        bc = BudgetController("P2")
        first = bc.budget
        bc.consume_retrieval_round()
        second = bc.allocate("P2")
        assert second is not first
        assert second.consumed.retrieval_rounds == 0

    def test_allocate_switches_path(self):
        """allocate 切换到新路径并应用对应预算"""
        bc = BudgetController("P2")
        assert bc.budget.max_retrieval_rounds == 2
        bc.allocate("P4")
        assert bc.path_id == "P4"
        assert bc.budget.max_retrieval_rounds == 5


class TestBudgetConsumption:
    """消耗追踪测试"""

    def test_consume_retrieval_round_tracking(self):
        """1 次检索 → consumed.retrieval_rounds=1"""
        bc = BudgetController("P2")  # max=2
        action = bc.consume_retrieval_round()
        assert bc.budget.consumed.retrieval_rounds == 1
        # 1/2=0.5 < 0.8 → CONTINUE
        assert action == BudgetAction.CONTINUE

    def test_consume_rewrite_tracking(self):
        """consume_rewrite 追踪改写次数"""
        bc = BudgetController("P2")  # max_rewrites=2
        action = bc.consume_rewrite()
        assert bc.budget.consumed.rewrites == 1
        # 1/2=0.5 < 0.8 → CONTINUE
        assert action == BudgetAction.CONTINUE

    def test_consume_subtask_tracking(self):
        """consume_subtask 追踪子任务数"""
        bc = BudgetController("P2")  # max_subtasks=5
        action = bc.consume_subtask()
        assert bc.budget.consumed.subtasks == 1
        # 1/5=0.2 < 0.8 → CONTINUE
        assert action == BudgetAction.CONTINUE

    def test_consume_tool_call_tracking(self):
        """consume_tool_call 追踪工具调用数"""
        bc = BudgetController("P2")  # max_tool_calls=3
        action = bc.consume_tool_call()
        assert bc.budget.consumed.tool_calls == 1
        # 1/3≈0.33 < 0.8 → CONTINUE
        assert action == BudgetAction.CONTINUE

    def test_consume_tokens_tracking(self):
        """消耗 5000 token → consumed.tokens=5000"""
        bc = BudgetController("P2")  # total_token_budget=20000
        action = bc.consume_tokens(5000)
        assert bc.budget.consumed.tokens == 5000
        # 5000/20000=0.25 < 0.8 → CONTINUE
        assert action == BudgetAction.CONTINUE

    def test_consume_time_tracking(self):
        """consume_time 追踪耗时"""
        bc = BudgetController("P2")  # total_timeout_ms=5000
        action = bc.consume_time(1000)
        assert bc.budget.consumed.elapsed_ms == 1000
        # 1000/5000=0.2 < 0.8 → CONTINUE
        assert action == BudgetAction.CONTINUE

    def test_consume_tokens_accumulates(self):
        """多次消耗 token 累加"""
        bc = BudgetController("P2")
        bc.consume_tokens(2000)
        bc.consume_tokens(3000)
        assert bc.budget.consumed.tokens == 5000

    def test_consume_retrieval_accumulates(self):
        """多次检索轮次累加"""
        bc = BudgetController("P4")  # max=5
        for _ in range(3):
            bc.consume_retrieval_round()
        assert bc.budget.consumed.retrieval_rounds == 3

    def test_consume_tokens_negative_ignored(self):
        """负值 token 消耗被忽略，不累加"""
        bc = BudgetController("P2")
        bc.consume_tokens(-100)
        assert bc.budget.consumed.tokens == 0

    def test_consume_time_negative_ignored(self):
        """负值耗时消耗被忽略，不累加"""
        bc = BudgetController("P2")
        bc.consume_time(-100)
        assert bc.budget.consumed.elapsed_ms == 0


class TestBudgetOverflow:
    """超预算（硬限制）测试"""

    def test_retrieval_rounds_overflow_triggers_stop(self):
        """超过 max_retrieval_rounds → 触发 STOP"""
        bc = BudgetController("P2")  # max=2
        bc.consume_retrieval_round()  # 1 → CONTINUE
        bc.consume_retrieval_round()  # 2 → DOWNGRADE（满额）
        action = bc.consume_retrieval_round()  # 3 > 2 → STOP
        assert action == BudgetAction.STOP

    def test_token_overflow_triggers_stop(self):
        """超过 total_token_budget → 触发 STOP"""
        bc = BudgetController("P2")  # token_budget=20000
        bc.consume_tokens(20000)  # 满额 → DOWNGRADE
        action = bc.consume_tokens(1)  # 20001 > 20000 → STOP
        assert action == BudgetAction.STOP

    def test_timeout_overflow_triggers_stop(self):
        """超过 total_timeout_ms → 触发 STOP（超时属硬限制）"""
        bc = BudgetController("P2")  # total_timeout_ms=5000
        action = bc.consume_time(5001)  # 5001 > 5000 → STOP
        assert action == BudgetAction.STOP
        assert bc.budget.consumed.elapsed_ms == 5001

    def test_rewrite_overflow_triggers_stop(self):
        """超过 max_rewrites → 触发 STOP"""
        bc = BudgetController("P2")  # max_rewrites=2
        bc.consume_rewrite()  # 1
        bc.consume_rewrite()  # 2 → DOWNGRADE
        action = bc.consume_rewrite()  # 3 > 2 → STOP
        assert action == BudgetAction.STOP

    def test_subtask_overflow_triggers_stop(self):
        """超过 max_subtasks → 触发 STOP"""
        bc = BudgetController("P2")  # max_subtasks=5
        for _ in range(5):
            bc.consume_subtask()  # 到 5 → DOWNGRADE
        action = bc.consume_subtask()  # 6 > 5 → STOP
        assert action == BudgetAction.STOP

    def test_tool_call_overflow_triggers_stop(self):
        """超过 max_tool_calls → 触发 STOP"""
        bc = BudgetController("P2")  # max_tool_calls=3
        for _ in range(3):
            bc.consume_tool_call()
        action = bc.consume_tool_call()  # 4 > 3 → STOP
        assert action == BudgetAction.STOP

    def test_p0_retrieval_immediately_stop(self):
        """P0 不允许检索，1 次即超限 STOP"""
        bc = BudgetController("P0")  # max_retrieval_rounds=0
        action = bc.consume_retrieval_round()  # 1 > 0 → STOP
        assert action == BudgetAction.STOP

    def test_stop_priority_over_downgrade(self):
        """STOP 优先级高于 DOWNGRADE：多维度同时越限时返回 STOP"""
        bc = BudgetController("P2")  # max_retrieval=2, token_budget=20000
        # token 进入软阈值区间（16000/20000=0.8 → DOWNGRADE）
        bc.consume_tokens(16000)
        # 检索维度超硬限制
        bc.consume_retrieval_round()  # 1
        bc.consume_retrieval_round()  # 2
        action = bc.consume_retrieval_round()  # 3 > 2 → STOP
        # 即使 token 处于 DOWNGRADE 区间，硬限制优先
        assert action == BudgetAction.STOP


class TestDowngradeThreshold:
    """降级阈值（软阈值 80%）测试"""

    def test_retrieval_80_percent_triggers_downgrade(self):
        """检索轮次达 80% → 触发 DOWNGRADE"""
        bc = BudgetController("P4")  # max=5, 80%→4
        for _ in range(3):
            bc.consume_retrieval_round()  # 3/5=0.6 → CONTINUE
        assert bc.budget.consumed.retrieval_rounds == 3
        # 第 4 次: 4/5=0.8 → DOWNGRADE
        action = bc.consume_retrieval_round()
        assert action == BudgetAction.DOWNGRADE

    def test_token_80_percent_triggers_downgrade(self):
        """token 达 80% → 触发 DOWNGRADE"""
        bc = BudgetController("P4")  # token_budget=50000, 80%→40000
        action = bc.consume_tokens(40000)  # 40000/50000=0.8 → DOWNGRADE
        assert action == BudgetAction.DOWNGRADE
        # 未超硬限制
        assert bc.budget.consumed.tokens <= bc.budget.total_token_budget

    def test_time_80_percent_triggers_downgrade(self):
        """耗时达 80% → 触发 DOWNGRADE"""
        bc = BudgetController("P4")  # total_timeout_ms=30000, 80%→24000
        action = bc.consume_time(24000)  # 24000/30000=0.8 → DOWNGRADE
        assert action == BudgetAction.DOWNGRADE

    def test_below_80_percent_continues(self):
        """低于 80% 时持续 CONTINUE"""
        bc = BudgetController("P4")  # max=5
        # 3/5=0.6 < 0.8
        for _ in range(3):
            assert bc.consume_retrieval_round() == BudgetAction.CONTINUE

    def test_at_full_but_not_over_returns_downgrade(self):
        """恰好满额（等于上限）但未超 → DOWNGRADE 而非 STOP"""
        bc = BudgetController("P4")  # max=5
        for _ in range(4):
            bc.consume_retrieval_round()
        action = bc.consume_retrieval_round()  # 5/5=1.0, 5 不 > 5 → DOWNGRADE
        assert action == BudgetAction.DOWNGRADE


class TestRemainingAndSummary:
    """剩余预算与摘要测试"""

    def test_get_remaining_initial(self):
        """初始剩余等于上限"""
        bc = BudgetController("P2")
        remaining = bc.get_remaining()
        assert remaining["retrieval_rounds"] == 2
        assert remaining["rewrites"] == 2
        assert remaining["subtasks"] == 5
        assert remaining["tool_calls"] == 3
        assert remaining["tokens"] == 20000
        assert remaining["elapsed_ms"] == 5000

    def test_get_remaining_after_consumption(self):
        """消耗后剩余正确递减"""
        bc = BudgetController("P2")
        bc.consume_retrieval_round()
        bc.consume_tokens(5000)
        remaining = bc.get_remaining()
        assert remaining["retrieval_rounds"] == 1
        assert remaining["tokens"] == 15000

    def test_get_remaining_clamped_to_zero(self):
        """超限时剩余钳制为 0（不出现负数）"""
        bc = BudgetController("P2")
        for _ in range(3):
            bc.consume_retrieval_round()  # 3 > 2
        remaining = bc.get_remaining()
        assert remaining["retrieval_rounds"] == 0

    def test_get_summary_structure(self):
        """get_summary 返回完整信息（path/limits/consumed/remaining）"""
        bc = BudgetController("P3")
        bc.consume_retrieval_round()
        bc.consume_tokens(1000)
        summary = bc.get_summary()
        assert summary["path"] == "P3"
        assert "limits" in summary
        assert "consumed" in summary
        assert "remaining" in summary
        # limits 字段完整
        limits = summary["limits"]
        for key in [
            "max_retrieval_rounds",
            "max_rewrites",
            "max_subtasks",
            "max_tool_calls",
            "total_timeout_ms",
            "total_token_budget",
            "per_step_timeout_ms",
        ]:
            assert key in limits
        # consumed 反映实际消耗
        assert summary["consumed"]["retrieval_rounds"] == 1
        assert summary["consumed"]["tokens"] == 1000
        # remaining 与 limits/consumed 一致
        assert (
            summary["remaining"]["retrieval_rounds"]
            == limits["max_retrieval_rounds"] - 1
        )

    def test_get_summary_path_matches(self):
        """摘要中 path 与控制器路径一致"""
        bc = BudgetController("P4")
        assert bc.get_summary()["path"] == "P4"


class TestBudgetEnforcer:
    """预算执行器测试"""

    def test_enforce_continue_returns_signal(self):
        """CONTINUE → "continue" 信号"""
        enforcer = BudgetEnforcer()
        signal = enforcer.enforce(BudgetAction.CONTINUE)
        assert signal == "continue"

    def test_enforce_stop_returns_signal(self):
        """STOP → "stop" 信号"""
        enforcer = BudgetEnforcer()
        signal = enforcer.enforce(BudgetAction.STOP)
        assert signal == "stop"

    def test_enforce_downgrade_returns_signal(self):
        """DOWNGRADE → "downgrade" 信号"""
        enforcer = BudgetEnforcer()
        signal = enforcer.enforce(BudgetAction.DOWNGRADE)
        assert signal == "downgrade"

    def test_enforce_continue_does_not_increment_count(self):
        """CONTINUE 不增加任何计数"""
        enforcer = BudgetEnforcer()
        enforcer.enforce(BudgetAction.CONTINUE)
        assert enforcer.stop_count == 0
        assert enforcer.downgrade_count == 0

    def test_enforce_stop_increments_count(self):
        """STOP 累加 stop_count"""
        enforcer = BudgetEnforcer()
        enforcer.enforce(BudgetAction.STOP)
        enforcer.enforce(BudgetAction.STOP)
        assert enforcer.stop_count == 2

    def test_enforce_downgrade_increments_count(self):
        """DOWNGRADE 累加 downgrade_count"""
        enforcer = BudgetEnforcer()
        enforcer.enforce(BudgetAction.DOWNGRADE)
        assert enforcer.downgrade_count == 1

    def test_should_stop(self):
        """should_stop 状态正确"""
        enforcer = BudgetEnforcer()
        assert enforcer.should_stop() is False
        enforcer.enforce(BudgetAction.STOP)
        assert enforcer.should_stop() is True
        assert enforcer.should_continue() is False

    def test_should_downgrade(self):
        """should_downgrade 状态正确"""
        enforcer = BudgetEnforcer()
        assert enforcer.should_downgrade() is False
        enforcer.enforce(BudgetAction.DOWNGRADE)
        assert enforcer.should_downgrade() is True
        assert enforcer.should_continue() is False

    def test_should_continue_initial(self):
        """初始状态为 CONTINUE"""
        enforcer = BudgetEnforcer()
        assert enforcer.should_continue() is True
        assert enforcer.last_action == BudgetAction.CONTINUE

    def test_last_context_stored(self):
        """enforce 保存上下文供诊断"""
        enforcer = BudgetEnforcer()
        ctx = {"path": "P3", "dimension": "tokens"}
        enforcer.enforce(BudgetAction.STOP, context=ctx)
        assert enforcer.get_last_context() == ctx

    def test_get_stats(self):
        """get_stats 返回累计统计"""
        enforcer = BudgetEnforcer()
        enforcer.enforce(BudgetAction.STOP)
        enforcer.enforce(BudgetAction.DOWNGRADE)
        enforcer.enforce(BudgetAction.DOWNGRADE)
        stats = enforcer.get_stats()
        assert stats["stop_count"] == 1
        assert stats["downgrade_count"] == 2

    def test_reset(self):
        """reset 清空动作缓存与计数"""
        enforcer = BudgetEnforcer()
        enforcer.enforce(BudgetAction.STOP, context={"a": 1})
        enforcer.reset()
        assert enforcer.last_action == BudgetAction.CONTINUE
        assert enforcer.stop_count == 0
        assert enforcer.downgrade_count == 0
        assert enforcer.get_last_context() is None
        assert enforcer.should_continue() is True


class TestControllerEnforcerIntegration:
    """BudgetController 与 BudgetEnforcer 集成测试"""

    def test_full_flow_continue(self):
        """正常流程：消耗未达阈值 → continue"""
        bc = BudgetController("P2")
        enforcer = BudgetEnforcer()
        action = bc.consume_retrieval_round()  # 1/2 → CONTINUE
        signal = enforcer.enforce(action, context=bc.get_summary())
        assert signal == "continue"
        assert enforcer.should_continue() is True

    def test_full_flow_stop_on_retrieval_overflow(self):
        """超预算流程：检索超限 → stop"""
        bc = BudgetController("P2")
        enforcer = BudgetEnforcer()
        signals = []
        for _ in range(3):
            action = bc.consume_retrieval_round()
            signals.append(enforcer.enforce(action))
        # 第1次 continue, 第2次 downgrade(满额), 第3次 stop(超限)
        assert signals == ["continue", "downgrade", "stop"]
        assert enforcer.should_stop() is True
        assert enforcer.stop_count == 1

    def test_full_flow_timeout_stop(self):
        """超时流程：超过 total_timeout_ms → stop"""
        bc = BudgetController("P2")
        enforcer = BudgetEnforcer()
        action = bc.consume_time(bc.budget.total_timeout_ms + 1)
        signal = enforcer.enforce(action, context={"reason": "timeout"})
        assert action == BudgetAction.STOP
        assert signal == "stop"
        assert enforcer.should_stop() is True
        assert enforcer.get_last_context() == {"reason": "timeout"}

    def test_full_flow_downgrade_signal(self):
        """降级流程：达 80% → downgrade 信号"""
        bc = BudgetController("P4")  # max=5, 80%→4
        enforcer = BudgetEnforcer()
        for _ in range(4):
            bc.consume_retrieval_round()
        action = bc.consume_retrieval_round()  # 5/5 → DOWNGRADE
        signal = enforcer.enforce(action, context={"path": "P4"})
        assert action == BudgetAction.DOWNGRADE
        assert signal == "downgrade"
        assert enforcer.should_downgrade() is True
        assert enforcer.downgrade_count == 1
