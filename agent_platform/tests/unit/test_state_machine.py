"""
状态机单元测试

测试用例:
  - 合法迁移: ROUTED → RETRIEVING → 成功
  - 非法迁移: RECEIVED → RETRIEVING → 抛出 IllegalTransition
  - 终态迁移: RESPONDING → RETRIEVING → 抛出 TerminalStateError
  - 事件记录: 每次迁移记录事件
  - 检查点保存和恢复
  - 完整流程状态序列
"""

import pytest

from agent_platform.orchestration.state_machine import (
    AgentState,
    Checkpoint,
    IllegalTransition,
    StateMachine,
    StateNotInitializedError,
    TerminalStateError,
    is_terminal,
    is_valid_transition,
)


class TestStateMachineBasics:
    """状态机基础功能测试"""

    def test_start_sets_initial_state(self):
        """启动状态机设置初始状态"""
        sm = StateMachine(session_id="test-001")
        state = sm.start()
        assert state == AgentState.RECEIVED
        assert sm.current_state == AgentState.RECEIVED

    def test_start_with_custom_initial_state(self):
        """支持自定义初始状态"""
        sm = StateMachine(session_id="test-002")
        state = sm.start(AgentState.ROUTED)
        assert state == AgentState.ROUTED

    def test_double_start_raises_error(self):
        """重复启动抛出异常"""
        sm = StateMachine(session_id="test-003")
        sm.start()
        with pytest.raises(RuntimeError):
            sm.start()

    def test_reset(self):
        """重置状态机"""
        sm = StateMachine(session_id="test-004")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        sm.reset()
        assert sm.current_state is None
        assert len(sm.events) == 0

    def test_not_initialized_raises(self):
        """未初始化时迁移抛出异常"""
        sm = StateMachine(session_id="test-005")
        with pytest.raises(StateNotInitializedError):
            sm.transition(AgentState.NORMALIZED)


class TestLegalTransitions:
    """合法迁移测试"""

    def test_legal_transition_succeeds(self):
        """合法迁移成功"""
        sm = StateMachine(session_id="test-010")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        assert sm.current_state == AgentState.NORMALIZED

    def test_routed_to_retrieving(self):
        """ROUTED → RETRIEVING 合法"""
        sm = StateMachine(session_id="test-011")
        sm.start()
        for target in [AgentState.NORMALIZED, AgentState.CONTEXT_RESOLVED,
                       AgentState.ANALYZED, AgentState.ROUTED, AgentState.RETRIEVING]:
            sm.transition(target)
        assert sm.current_state == AgentState.RETRIEVING

    def test_full_happy_path(self):
        """完整正常流程"""
        sm = StateMachine(session_id="test-012")
        sm.start()
        path = [
            AgentState.NORMALIZED,
            AgentState.CONTEXT_RESOLVED,
            AgentState.ANALYZED,
            AgentState.ROUTED,
            AgentState.RETRIEVING,
            AgentState.EVIDENCE_ASSEMBLING,
            AgentState.EVIDENCE_VALIDATING,
            AgentState.GENERATING,
            AgentState.ANSWER_VALIDATING,
            AgentState.RESPONDING,
        ]
        for target in path:
            sm.transition(target)
        assert sm.current_state == AgentState.RESPONDING
        assert sm.is_terminal()

    def test_refusal_path(self):
        """拒答流程"""
        sm = StateMachine(session_id="test-013")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        sm.transition(AgentState.CONTEXT_RESOLVED)
        sm.transition(AgentState.ANALYZED)
        sm.transition(AgentState.ROUTED)
        sm.transition(AgentState.RETRIEVING)
        sm.transition(AgentState.EVIDENCE_ASSEMBLING)
        sm.transition(AgentState.EVIDENCE_VALIDATING)
        sm.transition(AgentState.REFUSING)
        sm.transition(AgentState.RESPONDING)
        assert sm.is_terminal()


class TestIllegalTransitions:
    """非法迁移测试"""

    def test_illegal_transition_raises(self):
        """非法迁移抛出 IllegalTransition"""
        sm = StateMachine(session_id="test-020")
        sm.start()
        with pytest.raises(IllegalTransition) as exc_info:
            sm.transition(AgentState.RETRIEVING)
        assert exc_info.value.current_state == "RECEIVED"
        assert exc_info.value.target_state == "RETRIEVING"

    def test_terminal_state_raises(self):
        """终态迁移抛出 TerminalStateError"""
        sm = StateMachine(session_id="test-021")
        sm.start()
        # 走到 RESPONDING（终态）
        for target in [AgentState.NORMALIZED, AgentState.CONTEXT_RESOLVED,
                       AgentState.ANALYZED, AgentState.ROUTED, AgentState.RETRIEVING,
                       AgentState.EVIDENCE_ASSEMBLING, AgentState.EVIDENCE_VALIDATING,
                       AgentState.GENERATING, AgentState.ANSWER_VALIDATING,
                       AgentState.RESPONDING]:
            sm.transition(target)
        with pytest.raises(TerminalStateError):
            sm.transition(AgentState.RETRIEVING)

    def test_failed_is_terminal(self):
        """FAILED 是终态"""
        sm = StateMachine(session_id="test-022")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        sm.transition(AgentState.CONTEXT_RESOLVED)
        sm.transition(AgentState.ANALYZED)
        sm.transition(AgentState.ROUTED)
        sm.transition(AgentState.CLARIFYING)
        sm.transition(AgentState.FAILED)
        assert sm.is_terminal()


class TestEventRecording:
    """事件记录测试"""

    def test_events_recorded(self):
        """每次迁移记录事件"""
        sm = StateMachine(session_id="test-030")
        sm.start()
        sm.transition(AgentState.NORMALIZED, metadata={"step": "normalize"})
        events = sm.events
        assert len(events) == 2  # start + 1 transition
        assert events[1].from_state == AgentState.RECEIVED
        assert events[1].to_state == AgentState.NORMALIZED
        assert events[1].metadata == {"step": "normalize"}

    def test_event_has_timestamp(self):
        """事件包含时间戳"""
        sm = StateMachine(session_id="test-031")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        assert sm.events[-1].timestamp > 0

    def test_event_has_id(self):
        """事件包含唯一 ID"""
        sm = StateMachine(session_id="test-032")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        assert sm.events[-1].event_id  # 非空字符串

    def test_state_trace(self):
        """状态轨迹正确"""
        sm = StateMachine(session_id="test-033")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        sm.transition(AgentState.CONTEXT_RESOLVED)
        trace = sm.get_state_trace()
        assert trace == ["RECEIVED", "NORMALIZED", "CONTEXT_RESOLVED"]

    def test_event_log_format(self):
        """事件日志格式正确"""
        sm = StateMachine(session_id="test-034")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        log = sm.get_event_log()
        assert len(log) == 2
        assert "from_state" in log[1]
        assert "to_state" in log[1]
        assert "timestamp" in log[1]
        assert "metadata" in log[1]


class TestCheckpoint:
    """检查点测试"""

    def test_save_and_restore(self):
        """保存和恢复检查点"""
        sm = StateMachine(session_id="test-040")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        sm.transition(AgentState.CONTEXT_RESOLVED)

        # 保存检查点
        cp = sm.save_checkpoint(metadata={"round": 1})
        assert cp.state == AgentState.CONTEXT_RESOLVED
        assert cp.session_id == "test-040"

        # 继续迁移
        sm.transition(AgentState.ANALYZED)
        assert sm.current_state == AgentState.ANALYZED

        # 恢复检查点
        sm.restore_checkpoint(cp)
        assert sm.current_state == AgentState.CONTEXT_RESOLVED
        assert len(sm.events) == 3  # start + 2 transitions

    def test_checkpoint_to_dict(self):
        """检查点序列化"""
        sm = StateMachine(session_id="test-041")
        sm.start()
        sm.transition(AgentState.NORMALIZED)
        cp = sm.save_checkpoint()
        d = cp.to_dict()
        assert d["session_id"] == "test-041"
        assert d["state"] == "NORMALIZED"
        assert isinstance(d["events"], list)


class TestUtilityFunctions:
    """工具函数测试"""

    def test_is_valid_transition(self):
        """is_valid_transition 函数"""
        assert is_valid_transition(AgentState.RECEIVED, AgentState.NORMALIZED) is True
        assert is_valid_transition(AgentState.RECEIVED, AgentState.RETRIEVING) is False

    def test_is_terminal(self):
        """is_terminal 函数"""
        assert is_terminal(AgentState.RESPONDING) is True
        assert is_terminal(AgentState.FAILED) is True
        assert is_terminal(AgentState.RECEIVED) is False
        assert is_terminal(AgentState.RETRIEVING) is False

    def test_get_valid_targets(self):
        """get_valid_targets 函数"""
        targets = AgentState.RECEIVED
        from agent_platform.orchestration.state_machine import get_valid_targets
        valid = get_valid_targets(AgentState.ROUTED)
        assert AgentState.RETRIEVING in valid
        assert AgentState.CLARIFYING in valid
        assert AgentState.PLANNING in valid
