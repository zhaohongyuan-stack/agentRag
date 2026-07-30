"""
Agent 状态机核心 — 状态迁移、事件记录、检查点

设计要点:
  1. 每次迁移校验合法性，非法迁移抛出 IllegalTransition
  2. 每次迁移记录事件（from/to/timestamp/metadata），支持重放
  3. 支持检查点保存和恢复（checkpoint），用于故障恢复
  4. 线程安全：同一时刻只有一个状态，迁移操作加锁

使用方式:
    from agent_platform.orchestration.state_machine import StateMachine

    sm = StateMachine(session_id="sess-001")
    sm.start()  # RECEIVED
    sm.transition(AgentState.NORMALIZED)
    sm.transition(AgentState.CONTEXT_RESOLVED)
    ...
    sm.transition(AgentState.RESPONDING)  # 终态
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exceptions import (
    IllegalTransition,
    StateNotInitializedError,
    TerminalStateError,
)
from .states import (
    TRANSITIONS,
    TERMINAL_STATES,
    AgentState,
    is_terminal,
    is_valid_transition,
)


@dataclass
class StateEvent:
    """
    状态迁移事件 — 记录每次状态变更

    用于执行轨迹重放和审计。
    """

    event_id: str
    from_state: AgentState
    to_state: AgentState
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Checkpoint:
    """
    检查点 — 保存状态机快照用于故障恢复

    包含当前状态和事件历史，恢复后可从检查点继续执行。
    """

    session_id: str
    state: AgentState
    events: List[StateEvent]
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "events": [e.to_dict() for e in self.events],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class StateMachine:
    """
    Agent 状态机

    管理 Agent 执行流程中的状态迁移，确保只执行合法迁移，
    并记录完整的事件历史用于重放和审计。

    Attributes:
        session_id: 会话 ID
        current_state: 当前状态（start() 后才有值）
        events: 状态迁移事件历史
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[AgentState] = None
        self._events: List[StateEvent] = []
        self._lock = threading.Lock()

    # ============================================================
    # 生命周期
    # ============================================================

    def start(self, initial_state: AgentState = AgentState.RECEIVED) -> AgentState:
        """
        启动状态机，设置初始状态

        Args:
            initial_state: 初始状态，默认 RECEIVED

        Returns:
            初始状态

        Raises:
            RuntimeError: 如果状态机已启动
        """
        with self._lock:
            if self._state is not None:
                raise RuntimeError(
                    f"状态机已启动，当前状态: {self._state.value}。"
                    f"如需重置请使用 reset()。"
                )
            self._state = initial_state
            event = StateEvent(
                event_id=str(uuid.uuid4()),
                from_state=AgentState.RECEIVED,  # 虚拟起始
                to_state=initial_state,
                timestamp=time.time(),
                metadata={"type": "start"},
            )
            self._events.append(event)
            return self._state

    def reset(self) -> None:
        """重置状态机到未初始化状态，清空所有事件"""
        with self._lock:
            self._state = None
            self._events.clear()

    # ============================================================
    # 状态迁移
    # ============================================================

    def transition(
        self,
        target: AgentState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """
        执行状态迁移

        Args:
            target: 目标状态
            metadata: 附加元数据（如触发迁移的原因、检索轮次等）

        Returns:
            迁移后的新状态

        Raises:
            StateNotInitializedError: 状态机未启动
            TerminalStateError: 当前为终态
            IllegalTransition: 迁移不合法
        """
        with self._lock:
            if self._state is None:
                raise StateNotInitializedError()

            if is_terminal(self._state):
                raise TerminalStateError(self._state.value)

            if not is_valid_transition(self._state, target):
                raise IllegalTransition(self._state.value, target.value)

            old_state = self._state
            self._state = target

            event = StateEvent(
                event_id=str(uuid.uuid4()),
                from_state=old_state,
                to_state=target,
                timestamp=time.time(),
                metadata=metadata or {},
            )
            self._events.append(event)

            return self._state

    # ============================================================
    # 查询
    # ============================================================

    @property
    def current_state(self) -> Optional[AgentState]:
        """当前状态"""
        return self._state

    @property
    def events(self) -> List[StateEvent]:
        """状态迁移事件历史（只读副本）"""
        return list(self._events)

    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self._state is not None and is_terminal(self._state)

    def get_valid_targets(self) -> List[AgentState]:
        """获取当前状态的合法目标状态列表"""
        if self._state is None:
            return []
        return TRANSITIONS.get(self._state, [])

    # ============================================================
    # 检查点
    # ============================================================

    def save_checkpoint(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        保存检查点

        用于故障恢复，保存当前状态和完整事件历史。

        Args:
            metadata: 附加元数据

        Returns:
            Checkpoint 对象
        """
        if self._state is None:
            raise StateNotInitializedError()

        return Checkpoint(
            session_id=self.session_id,
            state=self._state,
            events=list(self._events),
            created_at=time.time(),
            metadata=metadata or {},
        )

    def restore_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        从检查点恢复状态机

        Args:
            checkpoint: 之前保存的检查点
        """
        with self._lock:
            self.session_id = checkpoint.session_id
            self._state = checkpoint.state
            self._events = list(checkpoint.events)

    # ============================================================
    # 事件历史查询
    # ============================================================

    def get_event_log(self) -> List[dict]:
        """获取事件历史（dict 列表格式，用于日志输出）"""
        return [e.to_dict() for e in self._events]

    def get_state_trace(self) -> List[str]:
        """获取状态轨迹（状态值列表，用于快速查看流程）"""
        if not self._events:
            return []
        trace = [self._events[0].to_state.value]
        for event in self._events[1:]:
            trace.append(event.to_state.value)
        return trace

    def __repr__(self) -> str:
        state_str = self._state.value if self._state else "NOT_STARTED"
        return f"StateMachine(session={self.session_id}, state={state_str}, events={len(self._events)})"
