"""
会话状态管理 — Phase 1 内存版

管理用户会话的状态机、历史记录和上下文。
Phase 2+ 会替换为 Redis 实现。

职责:
  - 创建/恢复会话
  - 保存会话状态（状态机、QuerySpec、证据等）
  - 管理会话 TTL
  - 幂等键缓存
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_platform.orchestration.state_machine import StateMachine


@dataclass
class SessionState:
    """会话状态"""

    session_id: str
    state_machine: StateMachine
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    turn_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_query_spec: Optional[Any] = None  # QuerySpec
    current_evidence_bundle: Optional[Any] = None  # EvidenceBundle
    current_route_decision: Optional[Any] = None  # RouteDecision
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self):
        """更新最后活跃时间"""
        self.last_active_at = time.time()

    def add_turn(self, query: str, answer: str, metadata: Optional[dict] = None):
        """添加一轮对话记录"""
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "query": query,
            "answer": answer,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        self.touch()


class SessionManager:
    """
    会话管理器 — Phase 1 内存版

    创建、恢复和管理用户会话。
    """

    def __init__(self, ttl_seconds: int = 1800):
        """
        Args:
            ttl_seconds: 会话过期时间（秒），默认 30 分钟
        """
        self._sessions: Dict[str, SessionState] = {}
        self._ttl = ttl_seconds
        self._idempotency_cache: Dict[str, Dict[str, Any]] = {}

    def create_session(self) -> SessionState:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        sm = StateMachine(session_id=session_id)
        sm.start()

        session = SessionState(
            session_id=session_id,
            state_machine=sm,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        获取会话，如果不存在或已过期返回 None

        Args:
            session_id: 会话 ID

        Returns:
            SessionState 对象，或 None
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        # 检查是否过期
        if self._is_expired(session):
            del self._sessions[session_id]
            return None

        session.touch()
        return session

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        """获取或创建会话"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        return self.create_session()

    def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    # ============================================================
    # 幂等键缓存
    # ============================================================

    def check_idempotency(self, key: str) -> Optional[Dict[str, Any]]:
        """
        检查幂等键，如果已存在返回缓存结果

        Args:
            key: 幂等键

        Returns:
            缓存的响应，或 None
        """
        return self._idempotency_cache.get(key)

    def cache_response(self, key: str, response: Dict[str, Any], ttl: int = 300):
        """
        缓存响应到幂等键

        Args:
            key: 幂等键
            response: 响应数据
            ttl: 缓存时间（秒）
        """
        self._idempotency_cache[key] = {
            "response": response,
            "cached_at": time.time(),
            "ttl": ttl,
        }

    # ============================================================
    # 内部方法
    # ============================================================

    def _is_expired(self, session: SessionState) -> bool:
        """检查会话是否过期"""
        elapsed = time.time() - session.last_active_at
        return elapsed > self._ttl

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量"""
        expired_ids = [
            sid for sid, session in self._sessions.items()
            if self._is_expired(session)
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)

    @property
    def active_count(self) -> int:
        """活跃会话数"""
        return len(self._sessions)
