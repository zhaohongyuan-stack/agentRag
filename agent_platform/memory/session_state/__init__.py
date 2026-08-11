"""
会话状态管理模块 — M2.5 记忆/会话状态

提供基于 Redis 的会话状态持久化管理，支持故障恢复和水平扩展。
开发环境可降级为内存存储（mock 模式），无需安装 Redis。

核心导出:
    RedisSessionManager — Redis 会话管理器（含内存降级）
    SessionState        — 会话完整状态（可序列化）
    SessionCheckpoint   — 状态机检查点快照
    SessionTurn         — 单轮对话记录

使用示例:
    from agent_platform.memory.session_state import RedisSessionManager

    manager = RedisSessionManager(mock=True)  # 开发环境
    session_id = manager.create_session()
    manager.add_turn(session_id, "你好", "你好！有什么可以帮您？")
"""

from .session_manager import RedisSessionManager
from .session_models import SessionCheckpoint, SessionState, SessionTurn

__all__ = [
    "RedisSessionManager",
    "SessionState",
    "SessionCheckpoint",
    "SessionTurn",
]
