"""
多轮记忆集成管理器 — M5.2 记忆模块

整合三类记忆（工作记忆、对话摘要、长期事实）+ 会话状态，
提供统一的多轮记忆管理接口。

职责:
  1. 每轮对话完成后更新所有记忆层
  2. 为新问题提供完整的多轮上下文（MemoryContext）
  3. 管理摘要生成时机
  4. 管理用户确认事实的持久化
  5. 会话清理（TTL 过期时删除工作记忆，保留长期事实）

用法:
    manager = MemoryManager()
    # 每轮完成后调用
    await manager.on_turn_complete(session_id, turn)
    # 新问题时获取上下文
    ctx = await manager.get_context_for_new_query(session_id)
"""

import logging
from typing import Any, Optional

from .conversation_summary.summarizer import ConversationSummarizer
from .long_term_memory.store import LongTermMemory
from .memory_models import MemoryContext, Turn
from .session_state.session_manager import RedisSessionManager
from .working_memory.manager import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    多轮记忆集成管理器

    整合工作记忆、对话摘要、长期事实记忆，提供统一接口。

    用法:
        # 使用默认配置（内存模式）
        manager = MemoryManager()

        # 使用 Redis
        manager = MemoryManager(redis_url="redis://localhost:6379/0")

        # 每轮完成后更新记忆
        await manager.on_turn_complete(session_id, turn)

        # 获取多轮上下文
        ctx = await manager.get_context_for_new_query(session_id)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        mock: bool = True,
        db_session: Optional[Any] = None,
        summary_interval: int = 5,
    ):
        """
        Args:
            redis_url: Redis 连接 URL，None 时使用环境变量或内存模式
            mock: 是否强制使用内存模式
            db_session: 数据库会话（用于长期记忆），None 时使用内存存储
            summary_interval: 摘要生成间隔（轮次数）
        """
        # 会话状态管理器（复用已有模块）
        self._session_manager = RedisSessionManager(
            redis_url=redis_url, mock=mock
        )

        # 获取 Redis 客户端（共享连接）
        redis_client = self._session_manager.client

        # 三类记忆
        self._working = WorkingMemory(redis_client=redis_client)
        self._summary = ConversationSummarizer(
            redis_client=redis_client,
            summary_interval=summary_interval,
        )
        self._facts = LongTermMemory(db_session=db_session)

        logger.info("MemoryManager 初始化完成 (mock=%s)", mock)

    # ============================================================
    # 核心接口
    # ============================================================

    async def on_turn_complete(self, session_id: str, turn: Turn) -> None:
        """
        每轮对话完成后调用

        更新所有记忆层:
          1. 追加原始事件到工作记忆历史
          2. 更新工作记忆状态（意图、实体、约束）
          3. 定期生成对话摘要
          4. 用户确认的事实 → 持久化到长期记忆

        Args:
            session_id: 会话 ID
            turn: 当前轮次记录
        """
        # 1. 追加原始事件到历史
        self._working.append_event(session_id, turn)

        # 2. 更新工作记忆（当前意图、实体、约束）
        self._working.update(session_id, turn)

        # 3. 定期生成对话摘要
        if self._summary.should_summarize(turn.turn_number):
            recent_turns = self._working.get_recent_turns(
                session_id, n=self._summary._interval
            )
            if recent_turns:
                summary = self._summary.summarize(session_id, recent_turns)
                self._summary.save(session_id, summary)
                logger.info(
                    "自动生成摘要: session=%s, turns=%s",
                    session_id,
                    summary.covered_turns,
                )

        # 4. 用户确认的事实 → 持久化
        if turn.user_confirmed_facts:
            await self._facts.save(
                session_id,
                turn.user_confirmed_facts,
                source_turn_id=turn.turn_id,
            )
            logger.info(
                "保存用户确认事实: session=%s, facts=%d",
                session_id,
                len(turn.user_confirmed_facts),
            )

        logger.debug(
            "记忆更新完成: session=%s, turn=%d",
            session_id,
            turn.turn_number,
        )

    async def get_context_for_new_query(
        self, session_id: str, recent_n: int = 3
    ) -> MemoryContext:
        """
        为新问题提供多轮上下文

        汇聚三类记忆 + 最近轮次，供指代消解和查询重写使用。

        Args:
            session_id: 会话 ID
            recent_n: 获取最近 n 轮对话

        Returns:
            MemoryContext 对象
        """
        # 工作记忆
        working_state = self._working.get(session_id)

        # 最新摘要
        summary = self._summary.get_latest(session_id)

        # 确认事实
        confirmed_facts = await self._facts.get(session_id)

        # 最近轮次
        recent_turns = self._working.get_recent_turns(session_id, n=recent_n)

        ctx = MemoryContext(
            working_memory=working_state,
            summary=summary,
            confirmed_facts=confirmed_facts,
            recent_turns=recent_turns,
        )

        logger.debug(
            "获取记忆上下文: session=%s, metrics=%d, docs=%d, turns=%d, facts=%d",
            session_id,
            len(ctx.mentioned_metrics),
            len(ctx.mentioned_docs),
            len(ctx.recent_turns),
            len(ctx.confirmed_facts),
        )
        return ctx

    # ============================================================
    # 会话管理
    # ============================================================

    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        return self._session_manager.create_session()

    def delete_session(self, session_id: str) -> None:
        """
        删除会话的所有记忆

        清除工作记忆和摘要，保留长期事实（跨会话持久化）。

        Args:
            session_id: 会话 ID
        """
        self._working.delete(session_id)
        self._summary.delete(session_id)
        self._session_manager.delete_session(session_id)
        logger.info("删除会话记忆: session=%s (长期事实已保留)", session_id)

    async def purge_session(self, session_id: str) -> None:
        """
        彻底清除会话所有数据（包括长期事实）

        Args:
            session_id: 会话 ID
        """
        self._working.delete(session_id)
        self._summary.delete(session_id)
        self._session_manager.delete_session(session_id)
        await self._facts.delete(session_id)
        logger.info("彻底清除会话数据: session=%s", session_id)

    # ============================================================
    # 属性访问
    # ============================================================

    @property
    def working_memory(self) -> WorkingMemory:
        return self._working

    @property
    def summarizer(self) -> ConversationSummarizer:
        return self._summary

    @property
    def long_term_memory(self) -> LongTermMemory:
        return self._facts

    @property
    def session_manager(self) -> RedisSessionManager:
        return self._session_manager
