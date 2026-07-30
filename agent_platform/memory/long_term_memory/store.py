"""
长期事实记忆存储 — M5.2 记忆模块

存储用户确认的事实，持久化到数据库（开发环境用内存模拟）。
与工作记忆不同，长期记忆不会随会话过期而消失。

职责:
  1. 保存用户确认的事实
  2. 按会话 ID 查询事实
  3. 按事实类型查询
  4. 删除过期或无效事实

持久化策略:
  - 生产环境: PostgreSQL（通过 AsyncSession）
  - 开发/测试环境: 内存字典模拟（_MemoryStore）

表结构（PostgreSQL）:
  CREATE TABLE confirmed_facts (
      fact_id        VARCHAR(64) PRIMARY KEY,
      session_id     VARCHAR(64) NOT NULL,
      fact_type      VARCHAR(32) NOT NULL,
      fact_content   TEXT NOT NULL,
      source_turn_id VARCHAR(64),
      confirmed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX idx_facts_session ON confirmed_facts(session_id);
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..memory_models import ConfirmedFact

logger = logging.getLogger(__name__)


class _MemoryStore:
    """
    内存事实存储 — 用于开发/测试环境

    模拟数据库表操作，支持按 session_id 和 fact_type 查询。
    非线程安全，仅用于单线程开发/测试。
    """

    def __init__(self):
        # _facts: fact_id → ConfirmedFact
        self._facts: Dict[str, ConfirmedFact] = {}

    def insert(self, fact: ConfirmedFact) -> None:
        self._facts[fact.fact_id] = fact

    def select_by_session(self, session_id: str) -> List[ConfirmedFact]:
        return [
            f for f in self._facts.values() if f.session_id == session_id
        ]

    def select_by_type(
        self, session_id: str, fact_type: str
    ) -> List[ConfirmedFact]:
        return [
            f
            for f in self._facts.values()
            if f.session_id == session_id and f.fact_type == fact_type
        ]

    def delete_by_session(self, session_id: str) -> int:
        to_delete = [
            fid for fid, f in self._facts.items() if f.session_id == session_id
        ]
        for fid in to_delete:
            del self._facts[fid]
        return len(to_delete)

    def count(self) -> int:
        return len(self._facts)


class LongTermMemory:
    """
    长期事实记忆存储

    存储用户在对话中确认的事实，跨会话持久化。

    用法:
        ltm = LongTermMemory()  # 内存模式
        # 保存用户确认的事实
        ltm.save(session_id, [
            {"fact_type": "regulation", "fact_content": "核心一级资本充足率最低8%"}
        ])
        # 查询事实
        facts = ltm.get(session_id)
    """

    def __init__(self, db_session: Optional[Any] = None):
        """
        Args:
            db_session: 数据库会话（AsyncSession），None 时使用内存存储
        """
        self._db = db_session
        self._store: Optional[_MemoryStore] = None

        if db_session is None:
            self._store = _MemoryStore()
            logger.info("长期记忆运行在内存模式（无数据库）")

    async def save(
        self,
        session_id: str,
        facts: List[Dict[str, Any]],
        source_turn_id: str = "",
    ) -> List[ConfirmedFact]:
        """
        保存用户确认的事实

        将事实列表持久化到长期记忆存储。

        Args:
            session_id: 会话 ID
            facts: 事实字典列表，每项含 fact_type 和 fact_content
            source_turn_id: 来源轮次 ID

        Returns:
            已保存的 ConfirmedFact 列表
        """
        saved: List[ConfirmedFact] = []

        for fact_data in facts:
            fact = ConfirmedFact(
                fact_id=f"fact-{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                fact_type=fact_data.get("fact_type", ""),
                fact_content=fact_data.get("fact_content", ""),
                source_turn_id=source_turn_id,
                confirmed_at=time.time(),
            )

            if self._store is not None:
                self._store.insert(fact)
            else:
                # 生产环境: INSERT 到数据库
                # await self._db.execute(
                #     "INSERT INTO confirmed_facts (...) VALUES (...)",
                #     ...
                # )
                logger.warning("数据库模式未实现，事实未保存")

            saved.append(fact)

        logger.info(
            "长期记忆保存: session=%s, facts=%d", session_id, len(saved)
        )
        return saved

    async def get(self, session_id: str) -> List[ConfirmedFact]:
        """
        获取会话的所有确认事实

        Args:
            session_id: 会话 ID

        Returns:
            ConfirmedFact 列表
        """
        if self._store is not None:
            return self._store.select_by_session(session_id)

        # 生产环境: SELECT FROM confirmed_facts WHERE session_id = ?
        logger.warning("数据库模式未实现，返回空列表")
        return []

    async def get_by_type(
        self, session_id: str, fact_type: str
    ) -> List[ConfirmedFact]:
        """
        按类型获取确认事实

        Args:
            session_id: 会话 ID
            fact_type: 事实类型

        Returns:
            ConfirmedFact 列表
        """
        if self._store is not None:
            return self._store.select_by_type(session_id, fact_type)

        logger.warning("数据库模式未实现，返回空列表")
        return []

    async def delete(self, session_id: str) -> int:
        """
        删除会话的所有确认事实

        Args:
            session_id: 会话 ID

        Returns:
            删除的事实数量
        """
        if self._store is not None:
            count = self._store.delete_by_session(session_id)
            logger.info("长期记忆删除: session=%s, count=%d", session_id, count)
            return count

        logger.warning("数据库模式未实现")
        return 0

    @property
    def is_mock(self) -> bool:
        """是否运行在内存模式"""
        return self._store is not None
