"""
工作记忆管理器 — M5.2 记忆模块

维护当前会话的"活跃"信息：最新意图、累积实体、活跃约束、任务进度。
基于 Redis 持久化，支持 TTL 过期。当 Redis 不可用时降级为内存存储。

职责:
  1. 追加原始对话事件到历史
  2. 每轮结束后更新工作记忆（意图、实体、约束）
  3. 累积维护 mentioned_metrics / mentioned_docs
  4. 提供 get_recent_turns 获取最近 n 轮对话

Redis Key 设计:
  - ace-rag:wm:{session_id}          → Hash: 工作记忆状态
  - ace-rag:wm:turns:{session_id}    → List: 最近对话轮次（JSON）
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..memory_models import Turn, WorkingMemoryState
from ..session_state.session_manager import _MockRedis

logger = logging.getLogger(__name__)

_WM_PREFIX = "ace-rag:wm:"
_WM_TURNS_PREFIX = "ace-rag:wm:turns:"

_DEFAULT_TTL = 3600
_MAX_TURNS_IN_MEMORY = 10  # 工作记忆中保留的最大轮次数


class WorkingMemory:
    """
    工作记忆管理器

    维护当前会话的活跃状态信息，每轮对话后更新。
    与 session_state 的区别：session_state 维护完整会话状态（状态机、预算等），
    WorkingMemory 专注于多轮对话所需的上下文信息（意图、实体、约束、指标、文档）。

    用法:
        wm = WorkingMemory()
        wm.update(session_id, turn)
        state = wm.get(session_id)
        recent = wm.get_recent_turns(session_id, n=3)
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        ttl_seconds: int = _DEFAULT_TTL,
        max_turns: int = _MAX_TURNS_IN_MEMORY,
    ):
        """
        Args:
            redis_client: Redis 客户端实例，None 时使用 _MockRedis
            ttl_seconds: 工作记忆 TTL（秒）
            max_turns: 保留的最大轮次数
        """
        self._client = redis_client if redis_client is not None else _MockRedis()
        self._ttl = ttl_seconds
        self._max_turns = max_turns

    # ============================================================
    # 对话事件管理
    # ============================================================

    def append_event(self, session_id: str, turn: Turn) -> None:
        """
        追加一轮对话事件到历史

        将 Turn 序列化后追加到 Redis List，保留最近 max_turns 轮。

        Args:
            session_id: 会话 ID
            turn: 对话轮次记录
        """
        key = f"{_WM_TURNS_PREFIX}{session_id}"
        turn_json = json.dumps(turn.to_dict(), ensure_ascii=False)

        # 使用 RPUSH 追加到列表尾部
        # _MockRedis 不支持 rpush，用替代方案
        if isinstance(self._client, _MockRedis):
            self._mock_rpush(key, turn_json)
        else:
            self._client.rpush(key, turn_json)
            # 裁剪到最近 max_turns 轮
            self._client.ltrim(key, -self._max_turns, -1)
            self._client.expire(key, self._ttl)

        logger.debug(
            "工作记忆追加事件: session=%s, turn=%d", session_id, turn.turn_number
        )

    def get_recent_turns(self, session_id: str, n: int = 3) -> List[Turn]:
        """
        获取最近 n 轮对话记录

        Args:
            session_id: 会话 ID
            n: 获取的轮次数

        Returns:
            Turn 列表，按时间正序（旧 -> 新）
        """
        key = f"{_WM_TURNS_PREFIX}{session_id}"
        if isinstance(self._client, _MockRedis):
            raw_list = self._mock_lrange(key, -n, -1)
        else:
            raw_list = self._client.lrange(key, -n, -1)

        if not raw_list:
            return []

        turns: List[Turn] = []
        for raw in raw_list:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                turns.append(Turn.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("解析对话轮次失败: %s", e)
                continue

        return turns

    # ============================================================
    # 工作记忆状态管理
    # ============================================================

    def update(self, session_id: str, turn: Turn) -> WorkingMemoryState:
        """
        更新工作记忆状态

        根据新一轮对话的意图、实体、约束更新工作记忆。
        累积维护 mentioned_metrics / mentioned_docs（去重）。

        Args:
            session_id: 会话 ID
            turn: 当前轮次记录

        Returns:
            更新后的 WorkingMemoryState
        """
        state = self.get(session_id)

        if state is None:
            state = WorkingMemoryState(session_id=session_id)

        # 更新当前意图
        if turn.intent:
            state.current_intent = turn.intent

        # 累积实体（去重）
        for entity in turn.entities:
            if entity not in state.current_entities:
                state.current_entities.append(entity)

            # 提取指标和文档名称
            entity_type = entity.get("entity_type", "")
            entity_value = entity.get("value", "")
            if entity_type == "metric_name" and entity_value:
                if entity_value not in state.mentioned_metrics:
                    state.mentioned_metrics.append(entity_value)
            elif entity_type == "doc_name" and entity_value:
                if entity_value not in state.mentioned_docs:
                    state.mentioned_docs.append(entity_value)

        # 累积约束（去重）
        for constraint in turn.constraints:
            if constraint not in state.active_constraints:
                state.active_constraints.append(constraint)

        # 更新任务进度
        if turn.query_spec:
            progress = turn.query_spec.get("task_progress", {})
            if isinstance(progress, dict):
                state.task_progress.update(progress)

        # 更新轮次计数
        state.turn_count = max(state.turn_count, turn.turn_number)
        state.updated_at = time.time()

        self._save_state(state)
        logger.debug(
            "工作记忆更新: session=%s, intent=%s, metrics=%d, docs=%d",
            session_id,
            state.current_intent,
            len(state.mentioned_metrics),
            len(state.mentioned_docs),
        )
        return state

    def get(self, session_id: str) -> Optional[WorkingMemoryState]:
        """
        获取工作记忆状态

        Args:
            session_id: 会话 ID

        Returns:
            WorkingMemoryState，不存在返回 None
        """
        key = f"{_WM_PREFIX}{session_id}"
        data = self._client.hgetall(key)
        if not data:
            return None
        return self._deserialize_state(data)

    def delete(self, session_id: str) -> None:
        """
        删除工作记忆

        清除工作记忆状态和对话轮次历史。
        用于会话结束或过期清理。

        Args:
            session_id: 会话 ID
        """
        self._client.delete(
            f"{_WM_PREFIX}{session_id}",
            f"{_WM_TURNS_PREFIX}{session_id}",
        )
        logger.debug("删除工作记忆: session=%s", session_id)

    # ============================================================
    # 内部方法
    # ============================================================

    def _save_state(self, state: WorkingMemoryState) -> None:
        """序列化存储工作记忆状态到 Redis Hash"""
        key = f"{_WM_PREFIX}{state.session_id}"
        mapping = {
            "session_id": state.session_id,
            "current_intent": state.current_intent,
            "current_entities": json.dumps(
                state.current_entities, ensure_ascii=False
            ),
            "active_constraints": json.dumps(
                state.active_constraints, ensure_ascii=False
            ),
            "task_progress": json.dumps(
                state.task_progress, ensure_ascii=False
            ),
            "mentioned_metrics": json.dumps(
                state.mentioned_metrics, ensure_ascii=False
            ),
            "mentioned_docs": json.dumps(
                state.mentioned_docs, ensure_ascii=False
            ),
            "turn_count": str(state.turn_count),
            "updated_at": str(state.updated_at),
        }
        self._client.hset(key, mapping=mapping)
        self._client.expire(key, self._ttl)

    @staticmethod
    def _deserialize_state(data: Dict[str, str]) -> WorkingMemoryState:
        """从 Redis Hash 反序列化 WorkingMemoryState"""
        def _parse_json(raw: str, default):
            if not raw:
                return default
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return default

        return WorkingMemoryState(
            session_id=data.get("session_id", ""),
            current_intent=data.get("current_intent", ""),
            current_entities=_parse_json(data.get("current_entities", "[]"), []),
            active_constraints=_parse_json(data.get("active_constraints", "[]"), []),
            task_progress=_parse_json(data.get("task_progress", "{}"), {}),
            mentioned_metrics=_parse_json(data.get("mentioned_metrics", "[]"), []),
            mentioned_docs=_parse_json(data.get("mentioned_docs", "[]"), []),
            turn_count=int(data.get("turn_count", 0)),
            updated_at=float(data.get("updated_at", 0)),
        )

    # ----------------------------------------------------------
    # _MockRedis 的 List 操作模拟
    # ----------------------------------------------------------

    def _mock_rpush(self, key: str, value: str) -> None:
        """模拟 RPUSH：追加到内部列表"""
        if key not in self._client._data:
            self._client._data[key] = []
            self._client._type[key] = "list"
        self._client._data[key].append(value)
        # 裁剪
        if len(self._client._data[key]) > self._max_turns:
            self._client._data[key] = self._client._data[key][-self._max_turns :]
        # 设置 TTL
        self._client._expiry[key] = time.time() + self._ttl

    def _mock_lrange(self, key: str, start: int, stop: int) -> List[str]:
        """模拟 LRANGE"""
        if key not in self._client._data:
            return []
        data = self._client._data[key]
        if not isinstance(data, list):
            return []
        # 处理负索引
        length = len(data)
        if start < 0:
            start = max(0, length + start)
        if stop < 0:
            stop = length + stop
        return data[start : stop + 1]
