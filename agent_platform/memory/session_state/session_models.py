"""
会话状态数据模型 — M2.5 记忆/会话状态模块

定义会话相关的数据结构：
  - SessionTurn: 单轮对话记录
  - SessionState: 会话完整状态（可序列化用于 Redis 持久化）
  - SessionCheckpoint: 状态机检查点快照

设计要点:
  1. 所有模型支持 to_dict / from_dict 序列化，适配 Redis 存储
  2. SessionState.current_state 为字符串（状态机状态值），而非 StateMachine 对象
     —— 这样可以完整序列化到 Redis，恢复后再重建状态机
  3. SessionState.add_turn 同时记录 query/answer，并支持从 metadata 提取 intent/complexity
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionTurn:
    """
    单轮对话记录

    记录用户查询、Agent 回答以及该轮的意图、复杂度等元信息。
    用于对话历史回溯和共指消解。

    Attributes:
        turn_id: 轮次唯一 ID
        query: 用户查询文本
        answer: Agent 回答文本
        intent: 意图分类结果（如 "factual", "procedural", "comparison"）
        complexity: 复杂度等级（如 "L0", "L1", "L2", "L3"）
        timestamp: 该轮时间戳（Unix epoch）
        metadata: 附加元数据（实体、检索轮次等）
    """

    turn_id: str
    query: str
    answer: str
    intent: str
    complexity: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典（可 JSON 化）"""
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "answer": self.answer,
            "intent": self.intent,
            "complexity": self.complexity,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionTurn":
        """从字典反序列化"""
        return cls(
            turn_id=data["turn_id"],
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            intent=data.get("intent", ""),
            complexity=data.get("complexity", ""),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionState:
    """
    会话完整状态

    包含会话 ID、当前状态机状态、对话历史、预算消耗等。
    可完整序列化到 Redis Hash，恢复后重建状态机。

    与 Phase 1 的 gateway/session_handler/session_state.SessionState 区别:
      - current_state 为字符串而非 StateMachine 对象（便于序列化）
      - turns 为 SessionTurn 列表而非原始 dict 列表（结构化）
      - 新增 budget_consumed、query_spec 字段（支持预算控制和计划复用）

    Attributes:
        session_id: 会话 ID
        current_state: 当前状态机状态（AgentState 枚举值字符串，如 "RECEIVED"）
        turns: 对话轮次历史列表
        created_at: 会话创建时间戳
        updated_at: 会话最后更新时间戳
        query_spec: 当前 QuerySpec（字典形式，None 表示未设置）
        budget_consumed: 预算消耗计数（如 {"retrieval_rounds": 1, "tokens": 500}）
        metadata: 附加元数据
    """

    session_id: str
    current_state: str
    turns: List[SessionTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    query_spec: Optional[Dict[str, Any]] = None
    budget_consumed: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典（可 JSON 化，用于 Redis 存储）"""
        return {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "query_spec": self.query_spec,
            "budget_consumed": self.budget_consumed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """从字典反序列化"""
        turns_raw = data.get("turns", [])
        turns = [SessionTurn.from_dict(t) for t in turns_raw] if turns_raw else []
        return cls(
            session_id=data["session_id"],
            current_state=data.get("current_state", "RECEIVED"),
            turns=turns,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            query_spec=data.get("query_spec"),
            budget_consumed=data.get("budget_consumed", {}),
            metadata=data.get("metadata", {}),
        )

    def add_turn(
        self,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionTurn:
        """
        添加一轮对话记录

        从 metadata 中提取 intent 和 complexity（若存在），
        创建 SessionTurn 并追加到历史列表，同时更新 updated_at。

        Args:
            query: 用户查询
            answer: Agent 回答
            metadata: 附加元数据，可包含 intent、complexity、entities 等

        Returns:
            新创建的 SessionTurn
        """
        meta = metadata or {}
        turn = SessionTurn(
            turn_id=str(uuid.uuid4()),
            query=query,
            answer=answer,
            intent=meta.get("intent", ""),
            complexity=meta.get("complexity", ""),
            timestamp=time.time(),
            metadata=meta,
        )
        self.turns.append(turn)
        self.updated_at = time.time()
        return turn

    def recent_queries(self, n: int = 5) -> List[str]:
        """
        获取最近 n 轮的用户查询（用于共指消解）

        共指消解模块（query_understanding/reference_resolver）使用此方法
        获取历史查询上下文，解析 "它"、"该规定" 等指代词。

        Args:
            n: 获取的轮次数

        Returns:
            查询文本列表，按时间正序（旧 -> 新）
        """
        if n <= 0:
            return []
        recent = self.turns[-n:]
        return [t.query for t in recent]

    def mentioned_entities(self) -> List[dict]:
        """
        提取历史轮次中提及的实体

        从每轮 metadata["entities"] 中收集实体信息。
        实体由 query_understanding/entity_extractor 模块提取并写入 metadata。

        典型实体结构: {"name": "GDPR", "type": "regulation", "version": "2018"}

        Returns:
            实体字典列表，按轮次时间顺序排列
        """
        entities: List[dict] = []
        for turn in self.turns:
            turn_entities = turn.metadata.get("entities", [])
            if isinstance(turn_entities, list):
                entities.extend(turn_entities)
        return entities


@dataclass
class SessionCheckpoint:
    """
    会话检查点 — 状态机快照

    保存状态机当前状态和事件历史，用于故障恢复。
    恢复后可从检查点继续执行剩余流程。

    与 orchestration/state_machine.Checkpoint 的关系:
      - 两者都保存状态机快照，但 SessionCheckpoint 面向 Redis 持久化
      - events 为字典列表（已序列化），而非 StateEvent 对象列表
      - 新增 checkpoint_id 字段，支持同一会话多个检查点版本

    Attributes:
        checkpoint_id: 检查点唯一 ID
        session_id: 所属会话 ID
        state_machine_state: 状态机当前状态（AgentState 枚举值字符串）
        events: 状态迁移事件历史（字典列表，与 StateEvent.to_dict() 格式一致）
        timestamp: 检查点创建时间戳
        metadata: 附加元数据
    """

    checkpoint_id: str
    session_id: str
    state_machine_state: str
    events: List[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "state_machine_state": self.state_machine_state,
            "events": self.events,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionCheckpoint":
        """从字典反序列化"""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data.get("session_id", ""),
            state_machine_state=data.get("state_machine_state", "RECEIVED"),
            events=data.get("events", []),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )
