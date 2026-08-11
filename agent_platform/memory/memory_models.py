"""
多轮记忆数据模型 — M5.2 记忆模块

定义记忆系统的核心数据结构：
  - Turn: 单轮对话完整记录（含用户确认事实、查询规格等）
  - WorkingMemoryState: 工作记忆状态（当前意图、实体、约束、任务进度）
  - ConversationSummaryData: 对话摘要数据
  - ConfirmedFact: 用户确认的事实记录
  - MemoryContext: 为新问题提供的完整多轮上下文

设计要点:
  1. 所有模型支持 to_dict / from_dict 序列化，适配 Redis/DB 存储
  2. WorkingMemoryState 记录"当前"状态（最新一轮的意图、实体、约束），
     而非完整历史（历史由 session_state 的 turns 列表维护）
  3. MemoryContext 汇聚三类记忆 + 最近轮次，供指代消解和查询重写使用
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """
    单轮对话完整记录

    比 session_state.SessionTurn 更丰富，包含查询规格、用户确认事实等
    多轮记忆所需的信息。

    Attributes:
        turn_id: 轮次唯一 ID
        turn_number: 轮次序号（从 1 开始）
        query: 用户查询文本
        answer: Agent 回答文本
        intent: 意图分类结果
        entities: 实体列表（每项为 dict，含 entity_type/value）
        constraints: 约束条件列表
        query_spec: 查询规格（QuerySpec 字典）
        user_confirmed_facts: 用户确认的事实列表
        timestamp: 时间戳
        metadata: 附加元数据
    """

    turn_id: str
    turn_number: int
    query: str
    answer: str = ""
    intent: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    query_spec: Optional[Dict[str, Any]] = None
    user_confirmed_facts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "turn_number": self.turn_number,
            "query": self.query,
            "answer": self.answer,
            "intent": self.intent,
            "entities": list(self.entities),
            "constraints": list(self.constraints),
            "query_spec": self.query_spec,
            "user_confirmed_facts": list(self.user_confirmed_facts),
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        return cls(
            turn_id=data.get("turn_id", str(uuid.uuid4())),
            turn_number=data.get("turn_number", 0),
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            intent=data.get("intent", ""),
            entities=data.get("entities", []),
            constraints=data.get("constraints", []),
            query_spec=data.get("query_spec"),
            user_confirmed_facts=data.get("user_confirmed_facts", []),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkingMemoryState:
    """
    工作记忆状态

    记录当前会话的"活跃"信息：最新意图、累积实体、活跃约束、任务进度。
    每轮对话后更新，过期后清除（由 Redis TTL 管理）。

    Attributes:
        session_id: 会话 ID
        current_intent: 当前意图（最新一轮的意图分类结果）
        current_entities: 当前累积的实体列表
        active_constraints: 活跃约束条件列表
        task_progress: 任务进度（如 {"retrieval": "done", "generation": "pending"}）
        mentioned_metrics: 对话中提到过的监管指标名称列表
        mentioned_docs: 对话中提到过的法规文档名称列表
        turn_count: 已完成的轮次数
        updated_at: 最后更新时间戳
    """

    session_id: str
    current_intent: str = ""
    current_entities: List[Dict[str, Any]] = field(default_factory=list)
    active_constraints: List[str] = field(default_factory=list)
    task_progress: Dict[str, str] = field(default_factory=dict)
    mentioned_metrics: List[str] = field(default_factory=list)
    mentioned_docs: List[str] = field(default_factory=list)
    turn_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "current_intent": self.current_intent,
            "current_entities": list(self.current_entities),
            "active_constraints": list(self.active_constraints),
            "task_progress": dict(self.task_progress),
            "mentioned_metrics": list(self.mentioned_metrics),
            "mentioned_docs": list(self.mentioned_docs),
            "turn_count": self.turn_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingMemoryState":
        return cls(
            session_id=data.get("session_id", ""),
            current_intent=data.get("current_intent", ""),
            current_entities=data.get("current_entities", []),
            active_constraints=data.get("active_constraints", []),
            task_progress=data.get("task_progress", {}),
            mentioned_metrics=data.get("mentioned_metrics", []),
            mentioned_docs=data.get("mentioned_docs", []),
            turn_count=data.get("turn_count", 0),
            updated_at=data.get("updated_at", time.time()),
        )


@dataclass
class ConversationSummaryData:
    """
    对话摘要数据

    定期生成的对话摘要，用于压缩历史轮次，为长会话提供上下文。

    Attributes:
        summary_id: 摘要唯一 ID
        session_id: 会话 ID
        summary_text: 摘要文本
        covered_turns: 涵盖的轮次范围（如 "1-5"）
        key_topics: 关键主题列表
        key_metrics: 关键指标列表
        key_docs: 关键文档列表
        turn_count: 生成时的轮次数
        created_at: 创建时间戳
    """

    summary_id: str
    session_id: str
    summary_text: str = ""
    covered_turns: str = ""
    key_topics: List[str] = field(default_factory=list)
    key_metrics: List[str] = field(default_factory=list)
    key_docs: List[str] = field(default_factory=list)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "summary_id": self.summary_id,
            "session_id": self.session_id,
            "summary_text": self.summary_text,
            "covered_turns": self.covered_turns,
            "key_topics": list(self.key_topics),
            "key_metrics": list(self.key_metrics),
            "key_docs": list(self.key_docs),
            "turn_count": self.turn_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSummaryData":
        return cls(
            summary_id=data.get("summary_id", ""),
            session_id=data.get("session_id", ""),
            summary_text=data.get("summary_text", ""),
            covered_turns=data.get("covered_turns", ""),
            key_topics=data.get("key_topics", []),
            key_metrics=data.get("key_metrics", []),
            key_docs=data.get("key_docs", []),
            turn_count=data.get("turn_count", 0),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ConfirmedFact:
    """
    用户确认的事实记录

    用户在对话中确认的事实，持久化到长期记忆，不会随会话过期而消失。

    Attributes:
        fact_id: 事实唯一 ID
        session_id: 确认该事实的会话 ID
        fact_type: 事实类型（如 "regulation", "metric", "definition"）
        fact_content: 事实内容
        source_turn_id: 来源轮次 ID
        confirmed_at: 确认时间戳
    """

    fact_id: str
    session_id: str
    fact_type: str = ""
    fact_content: str = ""
    source_turn_id: str = ""
    confirmed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "session_id": self.session_id,
            "fact_type": self.fact_type,
            "fact_content": self.fact_content,
            "source_turn_id": self.source_turn_id,
            "confirmed_at": self.confirmed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfirmedFact":
        return cls(
            fact_id=data.get("fact_id", ""),
            session_id=data.get("session_id", ""),
            fact_type=data.get("fact_type", ""),
            fact_content=data.get("fact_content", ""),
            source_turn_id=data.get("source_turn_id", ""),
            confirmed_at=data.get("confirmed_at", time.time()),
        )


@dataclass
class MemoryContext:
    """
    完整多轮记忆上下文

    为新问题提供的完整上下文，汇聚三类记忆 + 最近轮次。
    供指代消解和查询重写使用。

    Attributes:
        working_memory: 工作记忆状态（当前意图、实体、约束）
        summary: 最新对话摘要（None 表示尚无摘要）
        confirmed_facts: 用户确认的事实列表
        recent_turns: 最近 n 轮对话记录
    """

    working_memory: Optional[WorkingMemoryState] = None
    summary: Optional[ConversationSummaryData] = None
    confirmed_facts: List[ConfirmedFact] = field(default_factory=list)
    recent_turns: List[Turn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "working_memory": (
                self.working_memory.to_dict() if self.working_memory else None
            ),
            "summary": self.summary.to_dict() if self.summary else None,
            "confirmed_facts": [f.to_dict() for f in self.confirmed_facts],
            "recent_turns": [t.to_dict() for t in self.recent_turns],
        }

    @property
    def mentioned_metrics(self) -> List[str]:
        """合并工作记忆和摘要中的指标"""
        metrics: List[str] = []
        if self.working_memory:
            metrics.extend(self.working_memory.mentioned_metrics)
        if self.summary:
            metrics.extend(self.summary.key_metrics)
        # 去重保序
        seen: set = set()
        result: List[str] = []
        for m in metrics:
            if m and m not in seen:
                seen.add(m)
                result.append(m)
        return result

    @property
    def mentioned_docs(self) -> List[str]:
        """合并工作记忆和摘要中的文档"""
        docs: List[str] = []
        if self.working_memory:
            docs.extend(self.working_memory.mentioned_docs)
        if self.summary:
            docs.extend(self.summary.key_docs)
        seen: set = set()
        result: List[str] = []
        for d in docs:
            if d and d not in seen:
                seen.add(d)
                result.append(d)
        return result

    @property
    def previous_queries(self) -> List[str]:
        """最近轮次的用户查询列表"""
        return [t.query for t in self.recent_turns if t.query]

    @property
    def previous_entities(self) -> List[dict]:
        """最近轮次的实体列表"""
        entities: List[dict] = []
        for t in self.recent_turns:
            entities.extend(t.entities)
        if self.working_memory:
            entities.extend(self.working_memory.current_entities)
        return entities
