"""
检查点管理器 — 检查点数据模型

定义执行检查点 Checkpoint 数据结构，用于长流程的故障恢复。

检查点保存 Agent 执行过程中的完整上下文：
  - 状态机当前状态
  - 查询规格与物理计划
  - DAG 任务状态快照
  - 证据包与预算消耗

设计要点:
  1. 自描述: 通过 to_dict / from_dict 完成序列化，存储后端只处理 JSON
  2. 版本化: 每次保存版本号递增，便于回退到历史版本
  3. 轻量依赖: 仅依赖标准库，不与具体存储后端耦合

注意: 本模块的 Checkpoint 是面向持久化的完整执行检查点，与
orchestration/state_machine/machine.py 中的 Checkpoint（轻量内存快照）
定位不同，二者分别服务于「持久化恢复」与「状态机内存恢复」两个层次。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Checkpoint:
    """
    执行检查点 — 保存一次请求执行的完整可恢复快照

    用于在进程崩溃或异常中断后，从检查点恢复状态机、DAG 任务状态、
    证据包与预算消耗，避免从头重新执行已完成的步骤。

    Attributes:
        checkpoint_id: 检查点唯一标识（为空时自动生成）
        session_id: 会话 ID
        request_id: 请求 ID
        state: 当前状态机状态（AgentState 值字符串）
        query_spec: 查询规格（标准化后的查询参数）
        query_plan: 物理执行计划（序列化形式）
        dag_state: DAG 任务状态快照
        evidence_bundle: 证据包
        budget_consumed: 预算消耗记录
        timestamp: 检查点创建时间戳
        version: 检查点版本号
    """

    checkpoint_id: str
    session_id: str
    request_id: str
    state: str                     # 当前状态机状态（AgentState 值）
    query_spec: Dict[str, Any]     # 查询规格
    query_plan: Dict[str, Any]     # 物理计划（序列化）
    dag_state: Optional[Dict[str, Any]] = None  # DAG 任务状态快照
    evidence_bundle: Dict[str, Any] = field(default_factory=dict)  # 证据包
    budget_consumed: Dict[str, Any] = field(default_factory=dict)  # 预算消耗
    timestamp: str = ""
    version: int = 1               # 检查点版本号

    def __post_init__(self):
        """构造后补全默认值：时间戳与检查点 ID"""
        if not self.timestamp:
            self.timestamp = str(time.time())
        if not self.checkpoint_id:
            self.checkpoint_id = f"cp-{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        """
        转换为字典，用于 JSON 序列化与持久化存储

        所有字段均为 JSON 可序列化类型（str/int/dict/list/None），
        存储后端可直接 json.dumps 后写入。
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "state": self.state,
            "query_spec": self.query_spec,
            "query_plan": self.query_plan,
            "dag_state": self.dag_state,
            "evidence_bundle": self.evidence_bundle,
            "budget_consumed": self.budget_consumed,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        """
        从字典构建检查点，用于反序列化

        缺失字段使用安全默认值。若 checkpoint_id / timestamp 缺失，
        __post_init__ 会自动补全，保证对象始终处于可用状态。
        """
        return cls(
            checkpoint_id=data.get("checkpoint_id", ""),
            session_id=data.get("session_id", ""),
            request_id=data.get("request_id", ""),
            state=data.get("state", ""),
            query_spec=data.get("query_spec", {}),
            query_plan=data.get("query_plan", {}),
            dag_state=data.get("dag_state"),
            evidence_bundle=data.get("evidence_bundle", {}),
            budget_consumed=data.get("budget_consumed", {}),
            timestamp=data.get("timestamp", ""),
            version=data.get("version", 1),
        )

    def __repr__(self) -> str:
        return (
            f"Checkpoint(id={self.checkpoint_id}, "
            f"session={self.session_id}, request={self.request_id}, "
            f"state={self.state}, version={self.version})"
        )
