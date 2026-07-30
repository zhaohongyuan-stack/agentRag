"""
请求/响应 Pydantic 模型

定义 Agent 平台的 HTTP API 入口和出口数据结构。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """用户查询请求"""

    query: str = Field(..., description="用户原始问题")
    session_id: Optional[str] = Field(None, description="会话 ID，为空时创建新会话")
    idempotency_key: Optional[str] = Field(None, description="幂等键，防止重复请求")


class QueryResponse(BaseModel):
    """查询响应"""

    request_id: str = Field(..., description="请求唯一标识")
    session_id: str = Field(..., description="会话 ID")
    answer: str = Field(..., description="回答文本")
    citations: List[Dict[str, str]] = Field(default_factory=list, description="引用列表")
    intent: str = Field(..., description="识别的意图")
    complexity: str = Field(..., description="复杂度级别")
    is_refusal: bool = Field(False, description="是否为拒答")
    refusal_reason: Optional[str] = Field(None, description="拒答原因")
    confidence: float = Field(0.0, description="回答置信度")
    state_trace: List[str] = Field(default_factory=list, description="状态机轨迹")
    evidence_count: int = Field(0, description="证据数量")
    sufficiency_score: float = Field(0.0, description="证据充分性评分")
    latency_ms: float = Field(0.0, description="总延迟（毫秒）")
    claims_with_evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="声明-证据对齐"
    )
    ambiguities: List[Dict[str, Any]] = Field(
        default_factory=list, description="检测到的歧义"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = "ok"
    service: str = "agent-platform"
    version: str = "1.0.0-phase1"
