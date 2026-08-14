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
    state_trace_detail: Optional[List[Dict[str, Any]]] = Field(
        None, description="状态轨迹增强详情（含层级、标签、描述、耗时）"
    )

    # ── Phase 5/6: Agent 协作可视化数据 ──
    loop_count: int = Field(0, description="检索-评估 Loop 轮次（0=单轮）")
    agent_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Agent 决策列表，每项 {agent, decision, latency_ms, detail, round}",
    )


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = "ok"
    service: str = "agent-platform"
    version: str = "1.0.0-phase1"
    docs: Optional[int] = None
    chunks: Optional[int] = None


# ============================================================
# 认证相关模型
# ============================================================
class LoginRequest(BaseModel):
    """登录请求"""

    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class LoginResponse(BaseModel):
    """登录响应"""

    token: str = Field(..., description="JWT token")
    user: Dict[str, Any] = Field(..., description="用户信息")


# ============================================================
# 对话持久化相关模型
# ============================================================
class ConversationCreate(BaseModel):
    """创建对话记录请求"""

    session_id: str = Field(..., description="会话 ID")
    query: str = Field(..., description="用户原始问题")
    response: Dict[str, Any] = Field(..., description="QueryResponse 的字典形式")


class ReviewCreate(BaseModel):
    """提交审查请求"""

    review_status: str = Field(
        ...,
        description="审查状态: correct / incorrect / needs_improvement",
    )
    review_comment: str = Field(
        ..., min_length=10, description="审查评论（最少 10 字符）"
    )


class SuggestionCreate(BaseModel):
    """添加改进建议请求"""

    suggestion_text: str = Field(
        ..., min_length=1, description="建议内容"
    )


class QARequest(BaseModel):
    """管理员质检标注请求"""

    qa_result: str = Field(
        ..., description="质检结果: pass / fail / rework"
    )
    qa_comment: str = Field("", description="质检评论")
