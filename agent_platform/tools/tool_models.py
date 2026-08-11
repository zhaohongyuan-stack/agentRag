"""
工具平台数据模型 — M5.3 工具模块

定义工具注册和调用的核心数据结构：
  - RetryPolicy: 重试策略
  - ToolManifest: 工具清单（注册契约）
  - ToolResult: 工具调用结果
  - ToolEvent: 工具调用事件日志

设计要点:
  1. ToolManifest 包含完整的工具元信息（Schema、权限、超时、重试、降级）
  2. ToolResult 标准化输出（success/data/error/fallback_used）
  3. ToolEvent 用于审计和可观测性
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetryPolicy:
    """
    重试策略

    Attributes:
        max_retries: 最大重试次数
        backoff_base: 退避基数（秒）
        backoff_max: 最大退避时间（秒）
        retryable_errors: 可重试的错误类型列表
    """

    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 10.0
    retryable_errors: List[str] = field(default_factory=lambda: ["timeout", "connection_error"])

    def to_dict(self) -> dict:
        return {
            "max_retries": self.max_retries,
            "backoff_base": self.backoff_base,
            "backoff_max": self.backoff_max,
            "retryable_errors": list(self.retryable_errors),
        }


@dataclass
class ToolManifest:
    """
    工具清单 — 工具注册契约

    定义工具的完整元信息，包括输入输出 Schema、权限级别、超时、重试策略等。

    Attributes:
        name: 工具名称（唯一标识）
        version: 工具版本
        description: 工具描述
        input_schema: 输入 JSON Schema
        output_schema: 输出 JSON Schema
        capabilities: 能力列表（["read_only", "compute", "external_call"]）
        permission_level: 权限级别（"public", "internal", "restricted"）
        is_read_only: 是否只读
        timeout_ms: 超时时间（毫秒）
        retry_policy: 重试策略
        idempotent: 是否幂等
        concurrency_limit: 并发限制
        cost_level: 成本级别（"low", "medium", "high"）
        allowed_data_scope: 允许的数据范围
        result_trust_level: 结果信任级别（"verified", "unverified"）
        health_check: 健康检查命令
        fallback_tool: 降级工具名称（None 表示无降级）
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    permission_level: str = "public"
    is_read_only: bool = True
    timeout_ms: int = 5000
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    idempotent: bool = True
    concurrency_limit: int = 1
    cost_level: str = "low"
    allowed_data_scope: List[str] = field(default_factory=list)
    result_trust_level: str = "verified"
    health_check: str = ""
    fallback_tool: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "capabilities": list(self.capabilities),
            "permission_level": self.permission_level,
            "is_read_only": self.is_read_only,
            "timeout_ms": self.timeout_ms,
            "retry_policy": self.retry_policy.to_dict(),
            "idempotent": self.idempotent,
            "concurrency_limit": self.concurrency_limit,
            "cost_level": self.cost_level,
            "allowed_data_scope": list(self.allowed_data_scope),
            "result_trust_level": self.result_trust_level,
            "health_check": self.health_check,
            "fallback_tool": self.fallback_tool,
        }


@dataclass
class ToolResult:
    """
    工具调用结果

    Attributes:
        success: 是否成功
        data: 返回数据
        error: 错误信息（失败时）
        fallback_used: 是否使用了降级工具
        execution_time_ms: 执行时间（毫秒）
        retries: 重试次数
        tool_name: 实际执行的工具名称
    """

    success: bool
    data: Any = None
    error: str = ""
    fallback_used: bool = False
    execution_time_ms: float = 0.0
    retries: int = 0
    tool_name: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "fallback_used": self.fallback_used,
            "execution_time_ms": self.execution_time_ms,
            "retries": self.retries,
            "tool_name": self.tool_name,
        }


@dataclass
class ToolEvent:
    """
    工具调用事件日志

    用于审计和可观测性，记录每次工具调用的完整信息。

    Attributes:
        event_id: 事件 ID
        tool_name: 工具名称
        input_data: 输入数据
        output_data: 输出数据
        success: 是否成功
        error: 错误信息
        execution_time_ms: 执行时间
        retries: 重试次数
        fallback_used: 是否使用降级
        caller_id: 调用者 ID
        timestamp: 时间戳
    """

    event_id: str
    tool_name: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Any = None
    success: bool = False
    error: str = ""
    execution_time_ms: float = 0.0
    retries: int = 0
    fallback_used: bool = False
    caller_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "tool_name": self.tool_name,
            "input_data": dict(self.input_data),
            "output_data": self.output_data,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "retries": self.retries,
            "fallback_used": self.fallback_used,
            "caller_id": self.caller_id,
            "timestamp": self.timestamp,
        }
