"""
记忆模块 — 会话状态、对话摘要、工作记忆、长期记忆

M5.2 完整多轮记忆系统:
  - WorkingMemory: 工作记忆（当前意图、实体、约束、任务进度）
  - ConversationSummarizer: 对话摘要生成（定期压缩）
  - LongTermMemory: 长期事实记忆（用户确认的事实，持久化）
  - MemoryManager: 集成管理器（统一接口）
  - MemoryReferenceResolver: 基于记忆上下文的增强指代消解
  - MemoryContext: 完整多轮上下文
"""

from .conversation_summary.summarizer import ConversationSummarizer
from .long_term_memory.store import LongTermMemory
from .manager import MemoryManager
from .memory_models import (
    ConfirmedFact,
    ConversationSummaryData,
    MemoryContext,
    Turn,
    WorkingMemoryState,
)
from .reference_resolver.resolver import (
    EnhancedResolutionResult,
    MemoryReferenceResolver,
)
from .working_memory.manager import WorkingMemory

__all__ = [
    # 数据模型
    "Turn",
    "WorkingMemoryState",
    "ConversationSummaryData",
    "ConfirmedFact",
    "MemoryContext",
    # 子模块
    "WorkingMemory",
    "ConversationSummarizer",
    "LongTermMemory",
    # 集成管理器
    "MemoryManager",
    # 指代消解
    "MemoryReferenceResolver",
    "EnhancedResolutionResult",
]
