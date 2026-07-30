"""
Agent 状态机 — 状态枚举与合法迁移表

状态定义来源: contracts/enums/agent_states.yaml
迁移规则来源: 架构设计.md 状态机定义

状态机设计原则:
  1. 确定性规则: 状态迁移必须查表校验，LLM 的 next_action 需调度器验证
  2. 检查点恢复: 所有长流程都能从检查点恢复
  3. 事件记录: 每次迁移都记录事件（from/to/timestamp/metadata）
"""

from enum import Enum
from typing import Dict, List, Set


class AgentState(str, Enum):
    """Agent 状态枚举 — 与 contracts/enums/agent_states.yaml 完全一致"""

    RECEIVED = "RECEIVED"
    NORMALIZED = "NORMALIZED"
    CONTEXT_RESOLVED = "CONTEXT_RESOLVED"
    ANALYZED = "ANALYZED"
    ROUTED = "ROUTED"
    CLARIFYING = "CLARIFYING"
    PLANNING = "PLANNING"
    RETRIEVING = "RETRIEVING"
    TOOL_CALLING = "TOOL_CALLING"
    EVIDENCE_ASSEMBLING = "EVIDENCE_ASSEMBLING"
    EVIDENCE_VALIDATING = "EVIDENCE_VALIDATING"
    GENERATING = "GENERATING"
    ANSWER_VALIDATING = "ANSWER_VALIDATING"
    RESPONDING = "RESPONDING"
    RETRYING = "RETRYING"
    REFUSING = "REFUSING"
    FAILED = "FAILED"


# ============================================================
# 合法状态迁移表
# 来源: contracts/enums/agent_states.yaml transitions
# ============================================================
TRANSITIONS: Dict[AgentState, List[AgentState]] = {
    AgentState.RECEIVED: [AgentState.NORMALIZED],
    AgentState.NORMALIZED: [AgentState.CONTEXT_RESOLVED],
    AgentState.CONTEXT_RESOLVED: [AgentState.ANALYZED],
    AgentState.ANALYZED: [AgentState.ROUTED],
    AgentState.ROUTED: [
        AgentState.CLARIFYING,
        AgentState.PLANNING,
        AgentState.RETRIEVING,
        AgentState.TOOL_CALLING,
        AgentState.RESPONDING,  # L0 直接回复（问候/能力说明）
    ],
    AgentState.CLARIFYING: [AgentState.RECEIVED, AgentState.FAILED],
    AgentState.PLANNING: [AgentState.RETRIEVING],
    AgentState.RETRIEVING: [AgentState.EVIDENCE_ASSEMBLING],
    AgentState.TOOL_CALLING: [AgentState.RETRIEVING, AgentState.EVIDENCE_ASSEMBLING],
    AgentState.EVIDENCE_ASSEMBLING: [AgentState.EVIDENCE_VALIDATING],
    AgentState.EVIDENCE_VALIDATING: [
        AgentState.GENERATING,
        AgentState.RETRIEVING,
        AgentState.REFUSING,
    ],
    AgentState.GENERATING: [AgentState.ANSWER_VALIDATING],
    AgentState.ANSWER_VALIDATING: [
        AgentState.RESPONDING,
        AgentState.RETRYING,
        AgentState.REFUSING,
    ],
    AgentState.RETRYING: [AgentState.RETRIEVING, AgentState.GENERATING],
    AgentState.RESPONDING: [],  # 终态
    AgentState.REFUSING: [AgentState.RESPONDING],
    AgentState.FAILED: [],  # 终态
}

# 终态集合
TERMINAL_STATES: Set[AgentState] = {
    AgentState.RESPONDING,
    AgentState.FAILED,
}

# 状态中文标签（用于日志和 UI 展示）
STATE_LABELS: Dict[AgentState, str] = {
    AgentState.RECEIVED: "已接收",
    AgentState.NORMALIZED: "已标准化",
    AgentState.CONTEXT_RESOLVED: "上下文已解析",
    AgentState.ANALYZED: "已分析",
    AgentState.ROUTED: "已路由",
    AgentState.CLARIFYING: "澄清中",
    AgentState.PLANNING: "规划中",
    AgentState.RETRIEVING: "检索中",
    AgentState.TOOL_CALLING: "工具调用中",
    AgentState.EVIDENCE_ASSEMBLING: "证据组装中",
    AgentState.EVIDENCE_VALIDATING: "证据验证中",
    AgentState.GENERATING: "生成中",
    AgentState.ANSWER_VALIDATING: "回答验证中",
    AgentState.RESPONDING: "回复中",
    AgentState.RETRYING: "重试中",
    AgentState.REFUSING: "拒答中",
    AgentState.FAILED: "失败",
}


def is_valid_transition(current: AgentState, target: AgentState) -> bool:
    """检查状态迁移是否合法"""
    if current in TERMINAL_STATES:
        return False
    allowed = TRANSITIONS.get(current, [])
    return target in allowed


def get_valid_targets(state: AgentState) -> List[AgentState]:
    """获取某状态的合法目标状态列表"""
    return TRANSITIONS.get(state, [])


def is_terminal(state: AgentState) -> bool:
    """判断是否为终态"""
    return state in TERMINAL_STATES
