"""
状态轨迹增强模块 — ACE-RAG Agent 平台

将状态机的状态轨迹（state_trace）和迁移事件（events）增强为
带中文标签、层级、描述、耗时、元数据的结构化详情，供前端展示和分析。

核心方法:
  enhance_trace(state_trace, events) -> List[dict]
      返回 [{step, state, layer, label, description, duration_ms, metadata}]
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

print("[TraceEnhancer] 状态轨迹增强模块加载中...")

# ============================================================
# 17 个状态的中文标签 + 层级 + 描述
# ============================================================
STATE_TRACE_DETAIL: Dict[str, Dict[str, str]] = {
    "RECEIVED": {
        "layer": "网关层",
        "label": "请求已接收",
        "description": "用户请求到达 Agent 网关",
    },
    "NORMALIZED": {
        "layer": "标准化层",
        "label": "查询已标准化",
        "description": "查询文本清洗、格式规范化",
    },
    "CONTEXT_RESOLVED": {
        "layer": "上下文层",
        "label": "上下文已解析",
        "description": "指代消解、领域上下文检测完成",
    },
    "ANALYZED": {
        "layer": "分析层",
        "label": "查询已分析",
        "description": "意图分类、实体抽取、歧义检测完成",
    },
    "ROUTED": {
        "layer": "路由层",
        "label": "检索策略已确定",
        "description": "根据意图确定检索通道和路由级别",
    },
    "CLARIFYING": {
        "layer": "澄清层",
        "label": "等待用户澄清",
        "description": "检测到歧义，需用户补充信息",
    },
    "PLANNING": {
        "layer": "规划层",
        "label": "检索计划已生成",
        "description": "生成多步检索计划",
    },
    "RETRIEVING": {
        "layer": "检索层",
        "label": "正在执行检索",
        "description": "从知识库检索相关文档片段",
    },
    "TOOL_CALLING": {
        "layer": "工具层",
        "label": "正在调用工具",
        "description": "调用计算器等外部工具",
    },
    "EVIDENCE_ASSEMBLING": {
        "layer": "证据层",
        "label": "证据组装中",
        "description": "将检索结果组装为结构化证据包",
    },
    "EVIDENCE_VALIDATING": {
        "layer": "验证层",
        "label": "证据验证中",
        "description": "验证证据充分性，判断是否足以回答",
    },
    "GENERATING": {
        "layer": "生成层",
        "label": "回答生成中",
        "description": "LLM 基于证据生成回答",
    },
    "ANSWER_VALIDATING": {
        "layer": "校验层",
        "label": "回答校验中",
        "description": "验证回答质量和引用准确性",
    },
    "RESPONDING": {
        "layer": "终态",
        "label": "回答已返回",
        "description": "回答已准备就绪并返回用户",
    },
    "RETRYING": {
        "layer": "重试层",
        "label": "检索重试中",
        "description": "证据不足或回答不合格，正在重试",
    },
    "REFUSING": {
        "layer": "拒答层",
        "label": "拒答处理中",
        "description": "证据不足，生成拒答回复并说明原因",
    },
    "FAILED": {
        "layer": "终态",
        "label": "处理失败",
        "description": "系统内部错误，无法完成处理",
    },
}

# ============================================================
# 检索方式中文标签映射
# ============================================================
STRATEGY_LABELS: Dict[str, str] = {
    "exact": "精确检索",
    "lexical": "词法检索",
    "bm25": "词法检索(BM25)",
    "dense": "语义检索",
    "hybrid": "混合检索",
    "metadata": "元数据过滤",
    "table": "表格检索",
    "neighborhood": "邻域检索",
    "relation": "关系检索",
}


# ============================================================
# 核心方法
# ============================================================
def enhance_trace(
    state_trace: List[str],
    events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    将状态轨迹增强为带详情的结构化列表

    Args:
        state_trace: 状态机轨迹列表（如 ["RECEIVED", "NORMALIZED", ...]）
        events:      状态迁移事件列表，每项形如
                     {from, to, timestamp, metadata, ...}
                     其中 timestamp 为毫秒级时间戳或 ISO 字符串；
                     若提供，将用于计算每步耗时 duration_ms。

    Returns:
        增强后的轨迹列表，每项:
        {
            "step":         序号（从 1 开始）,
            "state":        状态名（如 "RECEIVED"）,
            "layer":        层级（如 "网关层"）,
            "label":        中文标签（如 "请求已接收"）,
            "description":  描述,
            "duration_ms":  该步骤耗时（毫秒，到下一步的时间差）,
            "metadata":     事件元数据
        }
    """
    if not state_trace:
        return []

    events = events or []

    # 构建 state -> event 索引（取该 state 作为 to 的事件）
    # 若 events 没有 to 字段，尝试用 state 列表索引对齐
    event_by_state: Dict[str, Dict[str, Any]] = {}
    event_by_index: Dict[int, Dict[str, Any]] = {}
    for idx, ev in enumerate(events):
        to_state = ev.get("to_state") or ev.get("to") or ev.get("state")
        if to_state:
            # 若同一状态多次出现，保留最后一次
            event_by_state[to_state] = ev
        event_by_index[idx] = ev

    result: List[Dict[str, Any]] = []
    for step_idx, state in enumerate(state_trace, start=1):
        detail = STATE_TRACE_DETAIL.get(state, {
            "layer": "未知层",
            "label": state,
            "description": "未定义的状态",
        })

        # 匹配事件（优先按状态名，其次按索引）
        ev = event_by_state.get(state)
        if ev is None:
            ev = event_by_index.get(step_idx - 1, {})

        metadata = ev.get("metadata", {}) if ev else {}

        # 计算耗时：当前事件 timestamp 与下一事件 timestamp 之差
        duration_ms = _compute_duration_ms(step_idx, ev, events, event_by_index)

        result.append({
            "step": step_idx,
            "state": state,
            "layer": detail["layer"],
            "label": detail["label"],
            "description": detail["description"],
            "duration_ms": duration_ms,
            "metadata": metadata,
        })

    logger.debug("[TraceEnhancer] 增强轨迹: %d 步", len(result))
    return result


def _compute_duration_ms(
    step_idx: int,
    current_ev: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
    event_by_index: Dict[int, Dict[str, Any]],
) -> Optional[float]:
    """
    计算当前步骤到下一步的耗时（毫秒）

    使用事件的 timestamp 字段（支持毫秒级数值或 ISO 字符串）。
    """
    if not current_ev:
        return None

    current_ts = current_ev.get("timestamp")
    if current_ts is None:
        return None

    current_val = _parse_timestamp(current_ts)
    if current_val is None:
        return None

    # 查找下一步事件
    next_ev = event_by_index.get(step_idx)  # step_idx 是从1开始，对应 events 索引为 step_idx
    if next_ev is None:
        # 没有下一步事件，可能是终态
        return None

    next_ts = next_ev.get("timestamp")
    if next_ts is None:
        return None

    next_val = _parse_timestamp(next_ts)
    if next_val is None:
        return None

    diff = next_val - current_val
    return round(diff, 2) if diff >= 0 else None


def _parse_timestamp(ts: Any) -> Optional[float]:
    """
    解析时间戳，统一返回毫秒级数值

    支持:
      - 数值（毫秒或秒）
      - ISO 字符串
    """
    if ts is None:
        return None

    # 数值类型
    if isinstance(ts, (int, float)):
        # 若数值过小（< 1e12），推测为秒级，转为毫秒
        if ts < 1e12:
            return float(ts) * 1000
        return float(ts)

    # 字符串类型：尝试 ISO 解析
    if isinstance(ts, str):
        try:
            from datetime import datetime
            # 兼容带 'Z' 的 ISO 格式
            ts_clean = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_clean)
            return dt.timestamp() * 1000
        except (ValueError, TypeError):
            # 尝试纯数值字符串
            try:
                return _parse_timestamp(float(ts))
            except (ValueError, TypeError):
                return None

    return None


def get_strategy_label(strategy: str) -> str:
    """
    获取检索方式的中文标签

    Args:
        strategy: 检索方式英文标识

    Returns:
        中文标签，未知则返回原值
    """
    return STRATEGY_LABELS.get(strategy, strategy)


def get_state_detail(state: str) -> Dict[str, str]:
    """
    获取单个状态的详情（layer / label / description）

    Args:
        state: 状态名

    Returns:
        详情字典，未知状态返回默认值
    """
    return STATE_TRACE_DETAIL.get(state, {
        "layer": "未知层",
        "label": state,
        "description": "未定义的状态",
    })


print("[TraceEnhancer] 状态轨迹增强模块加载完成（17 个状态 + 9 种检索策略）")
