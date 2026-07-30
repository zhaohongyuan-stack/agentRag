"""
结构化 JSON 日志格式化器 — M5.5 可观测性模块

输出 JSON 格式日志，包含 request_id、span_id、timestamp 等字段。
支持日志重放（从日志重建完整决策过程）。

日志格式:
  {"timestamp": "2024-01-01T12:00:00", "level": "INFO",
   "request_id": "req-123", "span_id": "span-456",
   "module": "agent_platform.gateway", "message": "...",
   "extra": {...}}
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional


class StructuredLogFormatter(logging.Formatter):
    """
    结构化 JSON 日志格式化器

    将日志记录格式化为 JSON，包含标准字段和自定义字段。

    用法:
        formatter = StructuredLogFormatter()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(record.created),
            ),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # 从 LogRecord 的自定义属性中提取
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "span_id"):
            log_entry["span_id"] = record.span_id
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id

        # extra 字段
        extra = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process",
                "request_id", "span_id", "session_id",
                "message", "taskName",
            }:
                try:
                    json.dumps(value)
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            log_entry["extra"] = extra

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class LogReplayer:
    """
    日志重放器

    从结构化日志中重建完整决策过程。

    用法:
        replayer = LogReplayer()
        replayer.load(log_lines)
        trace = replayer.rebuild_trace()
    """

    def __init__(self):
        self._entries: list = []

    def load(self, log_lines) -> int:
        """
        加载日志行

        Args:
            log_lines: 日志行列表（每行为 JSON 字符串）

        Returns:
            成功加载的条目数
        """
        count = 0
        for line in log_lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                self._entries.append(entry)
                count += 1
            except json.JSONDecodeError:
                continue
        return count

    def load_from_string(self, log_string: str) -> int:
        """从多行字符串加载"""
        return self.load(log_string.strip().split("\n"))

    def filter_by_request(self, request_id: str) -> list:
        """按 request_id 过滤"""
        return [
            e for e in self._entries
            if e.get("request_id") == request_id
        ]

    def rebuild_trace(self) -> Dict[str, Any]:
        """
        重建追踪树

        从日志条目中重建 Span 树结构。

        Returns:
            追踪树字典
        """
        spans_by_id: Dict[str, Dict] = {}
        root_span = None

        for entry in self._entries:
            span_id = entry.get("span_id")
            if not span_id:
                continue

            if span_id not in spans_by_id:
                spans_by_id[span_id] = {
                    "span_id": span_id,
                    "name": entry.get("module", ""),
                    "events": [],
                    "status": "ok",
                }

            span = spans_by_id[span_id]

            # 记录事件
            span["events"].append({
                "timestamp": entry.get("timestamp"),
                "level": entry.get("level"),
                "message": entry.get("message"),
            })

            if entry.get("level") == "ERROR":
                span["status"] = "error"

            # 从 extra 中提取 span 信息
            extra = entry.get("extra", {})
            if "span_name" in extra:
                span["name"] = extra["span_name"]
            if "parent_id" in extra and "parent_id" not in span:
                span["parent_id"] = extra["parent_id"]
                if extra["parent_id"] is None:
                    root_span = span_id

        # 构建树
        children_map: Dict[str, list] = {}
        for sid, span in spans_by_id.items():
            parent_id = span.get("parent_id")
            if parent_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(sid)

        def build_tree(span_id: str) -> dict:
            span = dict(spans_by_id[span_id])
            span["children"] = [
                build_tree(sid) for sid in children_map.get(span_id, [])
            ]
            return span

        if root_span:
            return build_tree(root_span)
        elif spans_by_id:
            # 无明确 root，返回第一个
            return build_tree(list(spans_by_id.keys())[0])
        return {}

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get_entries(self) -> list:
        return list(self._entries)


def generate_request_id() -> str:
    """生成 request_id"""
    return f"req-{uuid.uuid4().hex[:8]}"


def set_request_context(
    logger: logging.Logger,
    request_id: Optional[str] = None,
    span_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    设置请求上下文日志适配器

    用法:
        adapter = set_request_context(logger, request_id="req-123")
        adapter.info("处理请求")
    """
    extra = {}
    if request_id:
        extra["request_id"] = request_id
    if span_id:
        extra["span_id"] = span_id
    if session_id:
        extra["session_id"] = session_id

    return logging.LoggerAdapter(logger, extra)
