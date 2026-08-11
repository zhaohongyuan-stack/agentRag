"""
轻量级链路追踪器 — M5.5 可观测性模块

不依赖 OpenTelemetry，实现自定义 Span 树追踪。
支持嵌套 Span、属性标注、时间记录。

Span 树结构:
  QueryRequest (root span)
  ├── QueryUnderstanding
  │   ├── IntentClassification
  │   └── EntityExtraction
  ├── Retrieval
  │   ├── ExactRetrieval
  │   └── DenseRetrieval
  ├── Generation
  └── Verification
      └── NumericValidation
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """
    追踪 Span

    Attributes:
        span_id: Span 唯一 ID
        parent_id: 父 Span ID（None 为根 Span）
        name: Span 名称
        start_time: 开始时间戳
        end_time: 结束时间戳（None 表示未结束）
        attributes: 属性字典
        events: 事件列表
        status: 状态（ok / error）
        children: 子 Span 列表
    """

    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    children: List["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """持续时间（毫秒）"""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attributes) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes,
        })

    def set_error(self, error: str) -> None:
        self.status = "error"
        self.add_event("error", message=error)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "attributes": dict(self.attributes),
            "events": list(self.events),
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """
    链路追踪器

    用法:
        tracer = Tracer()
        with tracer.start_span("QueryRequest") as root:
            root.set_attribute("query", "核心一级资本充足率")
            with tracer.start_span("Retrieval", parent=root) as retrieval:
                retrieval.set_attribute("hits", 5)
            with tracer.start_span("Verification", parent=root) as verify:
                verify.set_attribute("passed", True)

        trace = tracer.finish()
        # trace 包含完整的 Span 树
    """

    def __init__(self):
        self._spans: Dict[str, Span] = {}
        self._root_id: Optional[str] = None
        self._stack: List[str] = []  # 当前 Span 栈

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        **attributes,
    ) -> Span:
        """
        开始一个新 Span

        Args:
            name: Span 名称
            parent: 父 Span（None 时使用栈顶或创建根 Span）
            **attributes: 初始属性

        Returns:
            Span 对象（需在结束后调用 end_span）
        """
        span_id = f"span-{uuid.uuid4().hex[:8]}"

        if parent is not None:
            parent_id = parent.span_id
        elif self._stack:
            parent_id = self._stack[-1]
        else:
            parent_id = None

        span = Span(
            span_id=span_id,
            parent_id=parent_id,
            name=name,
        )

        for k, v in attributes.items():
            span.set_attribute(k, v)

        self._spans[span_id] = span
        self._stack.append(span_id)

        if parent_id is None:
            self._root_id = span_id
        elif parent_id in self._spans:
            self._spans[parent_id].children.append(span)

        logger.debug("开始 Span: %s (parent=%s)", name, parent_id)
        return span

    def end_span(self, span: Span) -> None:
        """结束 Span"""
        span.end_time = time.time()
        if span.span_id in self._stack:
            self._stack.remove(span.span_id)
        logger.debug(
            "结束 Span: %s (%.1fms)", span.name, span.duration_ms
        )

    def finish(self) -> Optional[Span]:
        """
        完成追踪，返回根 Span（包含完整子树）

        Returns:
            根 Span，无追踪时返回 None
        """
        if self._root_id is None:
            return None
        return self._spans.get(self._root_id)

    def get_span(self, span_id: str) -> Optional[Span]:
        """获取指定 Span"""
        return self._spans.get(span_id)

    def get_all_spans(self) -> List[Span]:
        """获取所有 Span（平铺）"""
        return list(self._spans.values())

    def to_dict(self) -> dict:
        """导出完整追踪树"""
        root = self.finish()
        return root.to_dict() if root else {}


class SpanContext:
    """Span 上下文管理器（支持 with 语法）"""

    def __init__(self, tracer: Tracer, span: Span):
        self._tracer = tracer
        self._span = span

    def __enter__(self) -> Span:
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._span.set_error(str(exc_val))
        self._tracer.end_span(self._span)
        return False
