"""
轻量级指标采集器 — M5.5 可观测性模块

不依赖 Prometheus 客户端库，实现自定义指标采集。
支持 Counter（计数器）、Histogram（直方图）、Gauge（仪表盘）三种指标类型。

核心指标:
  - 延迟: agent_query_latency, retrieval_latency, generation_latency
  - 计数: agent_queries_total, agent_refusals_total, retrieval_calls_total
  - 质量: evidence_sufficiency_score, claim_coverage_ratio
  - 预算: budget_consumed_ratio
  - 缓存: plan_cache_hit_rate
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Counter:
    """计数器 — 只增不减"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = {}  # labels_key → value
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels) -> None:
        labels_key = self._labels_key(labels)
        with self._lock:
            self._values[labels_key] = self._values.get(labels_key, 0.0) + value

    def get(self, **labels) -> float:
        labels_key = self._labels_key(labels)
        return self._values.get(labels_key, 0.0)

    def get_all(self) -> Dict[str, float]:
        return dict(self._values)

    @staticmethod
    def _labels_key(labels: Dict) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))


class Histogram:
    """直方图 — 记录值分布"""

    DEFAULT_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[float, int] = {b: 0 for b in self._buckets}
        self._count = 0
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            for b in self._buckets:
                if value <= b:
                    self._counts[b] += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def avg(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def get_buckets(self) -> Dict[float, int]:
        return dict(self._counts)

    def percentile(self, p: float) -> float:
        """近似百分位值"""
        if self._count == 0:
            return 0.0
        target = p * self._count
        cumulative = 0
        for b in self._buckets:
            cumulative += self._counts[b]
            if cumulative >= target:
                return b
        return self._buckets[-1]


class Gauge:
    """仪表盘 — 可增可减"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        with self._lock:
            self._value -= value

    @property
    def value(self) -> float:
        return self._value


class MetricsCollector:
    """
    指标采集器

    统一管理所有指标，提供注册和查询接口。

    用法:
        collector = MetricsCollector()
        collector.counter("queries_total").inc()
        collector.histogram("latency").observe(0.5)
        collector.gauge("cache_hit_rate").set(0.85)
    """

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._lock = threading.RLock()  # 可重入锁，避免 reset → _init_default_metrics 死锁
        self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """初始化默认指标"""
        # 延迟
        self.histogram("agent_query_latency_seconds", "Agent 查询延迟")
        self.histogram("retrieval_latency_seconds", "检索延迟")
        self.histogram("generation_latency_seconds", "生成延迟")

        # 计数
        self.counter("agent_queries_total", "查询总数")
        self.counter("agent_queries_by_path_total", "按路径查询数")
        self.counter("agent_refusals_total", "拒答总数")
        self.counter("agent_retries_total", "重试总数")
        self.counter("retrieval_calls_total", "检索调用总数")
        self.counter("llm_calls_total", "LLM 调用总数")

        # 质量
        self.histogram("evidence_sufficiency_score", "证据充分性评分")
        self.histogram("claim_coverage_ratio", "声明覆盖率")

        # 预算
        self.gauge("budget_consumed_ratio", "预算消耗比例")
        self.counter("dag_tasks_completed", "DAG 任务完成数")
        self.counter("dag_tasks_failed", "DAG 任务失败数")

        # 缓存
        self.gauge("plan_cache_hit_rate", "计划缓存命中率")
        self.gauge("evidence_cache_hit_rate", "证据缓存命中率")

    def counter(self, name: str, description: str = "") -> Counter:
        """获取或创建计数器"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def histogram(
        self, name: str, description: str = "", buckets=None
    ) -> Histogram:
        """获取或创建直方图"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets)
            return self._histograms[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """获取或创建仪表盘"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def export(self) -> Dict[str, Any]:
        """导出所有指标（类似 Prometheus /metrics 格式）"""
        result = {}
        for name, c in self._counters.items():
            result[name] = {
                "type": "counter",
                "values": c.get_all(),
            }
        for name, h in self._histograms.items():
            result[name] = {
                "type": "histogram",
                "count": h.count,
                "sum": h.sum,
                "avg": round(h.avg, 4),
                "buckets": h.get_buckets(),
            }
        for name, g in self._gauges.items():
            result[name] = {
                "type": "gauge",
                "value": g.value,
            }
        return result

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._init_default_metrics()


# 全局单例
_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """获取全局指标采集器"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
