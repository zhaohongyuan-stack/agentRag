"""指标采集模块"""
from .collector import Counter, Gauge, Histogram, MetricsCollector, get_collector

__all__ = ["Counter", "Gauge", "Histogram", "MetricsCollector", "get_collector"]
