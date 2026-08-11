"""
早停评估器子模块 — 证据增量评估与早停判断

提供 EarlyStopEvaluator（早停评估器）与 EarlyStopResult（评估结果），
基于证据充分性、边际增益与预算状态判断是否应提前停止检索。
"""

from .evaluator import EarlyStopEvaluator, EarlyStopResult

__all__ = ["EarlyStopEvaluator", "EarlyStopResult"]
