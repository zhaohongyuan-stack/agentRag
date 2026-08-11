"""
对话摘要生成器 — M5.2 记忆模块

定期压缩对话历史，生成结构化摘要，为长会话提供上下文。

职责:
  1. 每隔 N 轮生成一次对话摘要
  2. 从历史轮次中提取关键主题、指标、文档
  3. 生成自然语言摘要文本
  4. 持久化摘要到 Redis，支持获取最新摘要

摘要策略:
  - 纯规则提取（不依赖 LLM）：从轮次的意图、实体、查询中提取关键信息
  - 结构化输出：summary_text + key_topics + key_metrics + key_docs
  - 增量更新：新摘要覆盖旧摘要，记录涵盖的轮次范围

Redis Key 设计:
  - ace-rag:wm:summary:{session_id} → String(JSON): 最新摘要
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..memory_models import ConversationSummaryData, Turn
from ..session_state.session_manager import _MockRedis

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "ace-rag:wm:summary:"
_DEFAULT_TTL = 7200  # 摘要 TTL: 2 小时（比工作记忆长）
_DEFAULT_SUMMARY_INTERVAL = 5  # 每 5 轮生成一次摘要


class ConversationSummarizer:
    """
    对话摘要生成器

    定期压缩对话历史，生成结构化摘要。

    用法:
        summarizer = ConversationSummarizer()
        # 每 5 轮自动生成
        if turn.turn_number % 5 == 0:
            summary = summarizer.summarize(session_id, turns)
            summarizer.save(session_id, summary)
        # 获取最新摘要
        latest = summarizer.get_latest(session_id)
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        ttl_seconds: int = _DEFAULT_TTL,
        summary_interval: int = _DEFAULT_SUMMARY_INTERVAL,
    ):
        """
        Args:
            redis_client: Redis 客户端实例，None 时使用 _MockRedis
            ttl_seconds: 摘要 TTL（秒）
            summary_interval: 摘要生成间隔（轮次数）
        """
        self._client = redis_client if redis_client is not None else _MockRedis()
        self._ttl = ttl_seconds
        self._interval = summary_interval

    def should_summarize(self, turn_number: int) -> bool:
        """
        判断当前轮次是否需要生成摘要

        Args:
            turn_number: 当前轮次序号

        Returns:
            True 表示需要生成摘要
        """
        return turn_number > 0 and turn_number % self._interval == 0

    def summarize(
        self,
        session_id: str,
        turns: List[Turn],
    ) -> ConversationSummaryData:
        """
        生成对话摘要

        从轮次列表中提取关键主题、指标、文档，生成自然语言摘要。

        纯规则策略（不依赖 LLM）:
          1. 收集所有轮次的意图、查询、实体
          2. 提取关键指标（entity_type=metric_name）
          3. 提取关键文档（entity_type=doc_name）
          4. 提取关键主题（意图分类 + 查询关键词）
          5. 生成摘要文本

        Args:
            session_id: 会话 ID
            turns: 要摘要的对话轮次列表

        Returns:
            ConversationSummaryData 对象
        """
        if not turns:
            return ConversationSummaryData(
                summary_id=f"sum-{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                summary_text="（无对话内容）",
                covered_turns="",
                turn_count=0,
            )

        # 收集关键信息
        key_metrics: List[str] = []
        key_docs: List[str] = []
        key_topics: List[str] = []
        query_summaries: List[str] = []

        for turn in turns:
            # 提取指标和文档
            for entity in turn.entities:
                entity_type = entity.get("entity_type", "")
                entity_value = entity.get("value", "")
                if entity_type == "metric_name" and entity_value:
                    if entity_value not in key_metrics:
                        key_metrics.append(entity_value)
                elif entity_type == "doc_name" and entity_value:
                    if entity_value not in key_docs:
                        key_docs.append(entity_value)

            # 提取主题（基于意图）
            if turn.intent and turn.intent not in key_topics:
                key_topics.append(turn.intent)

            # 构建查询摘要
            query_text = turn.query.strip()
            if query_text:
                query_summaries.append(f"第{turn.turn_number}轮: {query_text}")

        # 生成摘要文本
        summary_parts: List[str] = []

        if key_topics:
            topic_labels = self._intent_to_label(key_topics)
            summary_parts.append(f"对话主题: {', '.join(topic_labels)}")

        if key_metrics:
            summary_parts.append(f"讨论的监管指标: {', '.join(key_metrics)}")

        if key_docs:
            summary_parts.append(f"涉及的法规文档: {', '.join(key_docs)}")

        if query_summaries:
            # 限制摘要长度
            if len(query_summaries) > 5:
                query_summaries = query_summaries[:5] + ["..."]
            summary_parts.append("用户问题: " + " | ".join(query_summaries))

        summary_text = "。".join(summary_parts) if summary_parts else "（无有效摘要内容）"

        # 轮次范围
        turn_numbers = [t.turn_number for t in turns if t.turn_number > 0]
        if turn_numbers:
            covered_turns = f"{min(turn_numbers)}-{max(turn_numbers)}"
        else:
            covered_turns = ""

        summary = ConversationSummaryData(
            summary_id=f"sum-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            summary_text=summary_text,
            covered_turns=covered_turns,
            key_topics=key_topics,
            key_metrics=key_metrics,
            key_docs=key_docs,
            turn_count=len(turns),
            created_at=time.time(),
        )

        logger.info(
            "生成对话摘要: session=%s, turns=%s, metrics=%d, docs=%d",
            session_id,
            covered_turns,
            len(key_metrics),
            len(key_docs),
        )
        return summary

    def save(self, session_id: str, summary: ConversationSummaryData) -> None:
        """
        保存摘要到 Redis

        覆盖旧摘要（每个会话只保留最新摘要）。

        Args:
            session_id: 会话 ID
            summary: 摘要数据
        """
        key = f"{_SUMMARY_PREFIX}{session_id}"
        self._client.set(
            key,
            json.dumps(summary.to_dict(), ensure_ascii=False),
            ex=self._ttl,
        )
        logger.debug("保存摘要: session=%s, id=%s", session_id, summary.summary_id)

    def get_latest(self, session_id: str) -> Optional[ConversationSummaryData]:
        """
        获取最新摘要

        Args:
            session_id: 会话 ID

        Returns:
            ConversationSummaryData，不存在返回 None
        """
        key = f"{_SUMMARY_PREFIX}{session_id}"
        raw = self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return ConversationSummaryData.from_dict(json.loads(raw))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("解析摘要失败: %s", e)
            return None

    def delete(self, session_id: str) -> None:
        """删除会话摘要"""
        self._client.delete(f"{_SUMMARY_PREFIX}{session_id}")

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _intent_to_label(intents: List[str]) -> List[str]:
        """将意图代码转换为中文标签"""
        label_map = {
            "threshold_query": "阈值查询",
            "threshold": "阈值查询",
            "definition_query": "定义查询",
            "definition": "定义查询",
            "table_lookup": "表格取数",
            "clause_query": "条款查询",
            "comparison": "比较查询",
            "factual": "事实查询",
            "procedural": "流程查询",
        }
        return [label_map.get(i, i) for i in intents]
