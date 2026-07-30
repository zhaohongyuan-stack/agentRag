"""
多轮记忆模块单元测试 — M5.2 记忆模块

测试用例覆盖:
  - 工作记忆更新（意图、实体、约束被更新）
  - 摘要生成（5轮后生成摘要，压缩历史）
  - 事实持久化（用户确认事实写入存储）
  - 指代消解（"这个比例" → 具体指标）
  - 指代不确定（"那个文件" 多文件 → 标记歧义）
  - 上下文恢复（新问题返回最近3轮 + 摘要）
  - TTL 过期（会话超时 → 工作记忆删除，事实保留）

所有测试使用内存模式，无需真实 Redis / 数据库。
异步方法通过 asyncio.run() 同步调用，无需 pytest-asyncio。
"""

import asyncio
import time

import pytest

from agent_platform.memory import (
    ConfirmedFact,
    ConversationSummarizer,
    ConversationSummaryData,
    LongTermMemory,
    MemoryContext,
    MemoryManager,
    MemoryReferenceResolver,
    Turn,
    WorkingMemory,
    WorkingMemoryState,
)


# ============================================================
# 辅助函数
# ============================================================


def make_turn(
    turn_number: int,
    query: str,
    answer: str = "",
    intent: str = "",
    entities=None,
    constraints=None,
    confirmed_facts=None,
) -> Turn:
    """构造测试用 Turn"""
    return Turn(
        turn_id=f"turn-{turn_number:03d}",
        turn_number=turn_number,
        query=query,
        answer=answer,
        intent=intent,
        entities=entities or [],
        constraints=constraints or [],
        user_confirmed_facts=confirmed_facts or [],
    )


# ============================================================
# 工作记忆测试
# ============================================================


class TestWorkingMemory:
    """工作记忆测试"""

    @pytest.fixture
    def wm(self):
        return WorkingMemory()

    def test_update_sets_intent(self, wm):
        """完成一轮 → 意图被更新"""
        turn = make_turn(
            1,
            "核心一级资本充足率最低要求是多少？",
            intent="threshold_query",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"}
            ],
        )
        state = wm.update("sess-1", turn)

        assert state.current_intent == "threshold_query"
        assert state.turn_count == 1

    def test_update_accumulates_entities(self, wm):
        """多轮 → 实体累积（去重）"""
        turn1 = make_turn(
            1,
            "核心一级资本充足率是多少？",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"}
            ],
        )
        turn2 = make_turn(
            2,
            "杠杆率呢？",
            entities=[{"entity_type": "metric_name", "value": "杠杆率"}],
        )

        wm.update("sess-1", turn1)
        state = wm.update("sess-1", turn2)

        assert len(state.current_entities) == 2
        assert state.mentioned_metrics == [
            "核心一级资本充足率",
            "杠杆率",
        ]

    def test_update_accumulates_constraints(self, wm):
        """约束条件累积"""
        turn = make_turn(
            1,
            "2024年有效的核心一级资本充足率",
            constraints=["version=active", "date=2024"],
        )
        state = wm.update("sess-1", turn)

        assert "version=active" in state.active_constraints
        assert "date=2024" in state.active_constraints

    def test_get_nonexistent(self, wm):
        """获取不存在的工作记忆返回 None"""
        assert wm.get("nonexistent") is None

    def test_append_and_get_recent_turns(self, wm):
        """追加事件后可获取最近轮次"""
        for i in range(1, 4):
            turn = make_turn(i, f"问题{i}")
            wm.append_event("sess-1", turn)

        recent = wm.get_recent_turns("sess-1", n=2)
        assert len(recent) == 2
        assert recent[0].turn_number == 2
        assert recent[1].turn_number == 3

    def test_recent_turns_empty(self, wm):
        """无历史时返回空列表"""
        assert wm.get_recent_turns("sess-1", n=3) == []

    def test_delete(self, wm):
        """删除后工作记忆不可获取"""
        turn = make_turn(1, "测试", intent="factual")
        wm.update("sess-1", turn)
        wm.append_event("sess-1", turn)

        wm.delete("sess-1")

        assert wm.get("sess-1") is None
        assert wm.get_recent_turns("sess-1", n=3) == []

    def test_mentioned_docs_accumulate(self, wm):
        """文档名称累积"""
        turn = make_turn(
            1,
            "商业银行资本管理办法的规定",
            entities=[
                {"entity_type": "doc_name", "value": "商业银行资本管理办法"}
            ],
        )
        state = wm.update("sess-1", turn)

        assert "商业银行资本管理办法" in state.mentioned_docs


# ============================================================
# 对话摘要测试
# ============================================================


class TestConversationSummarizer:
    """对话摘要测试"""

    @pytest.fixture
    def summarizer(self):
        return ConversationSummarizer(summary_interval=5)

    def test_should_summarize_at_interval(self, summarizer):
        """第5轮、第10轮需要生成摘要"""
        assert summarizer.should_summarize(5) is True
        assert summarizer.should_summarize(10) is True
        assert summarizer.should_summarize(3) is False
        assert summarizer.should_summarize(0) is False

    def test_summarize_generates_text(self, summarizer):
        """生成摘要包含关键信息"""
        turns = [
            make_turn(
                1,
                "核心一级资本充足率最低是多少？",
                intent="threshold_query",
                entities=[
                    {"entity_type": "metric_name", "value": "核心一级资本充足率"}
                ],
            ),
            make_turn(
                2,
                "商业银行资本管理办法怎么说？",
                intent="clause_query",
                entities=[
                    {"entity_type": "doc_name", "value": "商业银行资本管理办法"}
                ],
            ),
        ]

        summary = summarizer.summarize("sess-1", turns)

        assert summary.summary_text
        assert "核心一级资本充足率" in summary.key_metrics
        assert "商业银行资本管理办法" in summary.key_docs
        assert summary.covered_turns == "1-2"
        assert summary.turn_count == 2

    def test_summarize_empty_turns(self, summarizer):
        """空轮次列表生成占位摘要"""
        summary = summarizer.summarize("sess-1", [])

        assert summary.summary_text
        assert summary.turn_count == 0

    def test_save_and_get_latest(self, summarizer):
        """保存后可获取最新摘要"""
        turns = [make_turn(1, "测试", intent="factual")]
        summary = summarizer.summarize("sess-1", turns)
        summarizer.save("sess-1", summary)

        latest = summarizer.get_latest("sess-1")
        assert latest is not None
        assert latest.summary_id == summary.summary_id
        assert latest.summary_text == summary.summary_text

    def test_get_latest_nonexistent(self, summarizer):
        """获取不存在的摘要返回 None"""
        assert summarizer.get_latest("nonexistent") is None

    def test_delete_summary(self, summarizer):
        """删除后摘要不可获取"""
        summary = summarizer.summarize("sess-1", [make_turn(1, "测试")])
        summarizer.save("sess-1", summary)

        summarizer.delete("sess-1")
        assert summarizer.get_latest("sess-1") is None


# ============================================================
# 长期事实记忆测试
# ============================================================


class TestLongTermMemory:
    """长期事实记忆测试"""

    @pytest.fixture
    def ltm(self):
        return LongTermMemory()

    def test_save_and_get_facts(self, ltm):
        """用户确认事实 → 写入存储 → 可查询"""
        facts = [
            {
                "fact_type": "regulation",
                "fact_content": "核心一级资本充足率不得低于8%",
            },
            {
                "fact_type": "metric",
                "fact_content": "杠杆率不低于4%",
            },
        ]
        saved = asyncio.run(ltm.save("sess-1", facts, source_turn_id="turn-001"))

        assert len(saved) == 2
        assert all(f.fact_id for f in saved)
        assert saved[0].fact_content == "核心一级资本充足率不得低于8%"

        # 查询
        retrieved = asyncio.run(ltm.get("sess-1"))
        assert len(retrieved) == 2

    def test_get_by_type(self, ltm):
        """按类型查询事实"""
        asyncio.run(
            ltm.save(
                "sess-1",
                [
                    {"fact_type": "regulation", "fact_content": "规定A"},
                    {"fact_type": "metric", "fact_content": "指标B"},
                ],
            )
        )

        regulations = asyncio.run(ltm.get_by_type("sess-1", "regulation"))
        assert len(regulations) == 1
        assert regulations[0].fact_content == "规定A"

    def test_delete_facts(self, ltm):
        """删除会话的所有事实"""
        asyncio.run(
            ltm.save(
                "sess-1",
                [{"fact_type": "regulation", "fact_content": "事实A"}],
            )
        )

        count = asyncio.run(ltm.delete("sess-1"))
        assert count == 1

        remaining = asyncio.run(ltm.get("sess-1"))
        assert remaining == []

    def test_facts_persist_across_sessions(self, ltm):
        """事实不随工作记忆过期而消失"""
        asyncio.run(
            ltm.save(
                "sess-1",
                [{"fact_type": "regulation", "fact_content": "持久事实"}],
            )
        )

        # 模拟工作记忆过期（长期记忆独立）
        facts = asyncio.run(ltm.get("sess-1"))
        assert len(facts) == 1
        assert facts[0].fact_content == "持久事实"

    def test_empty_session(self, ltm):
        """空会话查询返回空列表"""
        assert asyncio.run(ltm.get("nonexistent")) == []


# ============================================================
# 指代消解测试
# ============================================================


class TestMemoryReferenceResolver:
    """基于 MemoryContext 的指代消解测试"""

    @pytest.fixture
    def resolver(self):
        return MemoryReferenceResolver()

    def _make_context_with_metric(self, metric: str = "核心一级资本充足率"):
        """构造包含一个指标的上下文"""
        return MemoryContext(
            working_memory=WorkingMemoryState(
                session_id="sess-1",
                mentioned_metrics=[metric],
            ),
            recent_turns=[
                make_turn(1, f"{metric}是多少？"),
            ],
        )

    def _make_context_with_docs(self, docs):
        """构造包含多个文档的上下文"""
        return MemoryContext(
            working_memory=WorkingMemoryState(
                session_id="sess-1",
                mentioned_docs=docs,
            ),
            recent_turns=[],
        )

    def test_resolve_metric_reference(self, resolver):
        """指代消解: '这个比例' → 具体指标"""
        ctx = self._make_context_with_metric("核心一级资本充足率")
        result = resolver.resolve("这个比例适用于非系统重要性银行吗？", ctx)

        assert result.was_resolved is True
        assert result.resolved_entity == "核心一级资本充足率"
        assert "核心一级资本充足率" in result.resolved_query
        assert result.reference_type == "metric"

    def test_resolve_doc_reference(self, resolver):
        """指代消解: '该规定' → 具体文档"""
        ctx = self._make_context_with_docs(["商业银行资本管理办法"])
        result = resolver.resolve("该规定的实施日期是？", ctx)

        assert result.was_resolved is True
        assert result.resolved_entity == "商业银行资本管理办法"

    def test_ambiguous_multiple_metrics(self, resolver):
        """指代不确定: 多个指标 → 标记歧义"""
        ctx = MemoryContext(
            working_memory=WorkingMemoryState(
                session_id="sess-1",
                mentioned_metrics=["核心一级资本充足率", "杠杆率"],
            ),
        )
        result = resolver.resolve("这个比例是多少？", ctx)

        assert result.was_resolved is False
        assert result.ambiguity_flagged is True
        assert "多个候选" in result.ambiguity_reason

    def test_ambiguous_multiple_docs(self, resolver):
        """指代不确定: 多个文档 → 标记歧义"""
        ctx = self._make_context_with_docs(
            ["商业银行资本管理办法", "系统重要性银行评估办法"]
        )
        result = resolver.resolve("那个文件怎么说的？", ctx)

        assert result.was_resolved is False
        assert result.ambiguity_flagged is True

    def test_no_candidate(self, resolver):
        """无候选实体 → 标记无法消解"""
        ctx = MemoryContext()
        result = resolver.resolve("这个比例是多少？", ctx)

        assert result.was_resolved is False
        assert result.ambiguity_flagged is True

    def test_no_reference_in_query(self, resolver):
        """查询中无指代词 → 不消解"""
        ctx = self._make_context_with_metric("核心一级资本充足率")
        result = resolver.resolve("杠杆率是多少？", ctx)

        assert result.was_resolved is False
        assert result.ambiguity_flagged is False

    def test_resolve_from_summary(self, resolver):
        """从摘要中消解指代"""
        ctx = MemoryContext(
            summary=ConversationSummaryData(
                summary_id="sum-1",
                session_id="sess-1",
                key_metrics=["核心一级资本充足率"],
            ),
        )
        result = resolver.resolve("这个比例适用于所有银行吗？", ctx)

        assert result.was_resolved is True
        assert result.source == "summary"

    def test_resolve_from_recent_turns(self, resolver):
        """从最近轮次的实体中消解"""
        ctx = MemoryContext(
            recent_turns=[
                make_turn(
                    1,
                    "核心一级资本充足率是多少？",
                    entities=[
                        {"entity_type": "metric_name", "value": "核心一级资本充足率"}
                    ],
                ),
            ],
        )
        result = resolver.resolve("这个比例怎么计算？", ctx)

        assert result.was_resolved is True
        assert result.source == "recent_turns"
        assert len(result.derived_from_turn_ids) > 0


# ============================================================
# MemoryManager 集成测试
# ============================================================


class TestMemoryManager:
    """MemoryManager 集成测试"""

    @pytest.fixture
    def manager(self):
        return MemoryManager(mock=True, summary_interval=5)

    def test_on_turn_complete_updates_memory(self, manager):
        """完成一轮 → 工作记忆更新"""
        turn = make_turn(
            1,
            "核心一级资本充足率是多少？",
            answer="不得低于8%",
            intent="threshold_query",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"}
            ],
        )

        asyncio.run(manager.on_turn_complete("sess-1", turn))

        state = manager.working_memory.get("sess-1")
        assert state is not None
        assert state.current_intent == "threshold_query"
        assert "核心一级资本充足率" in state.mentioned_metrics

    def test_on_turn_complete_saves_facts(self, manager):
        """完成一轮 → 用户确认事实被保存"""
        turn = make_turn(
            1,
            "确认这个规定",
            confirmed_facts=[
                {"fact_type": "regulation", "fact_content": "核心一级资本充足率不低于8%"}
            ],
        )

        asyncio.run(manager.on_turn_complete("sess-1", turn))

        facts = asyncio.run(manager.long_term_memory.get("sess-1"))
        assert len(facts) == 1
        assert facts[0].fact_content == "核心一级资本充足率不低于8%"

    def test_summary_generated_at_interval(self, manager):
        """第5轮 → 自动生成摘要"""
        for i in range(1, 6):
            turn = make_turn(
                i,
                f"问题{i}",
                intent="threshold_query" if i <= 3 else "definition_query",
                entities=[
                    {"entity_type": "metric_name", "value": "核心一级资本充足率"}
                ],
            )
            asyncio.run(manager.on_turn_complete("sess-1", turn))

        summary = manager.summarizer.get_latest("sess-1")
        assert summary is not None
        assert summary.turn_count > 0

    def test_get_context_for_new_query(self, manager):
        """获取多轮上下文 → 包含最近3轮 + 工作记忆"""
        for i in range(1, 4):
            turn = make_turn(
                i,
                f"问题{i}",
                intent="threshold_query",
                entities=[{"entity_type": "metric_name", "value": f"指标{i}"}],
            )
            asyncio.run(manager.on_turn_complete("sess-1", turn))

        ctx = asyncio.run(manager.get_context_for_new_query("sess-1", recent_n=3))

        assert ctx.working_memory is not None
        assert len(ctx.recent_turns) == 3
        assert ctx.working_memory.turn_count == 3

    def test_get_context_empty_session(self, manager):
        """空会话 → 上下文为空"""
        ctx = asyncio.run(manager.get_context_for_new_query("nonexistent"))

        assert ctx.working_memory is None
        assert ctx.summary is None
        assert ctx.recent_turns == []
        assert ctx.confirmed_facts == []

    def test_delete_session_preserves_facts(self, manager):
        """删除会话 → 工作记忆删除，长期事实保留"""
        turn = make_turn(
            1,
            "确认事实",
            confirmed_facts=[
                {"fact_type": "regulation", "fact_content": "持久事实"}
            ],
        )
        asyncio.run(manager.on_turn_complete("sess-1", turn))

        manager.delete_session("sess-1")

        # 工作记忆已删除
        assert manager.working_memory.get("sess-1") is None
        # 长期事实保留
        facts = asyncio.run(manager.long_term_memory.get("sess-1"))
        assert len(facts) == 1

    def test_purge_session_removes_all(self, manager):
        """彻底清除 → 包括长期事实"""
        turn = make_turn(
            1,
            "确认事实",
            confirmed_facts=[
                {"fact_type": "regulation", "fact_content": "待清除"}
            ],
        )
        asyncio.run(manager.on_turn_complete("sess-1", turn))

        asyncio.run(manager.purge_session("sess-1"))

        assert manager.working_memory.get("sess-1") is None
        facts = asyncio.run(manager.long_term_memory.get("sess-1"))
        assert facts == []

    def test_full_multi_turn_flow(self, manager):
        """完整多轮流程: 3轮对话 + 指代消解"""
        # 第1轮: 提到指标
        turn1 = make_turn(
            1,
            "核心一级资本充足率最低要求是多少？",
            answer="不得低于8%",
            intent="threshold_query",
            entities=[
                {"entity_type": "metric_name", "value": "核心一级资本充足率"}
            ],
        )
        asyncio.run(manager.on_turn_complete("sess-1", turn1))

        # 第2轮: 指代消解
        ctx = asyncio.run(manager.get_context_for_new_query("sess-1"))
        resolver = MemoryReferenceResolver()
        result = resolver.resolve("这个比例适用于非系统重要性银行吗？", ctx)

        assert result.was_resolved is True
        assert result.resolved_entity == "核心一级资本充足率"

        # 第2轮完成
        turn2 = make_turn(
            2,
            result.resolved_query,
            answer="是的，适用于所有商业银行",
            intent="definition_query",
        )
        asyncio.run(manager.on_turn_complete("sess-1", turn2))

        # 验证上下文
        ctx2 = asyncio.run(manager.get_context_for_new_query("sess-1"))
        assert len(ctx2.recent_turns) == 2
        assert ctx2.mentioned_metrics == ["核心一级资本充足率"]


# ============================================================
# MemoryContext 属性测试
# ============================================================


class TestMemoryContext:
    """MemoryContext 数据聚合测试"""

    def test_mentioned_metrics_merge(self):
        """合并工作记忆和摘要中的指标"""
        ctx = MemoryContext(
            working_memory=WorkingMemoryState(
                session_id="s1",
                mentioned_metrics=["指标A"],
            ),
            summary=ConversationSummaryData(
                summary_id="sum-1",
                session_id="s1",
                key_metrics=["指标A", "指标B"],
            ),
        )
        assert ctx.mentioned_metrics == ["指标A", "指标B"]

    def test_mentioned_docs_merge(self):
        """合并工作记忆和摘要中的文档"""
        ctx = MemoryContext(
            working_memory=WorkingMemoryState(
                session_id="s1",
                mentioned_docs=["文档1"],
            ),
            summary=ConversationSummaryData(
                summary_id="sum-1",
                session_id="s1",
                key_docs=["文档2"],
            ),
        )
        assert ctx.mentioned_docs == ["文档1", "文档2"]

    def test_previous_queries(self):
        """从最近轮次提取查询"""
        ctx = MemoryContext(
            recent_turns=[
                make_turn(1, "问题1"),
                make_turn(2, "问题2"),
            ],
        )
        assert ctx.previous_queries == ["问题1", "问题2"]

    def test_empty_context(self):
        """空上下文属性为空"""
        ctx = MemoryContext()
        assert ctx.mentioned_metrics == []
        assert ctx.mentioned_docs == []
        assert ctx.previous_queries == []
        assert ctx.previous_entities == []
