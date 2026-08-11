"""
推测式检索单元测试（M4.3）

测试用例（对应开发计划测试用例表）:
  - T0 启动: 精确+词法 → T0 分支启动
  - T1 延迟: Dense → tier=1
  - T2 条件: table/relation → tier=2
  - 取消分支: 精确找到唯一条款 → Dense 被取消
  - 早停-充分性: score=0.9 → should_stop=True
  - 早停-边际增益: gain<0.05 → should_stop=True
  - 早停-预算: budget_exhausted → should_stop=True
  - 结果收集: 多分支结果 → 按 score 降序
  - 结果去重: 相同 chunk_id → 去重保留最高分
  - 空分支: 无结果 → 正常处理

额外测试用例:
  - 分支状态枚举值
  - 分层启动 launch_tiered 分组
  - 检索函数异常分支标记 FAILED
  - 结果按通道过滤
  - 分支取消-证据充分取消全部
  - 分支取消-相同父条款取消 table
"""

import pytest

from agent_platform.speculative_retrieval import (
    BranchCanceller,
    BranchStatus,
    EarlyStopEvaluator,
    EarlyStopResult,
    RetrievalBranch,
    ResultStream,
    SpeculativeLauncher,
)


# ============================================================
# 测试常量（与 SpeculativeLauncher 类属性对齐）
# ============================================================

T0_CHANNELS = ["exact", "lexical", "metadata"]
T1_CHANNELS = ["dense"]
T2_CHANNELS = ["table", "relation"]


# ============================================================
# 工厂函数
# ============================================================


def make_result(chunk_id="c1", score=0.5, **extra):
    """构造单条检索结果"""
    r = {"chunk_id": chunk_id, "score": score}
    r.update(extra)
    return r


def make_branch(
    channel="exact",
    tier=0,
    status=BranchStatus.COMPLETED,
    results=None,
    score=None,
    branch_id=None,
):
    """构造检索分支，score 缺省时取结果最高分"""
    results = results if results is not None else []
    if score is None:
        scores = [
            float(r.get("score", 0.0)) for r in results if r.get("score") is not None
        ]
        score = max(scores) if scores else 0.0
    return RetrievalBranch(
        branch_id=branch_id or f"{channel}-test0001",
        channel=channel,
        tier=tier,
        status=status,
        results=results,
        score=score,
    )


def channel_retrieval_func(channel_to_results):
    """构造按通道返回结果的检索函数"""

    def _func(channel):
        return list(channel_to_results.get(channel, []))

    return _func


# ============================================================
# SpeculativeLauncher 测试
# ============================================================


class TestBranchStatus:
    """分支状态枚举测试"""

    def test_status_values(self):
        """BranchStatus 枚举值正确"""
        assert BranchStatus.LAUNCHED.value == "launched"
        assert BranchStatus.RUNNING.value == "running"
        assert BranchStatus.COMPLETED.value == "completed"
        assert BranchStatus.CANCELLED.value == "cancelled"
        assert BranchStatus.FAILED.value == "failed"


class TestSpeculativeLauncher:
    """推测式检索启动器测试"""

    def test_t0_launch_exact_lexical(self):
        """T0 启动: 精确+词法 → T0 分支启动"""
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["exact", "lexical"])
        assert len(branches) == 2
        for b in branches:
            assert b.tier == 0
            assert b.channel in ("exact", "lexical")
            # 无检索函数 → 保持 LAUNCHED，结果为空
            assert b.status == BranchStatus.LAUNCHED
            assert b.results == []

    def test_t1_dense_tier(self):
        """T1 延迟: Dense → tier=1"""
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["dense"])
        assert len(branches) == 1
        assert branches[0].channel == "dense"
        assert branches[0].tier == 1

    def test_t2_table_relation_tier(self):
        """T2 条件: table/relation → tier=2"""
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["table", "relation"])
        assert len(branches) == 2
        assert {b.channel for b in branches} == {"table", "relation"}
        for b in branches:
            assert b.tier == 2

    def test_all_channels_tier_alignment(self):
        """各通道 tier 与类属性定义一致"""
        launcher = SpeculativeLauncher()
        all_channels = T0_CHANNELS + T1_CHANNELS + T2_CHANNELS
        branches = launcher.launch(all_channels)
        for b in branches:
            if b.channel in T0_CHANNELS:
                assert b.tier == 0
            elif b.channel in T1_CHANNELS:
                assert b.tier == 1
            elif b.channel in T2_CHANNELS:
                assert b.tier == 2

    def test_unknown_channel_defaults_to_t1(self):
        """未配置的通道默认归入 T1"""
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["unknown_channel"])
        assert branches[0].tier == 1

    def test_launch_with_retrieval_func_completes(self):
        """提供检索函数 → 分支 COMPLETED 并携带结果与得分"""
        launcher = SpeculativeLauncher()
        results = {
            "exact": [make_result("c1", 0.9), make_result("c2", 0.7)],
        }
        branches = launcher.launch(
            ["exact", "lexical"], channel_retrieval_func(results)
        )
        exact = next(b for b in branches if b.channel == "exact")
        lexical = next(b for b in branches if b.channel == "lexical")
        # exact 有结果
        assert exact.status == BranchStatus.COMPLETED
        assert len(exact.results) == 2
        assert exact.score == pytest.approx(0.9)  # max score
        assert exact.started_at is not None
        assert exact.completed_at is not None
        # lexical 无结果但已执行完成
        assert lexical.status == BranchStatus.COMPLETED
        assert lexical.results == []
        assert lexical.score == 0.0

    def test_launch_retrieval_func_failure_marks_failed(self):
        """检索函数抛异常 → 分支标记 FAILED 且不影响其他分支"""
        def boom(channel):
            if channel == "dense":
                raise RuntimeError("dense down")
            return [make_result("c1", 0.6)]

        launcher = SpeculativeLauncher()
        branches = launcher.launch(["exact", "dense"], boom)
        exact = next(b for b in branches if b.channel == "exact")
        dense = next(b for b in branches if b.channel == "dense")
        assert exact.status == BranchStatus.COMPLETED
        assert dense.status == BranchStatus.FAILED
        assert dense.error is not None
        assert dense.completed_at is not None

    def test_launch_retrieval_func_returns_none_treated_empty(self):
        """检索函数返回 None → 视为空结果，分支仍 COMPLETED"""
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["exact"], lambda c: None)
        assert branches[0].status == BranchStatus.COMPLETED
        assert branches[0].results == []
        assert branches[0].score == 0.0

    def test_branch_id_format(self):
        """branch_id 格式为 {channel}-{uuid}"""
        launcher = SpeculativeLauncher()
        branches = launcher.launch(["exact", "dense"])
        for b in branches:
            assert b.branch_id.startswith(f"{b.channel}-")
            # uuid 部分长度为 8
            suffix = b.branch_id.split("-", 1)[1]
            assert len(suffix) == 8

    def test_launch_tiered_groups_by_tier(self):
        """分层启动: T0 立即执行，T1/T2 保持 LAUNCHED"""
        launcher = SpeculativeLauncher()
        grouped = launcher.launch_tiered(
            ["exact", "dense", "table"],
            channel_retrieval_func({"exact": [make_result("c1", 0.8)]}),
        )
        assert set(grouped.keys()) >= {"T0", "T1", "T2"}
        # T0 执行完成
        t0 = grouped["T0"]
        assert len(t0) == 1
        assert t0[0].channel == "exact"
        assert t0[0].status == BranchStatus.COMPLETED
        assert t0[0].score == pytest.approx(0.8)
        # T1 未执行
        t1 = grouped["T1"]
        assert len(t1) == 1
        assert t1[0].channel == "dense"
        assert t1[0].status == BranchStatus.LAUNCHED
        assert t1[0].results == []
        # T2 未执行
        t2 = grouped["T2"]
        assert len(t2) == 1
        assert t2[0].channel == "table"
        assert t2[0].status == BranchStatus.LAUNCHED

    def test_launch_tiered_without_retrieval_func(self):
        """分层启动无检索函数 → T0 也保持 LAUNCHED"""
        launcher = SpeculativeLauncher()
        grouped = launcher.launch_tiered(["exact", "dense"])
        assert grouped["T0"][0].status == BranchStatus.LAUNCHED
        assert grouped["T1"][0].status == BranchStatus.LAUNCHED

    def test_branch_to_dict(self):
        """RetrievalBranch.to_dict 返回完整字段且状态为字符串值"""
        b = make_branch(
            channel="dense",
            tier=1,
            results=[make_result("c1", 0.8)],
        )
        d = b.to_dict()
        assert d["channel"] == "dense"
        assert d["tier"] == 1
        assert d["status"] == "completed"
        assert d["score"] == pytest.approx(0.8)
        assert len(d["results"]) == 1


# ============================================================
# ResultStream 测试
# ============================================================


class TestResultStream:
    """检索结果流收集器测试"""

    def test_get_all_results_sorted_by_score_desc(self):
        """结果收集: 多分支结果 → 按 score 降序"""
        stream = ResultStream()
        stream.add_results(
            "exact-aaa", [make_result("c1", 0.3), make_result("c2", 0.9)]
        )
        stream.add_results("dense-bbb", [make_result("c3", 0.6)])
        results = stream.get_all_results()
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(0.9)
        assert len(results) == 3

    def test_merge_and_dedupe_keeps_highest_score(self):
        """结果去重: 相同 chunk_id → 去重保留最高分"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [make_result("c1", 0.3)])
        stream.add_results(
            "dense-bbb", [make_result("c1", 0.9), make_result("c2", 0.5)]
        )
        merged = stream.merge_and_dedupe()
        # c1 出现两次，保留最高分 0.9
        c1 = next(r for r in merged if r["chunk_id"] == "c1")
        assert c1["score"] == pytest.approx(0.9)
        assert len(merged) == 2  # c1 + c2
        # 按分数降序
        assert merged[0]["score"] >= merged[1]["score"]

    def test_empty_branch_handled(self):
        """空分支: 无结果 → 正常处理"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [])
        assert stream.get_all_results() == []
        assert stream.get_result_count() == 0
        assert stream.merge_and_dedupe() == []

    def test_get_result_count_total(self):
        """get_result_count 返回未去重的总数"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [make_result("c1", 0.5), make_result("c2", 0.4)])
        stream.add_results("dense-bbb", [make_result("c1", 0.9)])  # 重复 chunk_id
        assert stream.get_result_count() == 3

    def test_get_results_by_channel(self):
        """按通道过滤结果"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [make_result("c1", 0.5)])
        stream.add_results("dense-bbb", [make_result("c2", 0.8)])
        exact_results = stream.get_results_by_channel("exact")
        assert len(exact_results) == 1
        assert exact_results[0]["chunk_id"] == "c1"
        dense_results = stream.get_results_by_channel("dense")
        assert len(dense_results) == 1
        assert dense_results[0]["chunk_id"] == "c2"

    def test_clear(self):
        """clear 清空所有结果"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [make_result("c1", 0.5)])
        stream.clear()
        assert stream.get_result_count() == 0
        assert stream.get_all_results() == []

    def test_merge_and_dedupe_no_chunk_id_kept(self):
        """无 chunk_id 的结果视为唯一，全部保留"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [{"score": 0.5}, {"score": 0.4}])
        merged = stream.merge_and_dedupe()
        assert len(merged) == 2

    def test_get_all_results_returns_copy(self):
        """get_all_results 返回副本，修改不影响内部状态"""
        stream = ResultStream()
        stream.add_results("exact-aaa", [make_result("c1", 0.5)])
        results = stream.get_all_results()
        results.clear()
        assert stream.get_result_count() == 1

    def test_channel_parsed_from_branch_id(self):
        """通道从 branch_id 前缀解析"""
        stream = ResultStream()
        stream.add_results("metadata-xyz", [make_result("c1", 0.5)])
        assert stream.get_results_by_channel("metadata") != []


# ============================================================
# EarlyStopEvaluator 测试
# ============================================================


class TestEarlyStopEvaluator:
    """早停评估器测试"""

    def test_sufficiency_triggers_stop(self):
        """早停-充分性: score=0.9 → should_stop=True"""
        evaluator = EarlyStopEvaluator()
        result = evaluator.evaluate([], sufficiency_score=0.9)
        assert result.should_stop is True
        assert result.reason == "sufficient"
        assert result.current_sufficiency == pytest.approx(0.9)

    def test_marginal_gain_low_triggers_stop(self):
        """早停-边际增益: gain<0.05 → should_stop=True"""
        evaluator = EarlyStopEvaluator()
        # current=0.5, previous=0.48 → gain=0.02 < 0.05
        result = evaluator.evaluate(
            [], sufficiency_score=0.5, previous_score=0.48
        )
        assert result.should_stop is True
        assert result.reason == "marginal_gain_low"
        assert result.marginal_gain == pytest.approx(0.02)

    def test_budget_exhausted_triggers_stop(self):
        """早停-预算: budget_exhausted → should_stop=True"""
        evaluator = EarlyStopEvaluator()
        result = evaluator.evaluate(
            [], sufficiency_score=0.3, budget_exhausted=True
        )
        assert result.should_stop is True
        assert result.reason == "budget_exhausted"

    def test_continue_when_below_threshold(self):
        """未达阈值且预算未耗尽且边际增益足够 → 继续"""
        evaluator = EarlyStopEvaluator()
        result = evaluator.evaluate(
            [], sufficiency_score=0.3, previous_score=0.1
        )
        assert result.should_stop is False
        assert result.reason == "none"
        assert result.marginal_gain == pytest.approx(0.2)

    def test_sufficiency_priority_over_budget(self):
        """充分性优先级高于预算耗尽"""
        evaluator = EarlyStopEvaluator()
        result = evaluator.evaluate(
            [], sufficiency_score=0.9, budget_exhausted=True
        )
        assert result.should_stop is True
        assert result.reason == "sufficient"

    def test_budget_priority_over_marginal_gain(self):
        """预算耗尽优先级高于边际增益"""
        evaluator = EarlyStopEvaluator()
        result = evaluator.evaluate(
            [],
            sufficiency_score=0.3,
            previous_score=0.29,  # gain=0.01 < 0.05
            budget_exhausted=True,
        )
        assert result.should_stop is True
        assert result.reason == "budget_exhausted"

    def test_first_round_no_marginal_gain_check(self):
        """首轮（previous_score=0）不判断边际增益，避免误判"""
        evaluator = EarlyStopEvaluator()
        # current=0.5, previous=0 → gain=0.5, 但 previous 不 > 0
        result = evaluator.evaluate(
            [], sufficiency_score=0.5, previous_score=0.0
        )
        assert result.should_stop is False
        assert result.reason == "none"

    def test_marginal_gain_negative_clamped_to_zero(self):
        """边际增益为负时钳制为 0"""
        evaluator = EarlyStopEvaluator()
        gain = evaluator.estimate_marginal_gain(0.3, 0.5)
        assert gain == 0.0

    def test_marginal_gain_at_threshold_does_not_stop(self):
        """边际增益恰好等于阈值不触发停止（< 才停止）"""
        evaluator = EarlyStopEvaluator()
        # current=0.55, previous=0.5 → gain=0.05 == threshold, 不 < threshold
        result = evaluator.evaluate(
            [], sufficiency_score=0.55, previous_score=0.5
        )
        assert result.should_stop is False
        assert result.marginal_gain == pytest.approx(0.05)

    def test_early_stop_result_to_dict(self):
        """EarlyStopResult.to_dict 返回完整字段"""
        r = EarlyStopResult(
            should_stop=True,
            reason="sufficient",
            current_sufficiency=0.9,
            marginal_gain=0.1,
        )
        d = r.to_dict()
        assert d == {
            "should_stop": True,
            "reason": "sufficient",
            "current_sufficiency": 0.9,
            "marginal_gain": 0.1,
        }


# ============================================================
# BranchCanceller 测试
# ============================================================


class TestBranchCanceller:
    """动态分支取消器测试"""

    def test_cancel_dense_when_exact_unique_clause(self):
        """取消分支: 精确找到唯一条款 → Dense 被取消"""
        canceller = BranchCanceller()
        exact = make_branch(
            channel="exact",
            tier=0,
            status=BranchStatus.COMPLETED,
            results=[make_result("c1", 0.9)],
        )
        dense = make_branch(channel="dense", tier=1, status=BranchStatus.LAUNCHED)
        relation = make_branch(
            channel="relation", tier=2, status=BranchStatus.LAUNCHED
        )
        canceller.evaluate_and_cancel(
            [exact, dense, relation], sufficiency_score=0.3
        )
        # exact 已完成不被取消
        assert exact.status == BranchStatus.COMPLETED
        # dense 和 relation 被取消
        assert dense.status == BranchStatus.CANCELLED
        assert relation.status == BranchStatus.CANCELLED

    def test_cancel_all_pending_when_sufficient(self):
        """证据充分 → 取消所有未完成分支"""
        canceller = BranchCanceller()
        exact = make_branch(
            channel="exact",
            status=BranchStatus.COMPLETED,
            results=[make_result("c1", 0.9)],
        )
        dense = make_branch(channel="dense", status=BranchStatus.LAUNCHED)
        table = make_branch(channel="table", status=BranchStatus.RUNNING)
        canceller.evaluate_and_cancel(
            [exact, dense, table], sufficiency_score=0.9
        )
        assert exact.status == BranchStatus.COMPLETED
        assert dense.status == BranchStatus.CANCELLED
        assert table.status == BranchStatus.CANCELLED

    def test_cancel_table_when_same_parent_hits(self):
        """lexical 与 dense 命中相同父条款 → 取消 table"""
        canceller = BranchCanceller()
        lexical = make_branch(
            channel="lexical",
            status=BranchStatus.COMPLETED,
            results=[make_result("clauseA-1", 0.8, parent_id="clauseA")],
        )
        dense = make_branch(
            channel="dense",
            status=BranchStatus.COMPLETED,
            results=[make_result("clauseA-2", 0.7, parent_id="clauseA")],
        )
        table = make_branch(channel="table", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel(
            [lexical, dense, table], sufficiency_score=0.3
        )
        assert table.status == BranchStatus.CANCELLED

    def test_no_cancel_when_no_rules_match(self):
        """无规则匹配 → 不取消任何分支"""
        canceller = BranchCanceller()
        exact = make_branch(channel="exact", status=BranchStatus.LAUNCHED)
        dense = make_branch(channel="dense", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel([exact, dense], sufficiency_score=0.3)
        assert all(b.status != BranchStatus.CANCELLED for b in [exact, dense])

    def test_completed_branches_not_cancelled(self):
        """已完成分支即使命中取消规则也不被取消"""
        canceller = BranchCanceller()
        exact = make_branch(
            channel="exact",
            status=BranchStatus.COMPLETED,
            results=[make_result("c1", 0.9)],
        )
        dense = make_branch(
            channel="dense",
            status=BranchStatus.COMPLETED,
            results=[make_result("c2", 0.5)],
        )
        canceller.evaluate_and_cancel([exact, dense], sufficiency_score=0.3)
        # exact 命中唯一条款规则 → 尝试取消 dense，但 dense 已完成
        assert dense.status == BranchStatus.COMPLETED

    def test_cancel_all_pending_directly(self):
        """cancel_all_pending 取消 LAUNCHED 和 RUNNING，保留其他"""
        canceller = BranchCanceller()
        launched = make_branch(channel="dense", status=BranchStatus.LAUNCHED)
        running = make_branch(channel="table", status=BranchStatus.RUNNING)
        completed = make_branch(
            channel="exact", status=BranchStatus.COMPLETED
        )
        canceller.cancel_all_pending([launched, running, completed])
        assert launched.status == BranchStatus.CANCELLED
        assert running.status == BranchStatus.CANCELLED
        assert completed.status == BranchStatus.COMPLETED

    def test_unique_clause_requires_single_result(self):
        """精确检索多条结果不视为唯一条款"""
        canceller = BranchCanceller()
        exact = make_branch(
            channel="exact",
            status=BranchStatus.COMPLETED,
            results=[make_result("c1", 0.9), make_result("c2", 0.8)],
        )
        dense = make_branch(channel="dense", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel([exact, dense], sufficiency_score=0.3)
        assert dense.status == BranchStatus.LAUNCHED

    def test_unique_clause_requires_high_score(self):
        """精确检索唯一结果但得分低于阈值不视为唯一条款"""
        canceller = BranchCanceller()
        exact = make_branch(
            channel="exact",
            status=BranchStatus.COMPLETED,
            results=[make_result("c1", 0.5)],  # < 0.8
        )
        dense = make_branch(channel="dense", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel([exact, dense], sufficiency_score=0.3)
        assert dense.status == BranchStatus.LAUNCHED

    def test_unique_clause_requires_completed_status(self):
        """精确检索未完成不视为唯一条款"""
        canceller = BranchCanceller()
        exact = make_branch(
            channel="exact",
            status=BranchStatus.RUNNING,
            results=[make_result("c1", 0.9)],
        )
        dense = make_branch(channel="dense", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel([exact, dense], sufficiency_score=0.3)
        assert dense.status == BranchStatus.LAUNCHED

    def test_same_parent_from_chunk_id_prefix(self):
        """无 parent_id 时从 chunk_id 前缀推导父条款"""
        canceller = BranchCanceller()
        lexical = make_branch(
            channel="lexical",
            status=BranchStatus.COMPLETED,
            results=[make_result("clauseA-1", 0.8)],  # 无 parent_id
        )
        dense = make_branch(
            channel="dense",
            status=BranchStatus.COMPLETED,
            results=[make_result("clauseA-2", 0.7)],  # 无 parent_id
        )
        table = make_branch(channel="table", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel(
            [lexical, dense, table], sufficiency_score=0.3
        )
        assert table.status == BranchStatus.CANCELLED

    def test_different_parent_does_not_cancel(self):
        """lexical 与 dense 父条款不同 → 不取消 table"""
        canceller = BranchCanceller()
        lexical = make_branch(
            channel="lexical",
            status=BranchStatus.COMPLETED,
            results=[make_result("c1", 0.8, parent_id="clauseA")],
        )
        dense = make_branch(
            channel="dense",
            status=BranchStatus.COMPLETED,
            results=[make_result("c2", 0.7, parent_id="clauseB")],
        )
        table = make_branch(channel="table", status=BranchStatus.LAUNCHED)
        canceller.evaluate_and_cancel(
            [lexical, dense, table], sufficiency_score=0.3
        )
        assert table.status == BranchStatus.LAUNCHED

    def test_returns_same_branch_list(self):
        """evaluate_and_cancel 返回与入参相同的列表对象"""
        canceller = BranchCanceller()
        branches = [make_branch(channel="exact", status=BranchStatus.LAUNCHED)]
        result = canceller.evaluate_and_cancel(branches, sufficiency_score=0.3)
        assert result is branches
