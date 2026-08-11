"""
Plan Optimizer 单元测试 (M4.1)

测试范围:
  - CostEstimator: 成本估计、空计划、成本表完整性、自定义成本表
  - PlanOptimizer: 多候选生成、计划选择、优化流程

测试用例表（开发计划）:
  | 测试用例       | 输入               | 预期                            |
  | 成本估计       | 3算子计划          | 返回 latency 和 gain            |
  | 计划选择       | 2个候选计划        | 选成本更低的                    |
  | 空计划         | 无阶段             | latency=0                       |
  | 多计划生成     | clause_query IR   | 生成>=2个候选                   |
  | 优化流程       | threshold IR      | 返回最优计划                    |
  | 成本表完整性   | 所有通道           | 都有 latency/gain/success_rate  |

模式参考: agent_platform/tests/unit/test_query_compiler.py
  - 使用 pytest
  - 定义测试数据工厂函数
"""

import pytest

from agent_platform.query_compiler import (
    IRBuilder,
    LogicalPlanner,
    PhysicalPlanner,
)
from agent_platform.query_compiler.physical_planner.cost_estimator import (
    DEFAULT_OPERATOR_COST,
    OPERATOR_COSTS,
    CostEstimator,
    PlanCost,
)
from agent_platform.query_compiler.physical_planner.plan_optimizer import (
    PlanOptimizer,
)
from agent_platform.query_compiler.physical_planner.planner import (
    PhysicalPlan,
    PlanStage,
)


# ============================================================
# 测试数据工厂
# ============================================================

def make_query_spec(
    intent: str = "clause_query",
    risk_level: str = "medium",
    entities: list = None,
    constraints: dict = None,
    top_k: int = 10,
    **kwargs,
) -> dict:
    """构造 query_spec 字典

    query_spec 需要有 "intent" 和 "risk_level" 字段。
    其余字段（entities / constraints / top_k 等）为可选项。
    """
    spec = {
        "intent": intent,
        "risk_level": risk_level,
        "entities": entities or [],
        "constraints": constraints or {},
        "top_k": top_k,
    }
    spec.update(kwargs)
    return spec


def make_claims(count: int = 2, slot_type: str = "metric|required") -> list:
    """构造声明槽位列表（dict 格式）

    slot_type 编码 "{template_key}|required" 或 "{template_key}|optional"，
    默认必填（required），用于 PlanValidator 的覆盖校验。
    """
    return [
        {
            "claim_id": f"c{i}",
            "description": f"声明槽位 {i}",
            "slot_type": slot_type,
            "status": "pending",
            "evidence_ids": [],
        }
        for i in range(count)
    ]


def build_ir(intent: str = "clause_query", risk_level: str = "medium"):
    """构建 QueryIR"""
    spec = make_query_spec(intent=intent, risk_level=risk_level)
    return IRBuilder().build(spec, make_claims(2))


def build_physical_plan(
    intent: str = "clause_query", risk_level: str = "medium"
) -> PhysicalPlan:
    """构建物理计划：IR → 逻辑计划 → 物理计划"""
    ir = build_ir(intent=intent, risk_level=risk_level)
    logical_plan = LogicalPlanner().plan(ir)
    return PhysicalPlanner().plan(logical_plan, ir)


def make_plan_with_channels(
    channels: list,
    rerank: bool = False,
    top_k: int = 10,
    intent: str = "clause_query",
    plan_id: str = "pp-test",
) -> PhysicalPlan:
    """构造指定通道的物理计划（单阶段）

    用于精确控制成本估计的输入，便于断言具体延迟/增益数值。
    """
    operations = [f"retrieve:{ch}" for ch in channels]
    if rerank:
        operations.append("rerank")
    if len(channels) > 1:
        operations.append("fuse")
    stage = PlanStage(
        name="retrieve",
        channels=list(channels),
        top_k=top_k,
        rerank=rerank,
        timeout_ms=5000,
        operations=operations,
        condition="all_operators_returned",
        claim_ids=["c0", "c1"],
    )
    return PhysicalPlan(
        plan_id=plan_id,
        intent=intent,
        stages=[stage],
        stop_conditions=[],
        budget_ms=5000,
    )


def make_empty_plan(intent: str = "clause_query") -> PhysicalPlan:
    """构造无阶段的空物理计划"""
    return PhysicalPlan(
        plan_id="pp-empty",
        intent=intent,
        stages=[],
        stop_conditions=[],
        budget_ms=5000,
    )


# ============================================================
# CostEstimator 测试
# ============================================================

class TestCostEstimator:
    """算子成本估计器测试"""

    def test_estimate_returns_latency_and_gain(self):
        """成本估计：3算子计划返回 latency 和 gain"""
        # threshold 意图: channels=[lexical, dense, metadata]（3 算子）+ rerank
        plan = build_physical_plan(intent="threshold")
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        assert isinstance(cost, PlanCost)
        assert cost.latency_ms > 0
        assert 0.0 < cost.estimated_gain <= 1.0
        assert 0.0 < cost.success_rate <= 1.0

    def test_estimate_three_operators_latency_no_rerank(self):
        """3算子无 rerank 计划延迟 = 通道并行取 max"""
        # lexical=100, dense=200, metadata=30 → max=200
        plan = make_plan_with_channels(
            channels=["lexical", "dense", "metadata"], rerank=False
        )
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        assert cost.latency_ms == 200

    def test_estimate_with_rerank_latency(self):
        """含 rerank 计划延迟 = max(通道) + rerank_latency"""
        # exact=50, metadata=30 → max=50; rerank=500 → 50+500=550
        plan = make_plan_with_channels(
            channels=["exact", "metadata"], rerank=True
        )
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        assert cost.latency_ms == 550

    def test_estimate_single_channel(self):
        """单通道计划延迟 = 通道延迟"""
        # exact=50
        plan = make_plan_with_channels(channels=["exact"], rerank=False)
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        assert cost.latency_ms == 50

    def test_estimate_empty_plan_latency_zero(self):
        """空计划：无阶段 → latency=0"""
        plan = make_empty_plan()
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        assert cost.latency_ms == 0
        # 空计划无通道 → 增益为 0
        assert cost.estimated_gain == 0.0
        # 空计划无阶段约束 → 成功率 = 1.0（_multiply([]) = 1.0）
        assert cost.success_rate == 1.0

    def test_estimate_empty_plan_total_cost_positive(self):
        """空计划总成本 > 0（增益不足惩罚生效）"""
        plan = make_empty_plan()
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        # total_cost = 0 + 200*(1-0) + 500*(1-1) = 200
        assert cost.total_cost > 0

    def test_estimate_breakdown_contains_all_operators(self):
        """成本明细包含所有算子"""
        plan = build_physical_plan(intent="threshold")
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        # threshold: channels=[lexical, dense, metadata] + rerank
        assert "retrieve:lexical" in cost.breakdown
        assert "retrieve:dense" in cost.breakdown
        assert "retrieve:metadata" in cost.breakdown
        assert "retrieve:rerank" in cost.breakdown

    def test_estimate_total_cost_positive(self):
        """非空计划总成本为正数"""
        plan = build_physical_plan(intent="clause_query")
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        assert cost.total_cost > 0

    def test_cost_table_completeness(self):
        """成本表完整性：所有通道都有 latency/gain/success_rate"""
        required_keys = {"latency_ms", "success_rate", "evidence_gain"}
        for channel, cost_entry in OPERATOR_COSTS.items():
            assert required_keys.issubset(
                cost_entry.keys()
            ), f"通道 '{channel}' 缺少必要成本字段: {required_keys - set(cost_entry.keys())}"
            assert cost_entry["latency_ms"] > 0, f"通道 '{channel}' latency_ms 应为正数"
            assert (
                0.0 <= cost_entry["success_rate"] <= 1.0
            ), f"通道 '{channel}' success_rate 应在 [0, 1] 范围内"
            assert (
                0.0 <= cost_entry["evidence_gain"] <= 1.0
            ), f"通道 '{channel}' evidence_gain 应在 [0, 1] 范围内"

    def test_cost_table_covers_all_retrieval_channels(self):
        """成本表覆盖所有检索通道与 rerank"""
        expected_channels = {"exact", "lexical", "dense", "metadata", "table", "rerank"}
        assert expected_channels.issubset(set(OPERATOR_COSTS.keys()))

    def test_unknown_operator_uses_default(self):
        """未知算子使用兜底成本"""
        plan = make_plan_with_channels(
            channels=["unknown_channel"], rerank=False
        )
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        # 兜底 latency_ms=200
        assert cost.latency_ms == DEFAULT_OPERATOR_COST["latency_ms"]
        # breakdown 中仍记录该算子
        assert "retrieve:unknown_channel" in cost.breakdown

    def test_custom_operator_costs_override(self):
        """自定义成本表覆盖默认值"""
        custom = {
            "exact": {
                "latency_ms": 1,
                "success_rate": 1.0,
                "evidence_gain": 1.0,
            }
        }
        plan = make_plan_with_channels(channels=["exact"], rerank=False)
        estimator = CostEstimator(operator_costs=custom)
        cost = estimator.estimate(plan)

        assert cost.latency_ms == 1

    def test_rerank_increases_latency(self):
        """rerank 增加延迟"""
        plan_no_rerank = make_plan_with_channels(
            channels=["exact"], rerank=False
        )
        plan_with_rerank = make_plan_with_channels(
            channels=["exact"], rerank=True
        )
        estimator = CostEstimator()

        cost_no = estimator.estimate(plan_no_rerank)
        cost_yes = estimator.estimate(plan_with_rerank)

        assert cost_yes.latency_ms > cost_no.latency_ms

    def test_multi_stage_latency_accumulates(self):
        """多阶段计划延迟 = 阶段延迟累加"""
        stage1 = PlanStage(
            name="retrieve",
            channels=["exact"],
            top_k=10,
            rerank=False,
            timeout_ms=5000,
            operations=["retrieve:exact"],
            condition="all_operators_returned",
            claim_ids=["c0"],
        )
        stage2 = PlanStage(
            name="verify_c1",
            channels=["lexical"],
            top_k=10,
            rerank=False,
            timeout_ms=5000,
            operations=["retrieve:lexical"],
            condition="claim_c1_resolved",
            claim_ids=["c1"],
        )
        plan = PhysicalPlan(
            plan_id="pp-multi",
            intent="clause_query",
            stages=[stage1, stage2],
            stop_conditions=[],
            budget_ms=10000,
        )
        estimator = CostEstimator()
        cost = estimator.estimate(plan)

        # exact=50 + lexical=100 = 150
        assert cost.latency_ms == 150

    def test_to_dict_serializable(self):
        """PlanCost 可序列化为字典"""
        plan = build_physical_plan(intent="clause_query")
        estimator = CostEstimator()
        cost = estimator.estimate(plan)
        data = cost.to_dict()

        assert "latency_ms" in data
        assert "estimated_gain" in data
        assert "success_rate" in data
        assert "total_cost" in data
        assert "breakdown" in data


# ============================================================
# PlanOptimizer 测试
# ============================================================

class TestPlanOptimizer:
    """计划优化器测试"""

    def test_generate_candidates_clause_query(self):
        """多计划生成：clause_query IR 生成 >=2 个候选"""
        ir = build_ir(intent="clause_query")
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)

        assert len(candidates) >= 2
        for plan in candidates:
            assert isinstance(plan, PhysicalPlan)

    def test_generate_candidates_threshold(self):
        """多计划生成：threshold IR 生成 >=2 个候选"""
        ir = build_ir(intent="threshold")
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)

        assert len(candidates) >= 2

    def test_generate_candidates_distinct_channels(self):
        """候选计划的通道集合不完全相同"""
        ir = build_ir(intent="clause_query")
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)

        channel_sets = [
            tuple(s.channels)
            for plan in candidates
            for s in plan.stages
            if s.channels
        ]
        # 至少有两个不同的通道组合
        assert len(set(channel_sets)) >= 2

    def test_generate_candidates_all_valid_plans(self):
        """所有候选计划都有阶段和通道"""
        ir = build_ir(intent="threshold")
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)

        for plan in candidates:
            assert len(plan.stages) > 0
            retrieve_stage = plan.stages[0]
            assert len(retrieve_stage.channels) > 0

    def test_select_best_returns_lower_cost_plan(self):
        """计划选择：2个候选计划，选成本更低的"""
        # 低成本计划：单通道无 rerank
        cheap_plan = make_plan_with_channels(
            channels=["exact"], rerank=False, plan_id="pp-cheap"
        )
        # 高成本计划：多通道 + rerank
        expensive_plan = make_plan_with_channels(
            channels=["exact", "dense", "metadata"],
            rerank=True,
            plan_id="pp-expensive",
        )

        optimizer = PlanOptimizer()
        best = optimizer.select_best([expensive_plan, cheap_plan])

        assert best is not None
        assert best.plan_id == "pp-cheap"

    def test_select_best_empty_candidates(self):
        """空候选列表返回 None"""
        optimizer = PlanOptimizer()
        assert optimizer.select_best([]) is None

    def test_select_best_single_candidate(self):
        """单候选直接返回该候选"""
        plan = make_plan_with_channels(
            channels=["exact"], rerank=False, plan_id="pp-single"
        )
        optimizer = PlanOptimizer()
        best = optimizer.select_best([plan])
        assert best is plan

    def test_select_best_stable_on_tie(self):
        """成本相同时保留首个最低者（稳定选择）"""
        plan_a = make_plan_with_channels(
            channels=["exact"], rerank=False, plan_id="pp-a"
        )
        plan_b = make_plan_with_channels(
            channels=["exact"], rerank=False, plan_id="pp-b"
        )
        optimizer = PlanOptimizer()
        best = optimizer.select_best([plan_a, plan_b])

        # 两者成本相同，应返回首个（plan_a）
        assert best is plan_a

    def test_optimize_threshold_returns_plan(self):
        """优化流程：threshold IR 返回最优计划"""
        ir = build_ir(intent="threshold")
        optimizer = PlanOptimizer()
        best = optimizer.optimize(ir)

        assert best is not None
        assert isinstance(best, PhysicalPlan)
        assert best.intent == "threshold"

    def test_optimize_clause_query_returns_plan(self):
        """优化流程：clause_query IR 返回最优计划"""
        ir = build_ir(intent="clause_query")
        optimizer = PlanOptimizer()
        best = optimizer.optimize(ir)

        assert best is not None
        assert isinstance(best, PhysicalPlan)
        assert best.intent == "clause_query"

    def test_optimize_best_has_lowest_cost(self):
        """optimize 返回的计划成本不高于任何候选"""
        ir = build_ir(intent="threshold")
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)
        best = optimizer.optimize(ir)

        estimator = CostEstimator()
        best_cost = estimator.estimate(best).total_cost
        for plan in candidates:
            cost = estimator.estimate(plan).total_cost
            assert best_cost <= cost

    def test_optimize_returns_none_for_empty(self):
        """optimize 在无候选时返回 None（select_best 容错）"""
        optimizer = PlanOptimizer()
        # select_best([]) → None，optimize 依赖此行为
        assert optimizer.select_best([]) is None

    def test_custom_cost_estimator_injection(self):
        """支持注入自定义 CostEstimator"""
        custom_estimator = CostEstimator(
            operator_costs={
                "exact": {
                    "latency_ms": 1000,
                    "success_rate": 0.5,
                    "evidence_gain": 0.1,
                }
            }
        )
        optimizer = PlanOptimizer(cost_estimator=custom_estimator)
        ir = build_ir(intent="clause_query")
        best = optimizer.optimize(ir)

        assert best is not None
        assert isinstance(best, PhysicalPlan)
