"""
Phase 4 端到端集成测试

验证 Phase 4 模块（DAG 执行器、推测式检索、检查点管理、计划缓存/优化器）
与 Phase 3 模块（声明规划、查询编译、证据组装）的协同工作。

测试场景:
  1. 比较查询 — DAG 拆解为 2 个独立分支 + 合并
  2. 合规判断 — DAG 多步拆解，依赖正确
  3. 跨文件引用 — DAG 多步检索 + 结果回查
  4. 推测式取消 — 精确检索直接命中，Dense 分支被取消
  5. 检查点恢复 — 模拟中途失败后恢复，从检查点继续，不重复
  6. 计划缓存命中 — 相似问题第二次执行，命中缓存
"""

import asyncio
import pytest

from agent_platform.evidence.claim_planner import ClaimPlanner, SlotFiller
from agent_platform.evidence.evidence_assembler.builder import (
    EvidenceBuilder,
    EvidenceBundle,
    EvidenceItem,
)
from agent_platform.evidence.sufficiency_scorer import SufficiencyScorer
from agent_platform.orchestration.checkpoint_manager import (
    Checkpoint,
    CheckpointManager,
    RecoveryManager,
)
from agent_platform.orchestration.dag_executor import (
    DagExecutor,
    DagState,
    DagTask,
    TaskStatus,
)
from agent_platform.query_compiler import (
    CacheContext,
    CacheKeyGenerator,
    CostEstimator,
    IRBuilder,
    LogicalPlanner,
    PlanCache,
    PlanOptimizer,
    PlanValidator,
)
from agent_platform.speculative_retrieval import (
    BranchCanceller,
    BranchStatus,
    EarlyStopEvaluator,
    ResultStream,
    SpeculativeLauncher,
)


# ============================================================
# 测试数据工厂
# ============================================================

def make_hit(
    chunk_id: str = "",
    content: str = "",
    score: float = 0.8,
    doc_name: str = "《商业银行资本管理办法》",
    chunk_type: str = "clause",
    citation: str = "",
    hierarchy_path: str = "",
    metadata: dict = None,
    parent_id: str = "",
) -> dict:
    """构造 RetrievalHit dict"""
    import uuid

    return {
        "chunk_id": chunk_id or f"chunk-{uuid.uuid4().hex[:8]}",
        "content": content,
        "evidence_snippet": content[:200],
        "score": score,
        "doc_name": doc_name,
        "doc_id": doc_name,
        "chunk_type": chunk_type,
        "citation": citation or f"{doc_name} 第1条",
        "hierarchy_path": hierarchy_path,
        "parent_id": parent_id,
        "metadata": metadata or {},
    }


def make_evidence_item(
    evidence_id: str = "",
    chunk_id: str = "",
    content: str = "",
    score: float = 0.8,
    source_doc: str = "《商业银行资本管理办法》",
    hierarchy_path: str = "",
    chunk_type: str = "clause",
    normative_level: str = "部门规章",
    version_status: str = "active",
    citation: str = "",
    metadata: dict = None,
) -> EvidenceItem:
    """构造 EvidenceItem"""
    import uuid

    return EvidenceItem(
        evidence_id=evidence_id or f"ev-{uuid.uuid4().hex[:8]}",
        chunk_id=chunk_id or f"chunk-{uuid.uuid4().hex[:8]}",
        content=content,
        evidence_snippet=content[:200],
        citation=citation or f"{source_doc} 第1条",
        score=score,
        source_doc=source_doc,
        hierarchy_path=hierarchy_path,
        chunk_type=chunk_type,
        normative_level=normative_level,
        version_status=version_status,
        metadata=metadata or {},
    )


# ============================================================
# 场景 1: 比较查询 — DAG 拆解为 2 个独立分支 + 合并
# ============================================================

class TestPhase4E2EComparison:
    """比较查询端到端 — DAG 拆解为独立分支 + 合并结果"""

    def test_comparison_dag_two_branches_merge(self):
        """比较查询 — DAG 拆解两个分支（系统重要性 vs 非系统重要性）+ 合并"""
        # 1. 声明规划 + 查询编译
        planner = ClaimPlanner()
        query_spec = {"intent": "comparison", "risk_level": "high"}
        slots = planner.plan(query_spec)
        assert len(slots) == 5

        ir = IRBuilder().build(query_spec, slots)
        assert ir.intent == "comparison"

        # 2. DAG 拆解：两个独立检索分支 + 合并任务
        tasks = [
            DagTask(
                task_id="t1_branch_a",
                description="检索系统重要性银行资本要求",
                dependencies=[],
                retrieval_strategy={"channels": ["lexical", "dense"], "top_k": 20},
                input_constraints={"filter": "systemically_important=true"},
            ),
            DagTask(
                task_id="t2_branch_b",
                description="检索非系统重要性银行资本要求",
                dependencies=[],
                retrieval_strategy={"channels": ["lexical", "dense"], "top_k": 20},
                input_constraints={"filter": "systemically_important=false"},
            ),
            DagTask(
                task_id="t3_merge",
                description="合并两个分支的检索结果",
                dependencies=["t1_branch_a", "t2_branch_b"],
                retrieval_strategy={"strategy": "fuse"},
                completion_condition="both_branches_returned",
            ),
            DagTask(
                task_id="t4_assemble",
                description="组装证据并评分",
                dependencies=["t3_merge"],
                retrieval_strategy={"strategy": "assemble"},
            ),
        ]

        # 3. DAG 执行
        executor = DagExecutor()
        state = executor.execute_sync(tasks)

        # 4. 验证执行结果
        assert state.is_complete, "DAG 应全部完成"
        assert not state.has_failure, "不应有失败任务"
        assert len(state.execution_order) == 4

        # t1 和 t2 应在 t3 和 t4 之前执行
        t1_idx = state.execution_order.index("t1_branch_a")
        t2_idx = state.execution_order.index("t2_branch_b")
        t3_idx = state.execution_order.index("t3_merge")
        t4_idx = state.execution_order.index("t4_assemble")

        assert t1_idx < t3_idx, "t1 应在 t3 之前"
        assert t2_idx < t3_idx, "t2 应在 t3 之前"
        assert t3_idx < t4_idx, "t3 应在 t4 之前"

    def test_comparison_query_with_plan_optimizer(self):
        """比较查询 — 计划优化器选择最优物理计划"""
        planner = ClaimPlanner()
        query_spec = {"intent": "comparison", "risk_level": "high"}
        slots = planner.plan(query_spec)

        ir = IRBuilder().build(query_spec, slots)

        # 生成候选计划并选择最优
        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)
        assert len(candidates) >= 2, "应生成至少 2 个候选计划"

        best_plan = optimizer.select_best(candidates)
        assert best_plan is not None, "应选择出最优计划"
        assert best_plan.intent == "comparison"

        # 验证最优计划的阶段包含多个通道
        primary_stage = best_plan.stages[0]
        assert len(primary_stage.channels) >= 2, "比较查询应有多通道"

    def test_comparison_parallel_branches_are_independent(self):
        """比较查询 — 两个分支无依赖关系，可并行执行"""
        tasks = [
            DagTask(
                task_id="br_a",
                description="分支A",
                dependencies=[],
            ),
            DagTask(
                task_id="br_b",
                description="分支B",
                dependencies=[],
            ),
            DagTask(
                task_id="merge",
                description="合并",
                dependencies=["br_a", "br_b"],
            ),
        ]

        executor = DagExecutor()
        state = executor.execute_sync(tasks)

        assert state.is_complete
        # br_a 和 br_b 应在第 1 层，merge 在第 2 层
        assert state.execution_order.index("merge") > max(
            state.execution_order.index("br_a"),
            state.execution_order.index("br_b"),
        )


# ============================================================
# 场景 2: 合规判断 — DAG 多步拆解，依赖正确
# ============================================================

class TestPhase4E2ECompliance:
    """合规判断端到端 — DAG 多步拆解"""

    def test_compliance_dag_multi_step(self):
        """合规判断 — 7步 DAG 拆解，依赖链正确"""
        # 模拟"某银行核心一级资本充足率7.8%是否符合要求"的 DAG 拆解
        tasks = [
            DagTask(
                task_id="step1_parse",
                description="解析问题，提取指标=核心一级资本充足率, 数值=7.8%",
                dependencies=[],
            ),
            DagTask(
                task_id="step2_query_threshold",
                description="检索核心一级资本充足率最低要求",
                dependencies=["step1_parse"],
                retrieval_strategy={"channels": ["exact", "metadata"]},
            ),
            DagTask(
                task_id="step3_query_scope",
                description="检索适用主体范围",
                dependencies=["step1_parse"],
                retrieval_strategy={"channels": ["lexical", "dense"]},
            ),
            DagTask(
                task_id="step4_check_version",
                description="检查法规版本是否当前有效",
                dependencies=["step2_query_threshold"],
            ),
            DagTask(
                task_id="step5_compare",
                description="比较实际值7.8%与法定最低要求",
                dependencies=["step2_query_threshold", "step3_query_scope"],
            ),
            DagTask(
                task_id="step6_assemble",
                description="组装合规判断证据",
                dependencies=["step4_check_version", "step5_compare"],
            ),
            DagTask(
                task_id="step7_generate",
                description="生成合规判断回答",
                dependencies=["step6_assemble"],
            ),
        ]

        executor = DagExecutor()
        state = executor.execute_sync(tasks)

        # 7 步全部完成
        assert state.is_complete, "7步 DAG 应全部完成"
        assert len(state.execution_order) == 7

        # 验证关键依赖关系
        order = state.execution_order
        assert order.index("step1_parse") < order.index("step2_query_threshold")
        assert order.index("step1_parse") < order.index("step3_query_scope")
        assert order.index("step2_query_threshold") < order.index("step4_check_version")
        assert order.index("step2_query_threshold") < order.index("step5_compare")
        assert order.index("step3_query_scope") < order.index("step5_compare")
        assert order.index("step4_check_version") < order.index("step6_assemble")
        assert order.index("step5_compare") < order.index("step6_assemble")
        assert order.index("step6_assemble") < order.index("step7_generate")

    def test_compliance_with_evidence_assembly(self):
        """合规判断 — DAG 执行 + 证据组装 + 充分性评分"""
        # 1. 查询编译
        planner = ClaimPlanner()
        query_spec = {"intent": "threshold", "risk_level": "high"}
        slots = planner.plan(query_spec)
        ir = IRBuilder().build(query_spec, slots)

        # 2. DAG 执行（简化为 3 步）
        tasks = [
            DagTask(
                task_id="t1_retrieve",
                description="检索资本充足率要求",
                dependencies=[],
            ),
            DagTask(
                task_id="t2_assemble",
                description="组装证据",
                dependencies=["t1_retrieve"],
            ),
            DagTask(
                task_id="t3_score",
                description="充分性评分",
                dependencies=["t2_assemble"],
            ),
        ]

        executor = DagExecutor()
        state = executor.execute_sync(tasks)
        assert state.is_complete

        # 3. 模拟证据组装
        hits = [
            make_hit(
                content="核心一级资本充足率不得低于5%。系统重要性银行额外满足附加资本要求。",
                score=0.92,
            ),
            make_hit(
                content="适用主体：中华人民共和国境内设立的商业银行。",
                score=0.80,
            ),
        ]

        claims_dict = [
            {"claim_id": s.claim_id, "description": s.description, "slot_type": s.slot_type}
            for s in slots
        ]
        builder = EvidenceBuilder()
        bundle = builder.build(hits, claims_dict)

        filler = SlotFiller()
        filled_slots = filler.fill(slots, bundle.evidence_items)
        bundle.claim_slots = filled_slots

        scorer = SufficiencyScorer()
        result = scorer.score(bundle)

        assert result.score > 0, "充分性评分应大于 0"


# ============================================================
# 场景 3: 跨文件引用 — DAG 多步检索 + 结果回查
# ============================================================

class TestPhase4E2ECrossFile:
    """跨文件引用端到端 — DAG 多步检索"""

    def test_cross_file_dag_with_dependency(self):
        """跨文件引用 — DAG 先检索条款，再回查引用的附件"""
        tasks = [
            DagTask(
                task_id="t1_find_clause",
                description="检索第43条原文",
                dependencies=[],
                retrieval_strategy={"channels": ["exact", "lexical"]},
            ),
            DagTask(
                task_id="t2_extract_reference",
                description="从条款中提取引用（附件2第3表）",
                dependencies=["t1_find_clause"],
                completion_condition="reference_extracted",
            ),
            DagTask(
                task_id="t3_lookup_attachment",
                description="检索附件2第3表",
                dependencies=["t2_extract_reference"],
                retrieval_strategy={"channels": ["table", "metadata"]},
            ),
            DagTask(
                task_id="t4_assemble",
                description="合并条款和附件证据",
                dependencies=["t1_find_clause", "t3_lookup_attachment"],
            ),
        ]

        executor = DagExecutor()
        state = executor.execute_sync(tasks)

        assert state.is_complete
        assert len(state.execution_order) == 4

        # t2 依赖 t1，t3 依赖 t2，t4 依赖 t1 和 t3
        order = state.execution_order
        assert order.index("t1_find_clause") < order.index("t2_extract_reference")
        assert order.index("t2_extract_reference") < order.index("t3_lookup_attachment")
        assert order.index("t3_lookup_attachment") < order.index("t4_assemble")

    def test_cross_file_with_table_channel(self):
        """跨文件引用 — 表格通道检索附件"""
        planner = ClaimPlanner()
        query_spec = {"intent": "table_lookup", "risk_level": "low"}
        slots = planner.plan(query_spec)

        ir = IRBuilder().build(query_spec, slots)
        logical_plan = LogicalPlanner().plan(ir)
        optimizer = PlanOptimizer()
        best_plan = optimizer.optimize(ir)

        assert best_plan is not None
        # 表格查询应包含 table 通道
        primary_stage = best_plan.stages[0]
        assert "table" in primary_stage.channels, "表格查询应包含 table 通道"


# ============================================================
# 场景 4: 推测式取消 — 精确检索直接命中，Dense 分支被取消
# ============================================================

class TestPhase4E2ESpeculativeCancel:
    """推测式取消端到端 — 精确检索命中后取消 Dense 分支"""

    def test_exact_hit_cancels_dense(self):
        """推测式取消 — exact 命中唯一条款 → Dense 被取消"""
        launcher = SpeculativeLauncher()
        canceller = BranchCanceller()

        # 模拟检索函数：exact 命中唯一高分条款，dense 有结果但将被取消
        def mock_retrieval(channel: str):
            if channel == "exact":
                return [
                    make_hit(
                        chunk_id="clause-43",
                        content="第43条：核心一级资本充足率不得低于5%。",
                        score=0.95,
                    ),
                ]
            if channel == "dense":
                return [
                    make_hit(
                        content="资本充足率相关条款...",
                        score=0.75,
                    ),
                ]
            if channel == "metadata":
                return [
                    make_hit(
                        content="《商业银行资本管理办法》第43条",
                        score=0.70,
                        chunk_type="metadata",
                    ),
                ]
            return []

        # T0 分层启动
        grouped = launcher.launch_tiered(
            ["exact", "lexical", "metadata", "dense"],
            mock_retrieval,
        )

        # 收集所有分支
        all_branches = []
        for tier_branches in grouped.values():
            all_branches.extend(tier_branches)

        # T0 通道（exact, lexical, metadata）应已完成
        t0_completed = [
            b for b in grouped["T0"]
            if b.status == BranchStatus.COMPLETED
        ]
        assert len(t0_completed) > 0, "T0 通道应已完成"

        # 执行分支取消（sufficiency 未达标但 exact 命中唯一条款）
        canceller.evaluate_and_cancel(all_branches, sufficiency_score=0.5)

        # exact 分支应保持 COMPLETED
        exact_branches = [b for b in all_branches if b.channel == "exact"]
        assert len(exact_branches) == 1
        assert exact_branches[0].status == BranchStatus.COMPLETED

        # dense 分支应被取消
        dense_branches = [b for b in all_branches if b.channel == "dense"]
        assert len(dense_branches) == 1
        assert dense_branches[0].status == BranchStatus.CANCELLED, \
            "exact 命中唯一条款后，dense 应被取消"

    def test_sufficient_score_cancels_all_pending(self):
        """推测式取消 — 证据充分 → 取消所有未完成分支"""
        launcher = SpeculativeLauncher()
        canceller = BranchCanceller()

        def mock_retrieval(channel: str):
            if channel == "exact":
                return [make_hit(content="精确命中", score=0.95)]
            if channel == "lexical":
                return [make_hit(content="关键词命中", score=0.88)]
            return []

        grouped = launcher.launch_tiered(
            ["exact", "lexical", "metadata", "dense", "table"],
            mock_retrieval,
        )

        all_branches = []
        for tier_branches in grouped.values():
            all_branches.extend(tier_branches)

        # 充分性达标 → 取消所有未完成
        canceller.evaluate_and_cancel(all_branches, sufficiency_score=0.90)

        # 所有未完成分支应被取消
        pending_after = [
            b for b in all_branches
            if b.status in (BranchStatus.LAUNCHED, BranchStatus.RUNNING)
        ]
        assert len(pending_after) == 0, "充分性达标后所有未完成分支应被取消"

    def test_early_stop_evaluator(self):
        """推测式取消 — 早停评估器判断是否停止"""
        evaluator = EarlyStopEvaluator()

        # 充分性达标 → 停止
        result = evaluator.evaluate(
            branches=[], sufficiency_score=0.90, previous_score=0.5,
        )
        assert result.should_stop
        assert result.reason == "sufficient"

        # 边际增益过低 → 停止
        result = evaluator.evaluate(
            branches=[], sufficiency_score=0.70, previous_score=0.68,
        )
        assert result.should_stop
        assert result.reason == "marginal_gain_low"

        # 充分性不足且增益足够 → 继续
        result = evaluator.evaluate(
            branches=[], sufficiency_score=0.50, previous_score=0.30,
        )
        assert not result.should_stop

    def test_result_stream_collection(self):
        """推测式取消 — 结果流收集器整合多分支结果"""
        stream = ResultStream()

        # 模拟多分支结果（按分支添加）
        stream.add_results("exact-abc123", [
            {"chunk_id": "c1", "content": "结果1", "score": 0.9},
        ])
        stream.add_results("dense-def456", [
            {"chunk_id": "c2", "content": "结果2", "score": 0.8},
            {"chunk_id": "c3", "content": "结果3", "score": 0.85},
        ])

        all_results = stream.get_all_results()
        assert len(all_results) == 3

        # 结果应按分数降序
        scores = [r["score"] for r in all_results]
        assert scores == sorted(scores, reverse=True)

        # 去重后仍应保留全部（chunk_id 不同）
        deduped = stream.merge_and_dedupe()
        assert len(deduped) == 3


# ============================================================
# 场景 5: 检查点恢复 — 模拟中途失败后恢复
# ============================================================

class TestPhase4E2ECheckpointRecovery:
    """检查点恢复端到端 — 从检查点继续，不重复已完成任务"""

    def test_checkpoint_save_and_restore(self):
        """检查点恢复 — 保存 → 恢复 → 状态正确"""
        manager = CheckpointManager()  # 内存模式

        # 模拟执行中途的检查点
        cp = Checkpoint(
            checkpoint_id="",
            session_id="sess-001",
            request_id="req-001",
            state="RETRIEVING",
            query_spec={"intent": "threshold", "risk_level": "medium"},
            query_plan={"plan_id": "pp-001", "stages": []},
            dag_state={
                "tasks": [
                    {"task_id": "t1", "status": "completed"},
                    {"task_id": "t2", "status": "completed"},
                    {"task_id": "t3", "status": "pending"},
                ],
                "execution_order": ["t1", "t2"],
                "is_complete": False,
                "has_failure": False,
            },
            evidence_bundle={"evidence_count": 2},
            budget_consumed={"retrieval_rounds": 1},
        )

        # 保存
        cp_id = manager.save(cp)
        assert cp_id, "应返回 checkpoint_id"

        # 加载最新
        loaded = manager.load_latest("sess-001", "req-001")
        assert loaded is not None
        assert loaded.state == "RETRIEVING"
        assert loaded.session_id == "sess-001"

    def test_checkpoint_recovery_no_repeat(self):
        """检查点恢复 — 恢复后不重复已完成任务"""
        manager = CheckpointManager()
        recovery = RecoveryManager(manager)

        # 保存一个中途检查点（2 个任务完成，1 个待执行）
        cp = Checkpoint(
            checkpoint_id="",
            session_id="sess-002",
            request_id="req-002",
            state="RETRIEVING",
            query_spec={"intent": "comparison"},
            query_plan={},
            dag_state={
                "tasks": [
                    {"task_id": "t1", "status": "completed"},
                    {"task_id": "t2", "status": "completed"},
                    {"task_id": "t3", "status": "pending"},
                ],
            },
        )
        manager.save(cp)

        # 恢复
        result = recovery.recover("sess-002", "req-002")

        assert result.success
        assert result.recovered_state == "RETRIEVING"
        assert "t1" in result.completed_task_ids
        assert "t2" in result.completed_task_ids
        assert "t3" in result.pending_task_ids
        assert len(result.completed_task_ids) == 2
        assert len(result.pending_task_ids) == 1

    def test_checkpoint_version_management(self):
        """检查点恢复 — 多版本检查点，保留最近 N 个"""
        manager = CheckpointManager()

        # 保存 5 个版本
        for i in range(5):
            cp = Checkpoint(
                checkpoint_id="",
                session_id="sess-003",
                request_id="req-003",
                state=f"STATE_{i}",
                query_spec={},
                query_plan={},
            )
            manager.save(cp)

        # 应只保留最近 3 个版本
        versions = manager.list_versions("sess-003", "req-003")
        assert len(versions) <= manager.MAX_VERSIONS

        # 最新版本应可加载
        latest = manager.load_latest("sess-003", "req-003")
        assert latest is not None
        assert latest.state == "STATE_4"

    def test_checkpoint_no_checkpoint_returns_none(self):
        """检查点恢复 — 无检查点时返回失败"""
        manager = CheckpointManager()
        recovery = RecoveryManager(manager)

        result = recovery.recover("nonexistent", "nonexistent")
        assert not result.success
        assert result.error is not None

    def test_checkpoint_with_dag_state(self):
        """检查点恢复 — DAG 状态完整保存和恢复"""
        manager = CheckpointManager()

        # 构造 DAG 状态
        tasks = [
            DagTask(task_id="dt1", description="任务1", status=TaskStatus.COMPLETED),
            DagTask(task_id="dt2", description="任务2", status=TaskStatus.COMPLETED),
            DagTask(task_id="dt3", description="任务3", status=TaskStatus.PENDING),
        ]
        dag_state = DagState(
            tasks=tasks,
            execution_order=["dt1", "dt2"],
            is_complete=False,
            has_failure=False,
        )

        cp = Checkpoint(
            checkpoint_id="",
            session_id="sess-004",
            request_id="req-004",
            state="EVIDENCE_ASSEMBLING",
            query_spec={"intent": "threshold"},
            query_plan={"plan_id": "pp-test"},
            dag_state=dag_state.to_dict(),
        )
        manager.save(cp)

        # 恢复
        loaded = manager.load_latest("sess-004", "req-004")
        assert loaded is not None
        assert loaded.dag_state is not None

        # 验证 DAG 状态
        restored_tasks = loaded.dag_state.get("tasks", [])
        assert len(restored_tasks) == 3

        # 从 DAG 状态恢复任务
        recovery = RecoveryManager(manager)
        result = recovery.recover("sess-004", "req-004")
        assert result.success
        assert "dt1" in result.completed_task_ids
        assert "dt2" in result.completed_task_ids
        assert "dt3" in result.pending_task_ids


# ============================================================
# 场景 6: 计划缓存命中 — 相似问题第二次执行命中缓存
# ============================================================

class TestPhase4E2EPlanCache:
    """计划缓存命中端到端 — 相似问题第二次执行跳过编译"""

    def test_plan_cache_hit_on_second_query(self):
        """计划缓存 — 相同查询第二次命中缓存"""
        cache = PlanCache()  # 内存模式
        key_gen = CacheKeyGenerator()

        # 第一次查询
        planner = ClaimPlanner()
        query_spec = {"intent": "threshold", "risk_level": "medium"}
        slots = planner.plan(query_spec)
        ir = IRBuilder().build(query_spec, slots)

        context = CacheContext(
            user_permissions="analyst",
            index_epoch="kb-2026-07",
        )

        cache_key = key_gen.make_key(ir, context)

        # 第一次：缓存未命中，编译并缓存
        cached_plan = cache.get(cache_key)
        assert cached_plan is None, "第一次应未命中"

        optimizer = PlanOptimizer()
        best_plan = optimizer.optimize(ir)
        cache.put(cache_key, best_plan, epoch="kb-2026-07")

        # 第二次：相同查询应命中缓存
        cached_plan = cache.get(cache_key)
        assert cached_plan is not None, "第二次应命中缓存"
        assert cached_plan.intent == "threshold"
        assert cached_plan.plan_id == best_plan.plan_id

        # 验证统计
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_plan_cache_different_intent_misses(self):
        """计划缓存 — 不同意图查询不命中"""
        cache = PlanCache()
        key_gen = CacheKeyGenerator()

        # 缓存 threshold 查询
        spec1 = {"intent": "threshold", "risk_level": "medium"}
        slots1 = ClaimPlanner().plan(spec1)
        ir1 = IRBuilder().build(spec1, slots1)
        ctx = CacheContext(index_epoch="kb-2026-07")
        key1 = key_gen.make_key(ir1, ctx)

        plan1 = PlanOptimizer().optimize(ir1)
        cache.put(key1, plan1)

        # definition 查询应不命中
        spec2 = {"intent": "definition", "risk_level": "low"}
        slots2 = ClaimPlanner().plan(spec2)
        ir2 = IRBuilder().build(spec2, slots2)
        key2 = key_gen.make_key(ir2, ctx)

        cached = cache.get(key2)
        assert cached is None, "不同意图应不命中"

    def test_plan_cache_invalidate_by_epoch(self):
        """计划缓存 — 知识库版本更新后缓存失效"""
        cache = PlanCache()
        key_gen = CacheKeyGenerator()

        spec = {"intent": "clause_query", "risk_level": "low"}
        slots = ClaimPlanner().plan(spec)
        ir = IRBuilder().build(spec, slots)

        ctx_old = CacheContext(index_epoch="kb-2026-07")
        key = key_gen.make_key(ir, ctx_old)

        plan = PlanOptimizer().optimize(ir)
        cache.put(key, plan, epoch="kb-2026-07")

        # 命中
        assert cache.get(key) is not None

        # 知识库版本更新，失效旧缓存
        count = cache.invalidate_by_epoch("kb-2026-07")
        assert count == 1, "应失效 1 个条目"

        # 不再命中
        assert cache.get(key) is None

    def test_plan_cache_full_pipeline_with_cache(self):
        """计划缓存 — 完整流水线：编译 → 缓存 → 命中 → 执行"""
        cache = PlanCache()
        key_gen = CacheKeyGenerator()

        spec = {"intent": "comparison", "risk_level": "high"}
        slots = ClaimPlanner().plan(spec)
        ir = IRBuilder().build(spec, slots)
        ctx = CacheContext(index_epoch="kb-2026-07", user_permissions="analyst")
        cache_key = key_gen.make_key(ir, ctx)

        # 第一次：编译 + 缓存
        optimizer = PlanOptimizer()
        plan1 = optimizer.optimize(ir)
        assert plan1 is not None
        cache.put(cache_key, plan1, epoch="kb-2026-07")

        # 第二次：命中缓存
        plan2 = cache.get(cache_key)
        assert plan2 is not None
        assert plan2.plan_id == plan1.plan_id

        # 验证缓存的计划可以校验通过
        validation = PlanValidator().validate(plan2, ir)
        assert validation.is_valid, "缓存的计划应校验通过"

    def test_cost_estimator_with_optimized_plan(self):
        """计划缓存 — 优化后的计划成本可估计"""
        spec = {"intent": "threshold", "risk_level": "medium"}
        slots = ClaimPlanner().plan(spec)
        ir = IRBuilder().build(spec, slots)

        optimizer = PlanOptimizer()
        candidates = optimizer.generate_candidates(ir)

        # 所有候选计划应可估计成本
        estimator = CostEstimator()
        for plan in candidates:
            cost = estimator.estimate(plan)
            assert cost.total_cost > 0, "每个候选计划应有正成本"
            assert cost.latency_ms > 0
            assert "breakdown" in cost.__dict__ or hasattr(cost, "breakdown")

        # 最优计划成本应 <= 任一候选
        best = optimizer.select_best(candidates)
        best_cost = estimator.estimate(best)
        for plan in candidates:
            plan_cost = estimator.estimate(plan)
            assert best_cost.total_cost <= plan_cost.total_cost, \
                "最优计划成本应不高于任一候选"


# ============================================================
# 综合场景: 完整 Phase 4 流水线
# ============================================================

class TestPhase4E2EFullPipeline:
    """Phase 4 完整流水线集成测试"""

    def test_full_pipeline_compile_dag_speculative_checkpoint(self):
        """完整流水线 — 编译 → DAG → 推测式检索 → 检查点"""
        # 1. 声明规划 + 查询编译 + 优化
        planner = ClaimPlanner()
        query_spec = {"intent": "comparison", "risk_level": "high"}
        slots = planner.plan(query_spec)
        ir = IRBuilder().build(query_spec, slots)

        optimizer = PlanOptimizer()
        best_plan = optimizer.optimize(ir)
        assert best_plan is not None

        # 计划校验
        validation = PlanValidator().validate(best_plan, ir)
        assert validation.is_valid

        # 2. DAG 执行
        dag_tasks = [
            DagTask(
                task_id="dag_t1",
                description="检索分支A",
                dependencies=[],
            ),
            DagTask(
                task_id="dag_t2",
                description="检索分支B",
                dependencies=[],
            ),
            DagTask(
                task_id="dag_t3",
                description="合并结果",
                dependencies=["dag_t1", "dag_t2"],
            ),
        ]
        dag_executor = DagExecutor()
        dag_state = dag_executor.execute_sync(dag_tasks)
        assert dag_state.is_complete

        # 3. 推测式检索
        launcher = SpeculativeLauncher()
        channels = best_plan.stages[0].channels if best_plan.stages else ["lexical", "dense"]

        def mock_retrieval(channel: str):
            return [make_hit(content=f"{channel} 通道结果", score=0.85)]

        branches = launcher.launch(channels, mock_retrieval)
        assert len(branches) > 0

        completed = [b for b in branches if b.status == BranchStatus.COMPLETED]
        assert len(completed) > 0, "至少有一个分支应完成"

        # 4. 检查点保存
        cp_manager = CheckpointManager()
        checkpoint = Checkpoint(
            checkpoint_id="",
            session_id="sess-full",
            request_id="req-full",
            state="EVIDENCE_ASSEMBLING",
            query_spec=query_spec,
            query_plan=best_plan.to_dict(),
            dag_state=dag_state.to_dict(),
        )
        cp_id = cp_manager.save(checkpoint)
        assert cp_id

        # 5. 检查点恢复
        recovery = RecoveryManager(cp_manager)
        result = recovery.recover("sess-full", "req-full")
        assert result.success
        assert result.recovered_state == "EVIDENCE_ASSEMBLING"
        assert "dag_t1" in result.completed_task_ids
        assert "dag_t2" in result.completed_task_ids
        assert "dag_t3" in result.completed_task_ids

    def test_dag_failure_propagation_with_checkpoint(self):
        """完整流水线 — DAG 任务失败 → 传播取消 → 检查点保存部分结果"""

        async def failing_executor(task: DagTask) -> DagTask:
            """模拟执行器：t2 失败"""
            if task.task_id == "t2_fail":
                task.status = TaskStatus.FAILED
                task.failure_reason = "检索超时"
                return task
            task.status = TaskStatus.COMPLETED
            task.result = {"status": "ok"}
            return task

        tasks = [
            DagTask(task_id="t1", description="任务1", dependencies=[]),
            DagTask(task_id="t2_fail", description="任务2（将失败）", dependencies=["t1"]),
            DagTask(task_id="t3", description="任务3（依赖t2）", dependencies=["t2_fail"]),
            DagTask(task_id="t4", description="任务4（依赖t3）", dependencies=["t3"]),
        ]

        executor = DagExecutor()
        state = executor.execute_sync(tasks, failing_executor)

        # DAG 应有失败
        assert state.has_failure, "应有失败任务"

        # t1 应完成
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t1"].status == TaskStatus.COMPLETED

        # t2 应失败
        assert task_map["t2_fail"].status == TaskStatus.FAILED

        # t3 和 t4 应被取消（失败传播）
        assert task_map["t3"].status == TaskStatus.CANCELLED, \
            "t3 应因 t2 失败被取消"
        assert task_map["t4"].status == TaskStatus.CANCELLED, \
            "t4 应因 t3 被取消而取消"

        # 保存检查点（部分结果）
        cp_manager = CheckpointManager()
        checkpoint = Checkpoint(
            checkpoint_id="",
            session_id="sess-fail",
            request_id="req-fail",
            state="RETRYING",
            query_spec={"intent": "threshold"},
            query_plan={},
            dag_state=state.to_dict(),
        )
        cp_manager.save(checkpoint)

        # 恢复：t1 已完成，其余待执行
        recovery = RecoveryManager(cp_manager)
        result = recovery.recover("sess-fail", "req-fail")
        assert result.success
        assert "t1" in result.completed_task_ids
        # 失败和取消的任务应出现在 pending 中（需重新执行）
        assert "t2_fail" in result.pending_task_ids

    def test_plan_cache_with_dag_execution(self):
        """完整流水线 — 计划缓存 + DAG 执行"""
        cache = PlanCache()
        key_gen = CacheKeyGenerator()

        # 第一次执行：编译 + 缓存 + DAG
        spec = {"intent": "table_lookup", "risk_level": "low"}
        slots = ClaimPlanner().plan(spec)
        ir = IRBuilder().build(spec, slots)
        ctx = CacheContext(index_epoch="kb-2026-07")
        cache_key = key_gen.make_key(ir, ctx)

        # 编译并缓存
        plan1 = PlanOptimizer().optimize(ir)
        cache.put(cache_key, plan1, epoch="kb-2026-07")

        # DAG 执行
        tasks = [
            DagTask(task_id="dt1", description="表格检索", dependencies=[]),
            DagTask(task_id="dt2", description="结果组装", dependencies=["dt1"]),
        ]
        state = DagExecutor().execute_sync(tasks)
        assert state.is_complete

        # 第二次执行：命中缓存 + DAG
        plan2 = cache.get(cache_key)
        assert plan2 is not None, "第二次应命中缓存"
        assert plan2.plan_id == plan1.plan_id

        # 用缓存的计划重新执行 DAG
        state2 = DagExecutor().execute_sync(tasks)
        assert state2.is_complete
