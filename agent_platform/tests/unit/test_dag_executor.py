"""
DAG 执行器单元测试（M4.2）

测试用例（对应开发计划测试用例表）:
  - 线性依赖: t1→t2→t3 顺序执行，全部完成
  - 并行无依赖: t1, t2 独立并行执行，全部完成
  - 菱形依赖: t1→t2,t1→t3,t2+t3→t4，全部完成
  - 任务失败: t2 失败 → t3 被取消，返回部分结果
  - 循环依赖: t1→t2→t1 → 报错拒绝执行
  - 澄清阻塞: t2 requires_clarification → BLOCKED，等待用户输入

额外测试用例:
  - 空任务列表
  - 单任务无依赖
  - 多层依赖（t1→t2→t3→t4→t5）
  - DagTask.to_dict/from_dict 序列化往返
  - DagState.to_dict/from_dict 序列化往返
  - DependencyResolver.get_dependents
  - DependencyResolver.get_ready_tasks
  - 自定义 executor_func（异步函数，模拟实际执行）
  - 任务超时（ParallelScheduler.execute_with_timeout）
  - 异常隔离（单任务异常不影响同层其他任务）
"""

import asyncio

import pytest

from agent_platform.orchestration.dag_executor import (
    DagExecutor,
    DagState,
    DagTask,
    DependencyResolver,
    ParallelScheduler,
    TaskStatus,
)


# ============================================================
# 工厂函数 — 构造典型 DAG 结构
# ============================================================


def make_linear_tasks():
    """线性依赖: t1 → t2 → t3"""
    return [
        DagTask(task_id="t1", description="第一步"),
        DagTask(task_id="t2", description="第二步", dependencies=["t1"]),
        DagTask(task_id="t3", description="第三步", dependencies=["t2"]),
    ]


def make_parallel_tasks():
    """并行无依赖: t1, t2 独立"""
    return [
        DagTask(task_id="t1", description="独立任务 A"),
        DagTask(task_id="t2", description="独立任务 B"),
    ]


def make_diamond_tasks():
    """菱形依赖: t1 → (t2 || t3) → t4"""
    return [
        DagTask(task_id="t1", description="起点"),
        DagTask(task_id="t2", description="分支 A", dependencies=["t1"]),
        DagTask(task_id="t3", description="分支 B", dependencies=["t1"]),
        DagTask(task_id="t4", description="汇合", dependencies=["t2", "t3"]),
    ]


def make_multi_layer_tasks():
    """多层依赖: t1 → t2 → t3 → t4 → t5"""
    return [
        DagTask(task_id="t1", description="第 1 层"),
        DagTask(task_id="t2", description="第 2 层", dependencies=["t1"]),
        DagTask(task_id="t3", description="第 3 层", dependencies=["t2"]),
        DagTask(task_id="t4", description="第 4 层", dependencies=["t3"]),
        DagTask(task_id="t5", description="第 5 层", dependencies=["t4"]),
    ]


def make_cycle_tasks():
    """循环依赖: t1 → t2 → t1"""
    return [
        DagTask(task_id="t1", description="环节点 1", dependencies=["t2"]),
        DagTask(task_id="t2", description="环节点 2", dependencies=["t1"]),
    ]


def make_failure_tasks():
    """失败传播结构: t1 → t2 → t3（t2 将失败）"""
    return [
        DagTask(task_id="t1", description="第一步"),
        DagTask(task_id="t2", description="第二步（将失败）", dependencies=["t1"]),
        DagTask(task_id="t3", description="第三步（应被取消）", dependencies=["t2"]),
    ]


# ============================================================
# 自定义执行器工厂
# ============================================================


async def completing_executor(task: DagTask) -> DagTask:
    """将任务标记为 COMPLETED 的自定义执行器"""
    task.status = TaskStatus.COMPLETED
    task.result = {"executed": task.task_id}
    return task


def make_failing_executor(task_id_to_fail: str):
    """构造一个执行器：指定任务失败，其余完成"""

    async def _executor(task: DagTask) -> DagTask:
        if task.task_id == task_id_to_fail:
            task.status = TaskStatus.FAILED
            task.failure_reason = "模拟执行失败"
            return task
        task.status = TaskStatus.COMPLETED
        task.result = {"executed": task.task_id}
        return task

    return _executor


def make_blocking_executor():
    """构造一个执行器：requires_clarification 的任务标记为 BLOCKED"""

    async def _executor(task: DagTask) -> DagTask:
        if task.requires_clarification:
            task.status = TaskStatus.BLOCKED
            task.failure_reason = "需要用户澄清输入"
            return task
        task.status = TaskStatus.COMPLETED
        task.result = {"executed": task.task_id}
        return task

    return _executor


async def slow_executor(task: DagTask) -> DagTask:
    """模拟耗时执行的执行器"""
    await asyncio.sleep(0.3)
    task.status = TaskStatus.COMPLETED
    task.result = {"slow": task.task_id}
    return task


async def raising_executor(task: DagTask) -> DagTask:
    """总是抛出异常的执行器"""
    raise RuntimeError(f"执行异常: {task.task_id}")


# ============================================================
# TaskStatus 枚举与数据模型测试
# ============================================================


class TestTaskStatus:
    """TaskStatus 枚举测试"""

    def test_status_values(self):
        """枚举值字符串正确"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.READY.value == "ready"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.BLOCKED.value == "blocked"

    def test_status_is_str(self):
        """继承 str，可直接与字符串比较"""
        assert TaskStatus.COMPLETED == "completed"
        assert isinstance(TaskStatus.PENDING, str)


class TestDagTaskModel:
    """DagTask 数据模型测试"""

    def test_default_status_is_pending(self):
        """新建任务默认状态为 PENDING"""
        task = DagTask(task_id="t1", description="测试任务")
        assert task.status == TaskStatus.PENDING

    def test_default_dependencies_empty(self):
        """未指定 dependencies 时默认为空列表"""
        task = DagTask(task_id="t1", description="测试任务")
        assert task.dependencies == []

    def test_task_with_dependencies(self):
        """显式声明 dependencies"""
        task = DagTask(
            task_id="t2", description="依赖任务", dependencies=["t1"]
        )
        assert task.dependencies == ["t1"]

    def test_task_default_fields(self):
        """默认字段值正确"""
        task = DagTask(task_id="t1", description="测试任务")
        assert task.input_constraints == {}
        assert task.retrieval_strategy == {}
        assert task.completion_condition == ""
        assert task.evidence_ids == []
        assert task.result == {}
        assert task.failure_reason is None
        assert task.allow_parallel is True
        assert task.requires_clarification is False
        assert task.started_at is None
        assert task.completed_at is None

    def test_to_dict_contains_all_fields(self):
        """to_dict 包含全部字段"""
        task = DagTask(
            task_id="t1",
            description="检索任务",
            input_constraints={"top_k": 5},
            dependencies=["t0"],
            retrieval_strategy={"strategy": "vector"},
            completion_condition="命中 3 条",
            evidence_ids=["e1", "e2"],
            result={"hits": 3},
            failure_reason=None,
            allow_parallel=False,
            requires_clarification=True,
            status=TaskStatus.RUNNING,
            started_at="2024-01-01T00:00:00Z",
            completed_at=None,
        )
        d = task.to_dict()
        assert d["task_id"] == "t1"
        assert d["description"] == "检索任务"
        assert d["input_constraints"] == {"top_k": 5}
        assert d["dependencies"] == ["t0"]
        assert d["retrieval_strategy"] == {"strategy": "vector"}
        assert d["completion_condition"] == "命中 3 条"
        assert d["evidence_ids"] == ["e1", "e2"]
        assert d["result"] == {"hits": 3}
        assert d["failure_reason"] is None
        assert d["allow_parallel"] is False
        assert d["requires_clarification"] is True
        assert d["status"] == "running"
        assert d["started_at"] == "2024-01-01T00:00:00Z"
        assert d["completed_at"] is None

    def test_from_dict_roundtrip(self):
        """to_dict → from_dict 往返保持一致"""
        task = DagTask(
            task_id="t1",
            description="往返测试",
            dependencies=["t0"],
            retrieval_strategy={"strategy": "hybrid"},
            evidence_ids=["e1"],
            result={"ok": True},
            failure_reason="原因",
            status=TaskStatus.FAILED,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:01:00Z",
        )
        restored = DagTask.from_dict(task.to_dict())
        assert restored.task_id == task.task_id
        assert restored.description == task.description
        assert restored.dependencies == task.dependencies
        assert restored.retrieval_strategy == task.retrieval_strategy
        assert restored.evidence_ids == task.evidence_ids
        assert restored.result == task.result
        assert restored.failure_reason == task.failure_reason
        assert restored.status == task.status
        assert restored.started_at == task.started_at
        assert restored.completed_at == task.completed_at

    def test_from_dict_status_from_string(self):
        """from_dict 从字符串恢复状态枚举"""
        data = {"task_id": "t1", "description": "", "status": "completed"}
        task = DagTask.from_dict(data)
        assert task.status == TaskStatus.COMPLETED

    def test_from_dict_with_minimal_data(self):
        """from_dict 仅含必需字段时使用默认值"""
        task = DagTask.from_dict({"task_id": "t1", "description": "最小"})
        assert task.task_id == "t1"
        assert task.dependencies == []
        assert task.status == TaskStatus.PENDING
        assert task.allow_parallel is True

    def test_to_dict_dependencies_is_copy(self):
        """to_dict 返回的 dependencies 是副本，修改不影响原对象"""
        task = DagTask(task_id="t1", description="t", dependencies=["t0"])
        d = task.to_dict()
        d["dependencies"].append("t9")
        assert task.dependencies == ["t0"]


class TestDagStateModel:
    """DagState 状态快照测试"""

    def test_default_empty_state(self):
        """默认 DagState 为空"""
        state = DagState()
        assert state.tasks == []
        assert state.execution_order == []
        assert state.is_complete is False
        assert state.has_failure is False

    def test_to_dict_structure(self):
        """to_dict 结构正确"""
        task = DagTask(task_id="t1", description="t", status=TaskStatus.COMPLETED)
        state = DagState(
            tasks=[task],
            execution_order=["t1"],
            is_complete=True,
            has_failure=False,
        )
        d = state.to_dict()
        assert isinstance(d["tasks"], list)
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["task_id"] == "t1"
        assert d["execution_order"] == ["t1"]
        assert d["is_complete"] is True
        assert d["has_failure"] is False

    def test_from_dict_roundtrip(self):
        """to_dict → from_dict 往返保持一致"""
        tasks = [
            DagTask(task_id="t1", description="a", status=TaskStatus.COMPLETED),
            DagTask(
                task_id="t2",
                description="b",
                dependencies=["t1"],
                status=TaskStatus.FAILED,
                failure_reason="失败",
            ),
        ]
        state = DagState(
            tasks=tasks,
            execution_order=["t1", "t2"],
            is_complete=False,
            has_failure=True,
        )
        restored = DagState.from_dict(state.to_dict())
        assert len(restored.tasks) == 2
        assert restored.tasks[0].task_id == "t1"
        assert restored.tasks[1].task_id == "t2"
        assert restored.tasks[1].status == TaskStatus.FAILED
        assert restored.tasks[1].failure_reason == "失败"
        assert restored.execution_order == ["t1", "t2"]
        assert restored.is_complete is False
        assert restored.has_failure is True

    def test_from_dict_empty(self):
        """from_dict 空字典恢复为空状态"""
        state = DagState.from_dict({})
        assert state.tasks == []
        assert state.execution_order == []
        assert state.is_complete is False
        assert state.has_failure is False

    def test_repr(self):
        """__repr__ 包含关键信息"""
        state = DagState(tasks=[DagTask(task_id="t1", description="")], execution_order=["t1"])
        repr_str = repr(state)
        assert "DagState" in repr_str
        assert "complete" in repr_str


# ============================================================
# DependencyResolver 测试
# ============================================================


class TestTopologicalSort:
    """拓扑排序测试"""

    def test_linear_sort(self):
        """线性依赖排序为三层"""
        resolver = DependencyResolver()
        layers = resolver.topological_sort(make_linear_tasks())
        assert layers == [["t1"], ["t2"], ["t3"]]

    def test_parallel_sort(self):
        """无依赖任务同层"""
        resolver = DependencyResolver()
        layers = resolver.topological_sort(make_parallel_tasks())
        assert layers == [["t1", "t2"]]

    def test_diamond_sort(self):
        """菱形依赖分三层"""
        resolver = DependencyResolver()
        layers = resolver.topological_sort(make_diamond_tasks())
        assert layers == [["t1"], ["t2", "t3"], ["t4"]]

    def test_multi_layer_sort(self):
        """多层依赖分五层"""
        resolver = DependencyResolver()
        layers = resolver.topological_sort(make_multi_layer_tasks())
        assert layers == [["t1"], ["t2"], ["t3"], ["t4"], ["t5"]]

    def test_empty_tasks(self):
        """空任务列表返回空层"""
        resolver = DependencyResolver()
        assert resolver.topological_sort([]) == []

    def test_single_task(self):
        """单任务一层"""
        resolver = DependencyResolver()
        layers = resolver.topological_sort([DagTask(task_id="t1", description="x")])
        assert layers == [["t1"]]

    def test_cycle_raises_value_error(self):
        """循环依赖抛出 ValueError"""
        resolver = DependencyResolver()
        with pytest.raises(ValueError, match="循环依赖"):
            resolver.topological_sort(make_cycle_tasks())

    def test_same_layer_sorted_by_id(self):
        """同层任务按 task_id 排序，保证输出稳定"""
        resolver = DependencyResolver()
        tasks = [
            DagTask(task_id="c", description=""),
            DagTask(task_id="a", description=""),
            DagTask(task_id="b", description=""),
        ]
        layers = resolver.topological_sort(tasks)
        assert layers == [["a", "b", "c"]]


class TestHasCycle:
    """环检测测试"""

    def test_no_cycle_in_linear(self):
        """线性依赖无环"""
        resolver = DependencyResolver()
        assert resolver.has_cycle(make_linear_tasks()) is False

    def test_no_cycle_in_diamond(self):
        """菱形依赖无环"""
        resolver = DependencyResolver()
        assert resolver.has_cycle(make_diamond_tasks()) is False

    def test_cycle_detected(self):
        """t1→t2→t1 检测到环"""
        resolver = DependencyResolver()
        assert resolver.has_cycle(make_cycle_tasks()) is True

    def test_empty_no_cycle(self):
        """空任务列表无环"""
        resolver = DependencyResolver()
        assert resolver.has_cycle([]) is False

    def test_self_dependency_not_cycle(self):
        """自依赖被忽略，不构成环"""
        resolver = DependencyResolver()
        task = DagTask(task_id="t1", description="自环", dependencies=["t1"])
        assert resolver.has_cycle([task]) is False

    def test_longer_cycle_detected(self):
        """三节点环 t1→t2→t3→t1 检测到"""
        resolver = DependencyResolver()
        tasks = [
            DagTask(task_id="t1", description="", dependencies=["t3"]),
            DagTask(task_id="t2", description="", dependencies=["t1"]),
            DagTask(task_id="t3", description="", dependencies=["t2"]),
        ]
        assert resolver.has_cycle(tasks) is True


class TestGetReadyTasks:
    """就绪任务筛选测试"""

    def test_initial_ready_tasks(self):
        """初始仅有无依赖任务就绪"""
        resolver = DependencyResolver()
        ready = resolver.get_ready_tasks(make_linear_tasks())
        assert ready == ["t1"]

    def test_none_ready_when_all_blocked(self):
        """所有任务均有未完成依赖时无就绪任务"""
        resolver = DependencyResolver()
        tasks = [
            DagTask(task_id="t2", description="", dependencies=["t1"]),
        ]
        assert resolver.get_ready_tasks(tasks) == []

    def test_ready_after_dependency_completed(self):
        """依赖完成后后继任务变为就绪"""
        resolver = DependencyResolver()
        tasks = make_linear_tasks()
        # t1 完成后 t2 应就绪
        tasks[0].status = TaskStatus.COMPLETED
        ready = resolver.get_ready_tasks(tasks)
        assert ready == ["t2"]

    def test_ready_excludes_non_pending(self):
        """非 PENDING 状态的任务不算就绪"""
        resolver = DependencyResolver()
        tasks = make_parallel_tasks()
        tasks[0].status = TaskStatus.COMPLETED
        ready = resolver.get_ready_tasks(tasks)
        # 仅 t2 仍为 PENDING
        assert ready == ["t2"]

    def test_empty_tasks(self):
        """空任务列表无就绪任务"""
        resolver = DependencyResolver()
        assert resolver.get_ready_tasks([]) == []


class TestGetDependents:
    """后继查询测试"""

    def test_get_dependents_linear(self):
        """线性依赖中 t1 的后继为 t2"""
        resolver = DependencyResolver()
        tasks = make_linear_tasks()
        assert resolver.get_dependents("t1", tasks) == ["t2"]

    def test_get_dependents_diamond(self):
        """菱形依赖中 t1 的后继为 t2、t3"""
        resolver = DependencyResolver()
        tasks = make_diamond_tasks()
        deps = resolver.get_dependents("t1", tasks)
        assert set(deps) == {"t2", "t3"}

    def test_get_dependents_leaf(self):
        """叶子节点（无后继）返回空列表"""
        resolver = DependencyResolver()
        tasks = make_diamond_tasks()
        assert resolver.get_dependents("t4", tasks) == []

    def test_get_dependents_excludes_self(self):
        """查询结果不含自身"""
        resolver = DependencyResolver()
        task = DagTask(
            task_id="t1", description="自依赖", dependencies=["t1"]
        )
        assert resolver.get_dependents("t1", [task]) == []


# ============================================================
# DagExecutor 基础测试
# ============================================================


class TestDagExecutorBasics:
    """DagExecutor 基础功能测试"""

    def test_empty_tasks_returns_empty_state(self):
        """空任务列表返回空 DagState"""
        executor = DagExecutor()
        state = executor.execute_sync([])
        assert state.tasks == []
        assert state.execution_order == []
        assert state.is_complete is False
        assert state.has_failure is False

    def test_single_task_no_dependencies(self):
        """单任务无依赖 → 完成"""
        executor = DagExecutor()
        tasks = [DagTask(task_id="t1", description="唯一任务")]
        state = executor.execute_sync(tasks)
        assert state.is_complete is True
        assert state.has_failure is False
        assert state.execution_order == ["t1"]
        assert state.tasks[0].status == TaskStatus.COMPLETED

    def test_default_executor_marks_completed(self):
        """默认 mock 执行器将任务标记为 COMPLETED 并写入 result"""
        executor = DagExecutor()
        tasks = [DagTask(task_id="t1", description="t")]
        state = executor.execute_sync(tasks)
        assert state.tasks[0].status == TaskStatus.COMPLETED
        assert state.tasks[0].result == {"status": "mock_completed"}

    def test_default_executor_sets_timestamps(self):
        """默认执行器记录开始与完成时间"""
        executor = DagExecutor()
        tasks = [DagTask(task_id="t1", description="t")]
        state = executor.execute_sync(tasks)
        assert state.tasks[0].started_at is not None
        assert state.tasks[0].completed_at is not None


# ============================================================
# DAG 执行模式测试（对应开发计划测试用例表）
# ============================================================


class TestDagExecutionPatterns:
    """DAG 执行模式测试"""

    def test_linear_dependency_all_completed(self):
        """线性依赖 t1→t2→t3 顺序执行，全部完成"""
        executor = DagExecutor()
        state = executor.execute_sync(make_linear_tasks())
        assert state.is_complete is True
        assert state.has_failure is False
        assert state.execution_order == ["t1", "t2", "t3"]
        for task in state.tasks:
            assert task.status == TaskStatus.COMPLETED

    def test_parallel_tasks_all_completed(self):
        """并行无依赖 t1, t2 同层执行，全部完成"""
        executor = DagExecutor()
        state = executor.execute_sync(make_parallel_tasks())
        assert state.is_complete is True
        assert state.has_failure is False
        assert set(state.execution_order) == {"t1", "t2"}

    def test_diamond_dependency_all_completed(self):
        """菱形依赖 t1→(t2||t3)→t4 全部完成"""
        executor = DagExecutor()
        state = executor.execute_sync(make_diamond_tasks())
        assert state.is_complete is True
        assert state.has_failure is False
        assert state.execution_order == ["t1", "t2", "t3", "t4"]

    def test_multi_layer_all_completed(self):
        """多层依赖 t1→t2→t3→t4→t5 全部完成"""
        executor = DagExecutor()
        state = executor.execute_sync(make_multi_layer_tasks())
        assert state.is_complete is True
        assert state.has_failure is False
        assert state.execution_order == ["t1", "t2", "t3", "t4", "t5"]

    def test_execution_order_respects_dependencies(self):
        """执行顺序满足依赖约束：依赖项先于被依赖项执行"""
        executor = DagExecutor()
        state = executor.execute_sync(make_diamond_tasks())
        order = state.execution_order
        # t1 在 t2/t3 之前
        assert order.index("t1") < order.index("t2")
        assert order.index("t1") < order.index("t3")
        # t2/t3 在 t4 之前
        assert order.index("t2") < order.index("t4")
        assert order.index("t3") < order.index("t4")

    def test_execute_async_entrypoint(self):
        """execute 异步入口正常工作"""
        executor = DagExecutor()
        state = asyncio.run(executor.execute(make_linear_tasks()))
        assert state.is_complete is True
        assert state.execution_order == ["t1", "t2", "t3"]


# ============================================================
# 失败传播测试
# ============================================================


class TestFailurePropagation:
    """任务失败与传播取消测试"""

    def test_task_failure_cancels_dependents(self):
        """t2 失败 → t3 被取消"""
        executor = DagExecutor()
        tasks = make_failure_tasks()
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t2")
        )
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t1"].status == TaskStatus.COMPLETED
        assert task_map["t2"].status == TaskStatus.FAILED
        assert task_map["t2"].failure_reason == "模拟执行失败"
        assert task_map["t3"].status == TaskStatus.CANCELLED

    def test_failure_returns_partial_results(self):
        """失败时返回部分结果：is_complete=False, has_failure=True"""
        executor = DagExecutor()
        tasks = make_failure_tasks()
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t2")
        )
        assert state.is_complete is False
        assert state.has_failure is True
        # t3 未被执行，不在 execution_order 中
        assert "t1" in state.execution_order
        assert "t2" in state.execution_order
        assert "t3" not in state.execution_order

    def test_failure_in_diamond_cancels_only_affected_branch(self):
        """菱形中 t2 失败 → t4 被取消，t1/t3 仍完成"""
        executor = DagExecutor()
        tasks = make_diamond_tasks()
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t2")
        )
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t1"].status == TaskStatus.COMPLETED
        assert task_map["t2"].status == TaskStatus.FAILED
        assert task_map["t3"].status == TaskStatus.COMPLETED
        # t4 依赖 t2，应被取消
        assert task_map["t4"].status == TaskStatus.CANCELLED
        assert state.has_failure is True

    def test_cancelled_task_has_failure_reason(self):
        """被取消任务记录取消原因"""
        executor = DagExecutor()
        tasks = make_failure_tasks()
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t2")
        )
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t3"].failure_reason is not None
        assert "t2" in task_map["t3"].failure_reason

    def test_first_task_failure_cancels_all(self):
        """首个任务失败 → 全部后继被取消"""
        executor = DagExecutor()
        tasks = make_linear_tasks()
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t1")
        )
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t1"].status == TaskStatus.FAILED
        assert task_map["t2"].status == TaskStatus.CANCELLED
        assert task_map["t3"].status == TaskStatus.CANCELLED
        assert state.has_failure is True
        assert state.is_complete is False


# ============================================================
# 循环依赖测试
# ============================================================


class TestCycleDetection:
    """循环依赖检测测试"""

    def test_cycle_raises_value_error_sync(self):
        """execute_sync 检测到循环依赖抛出 ValueError"""
        executor = DagExecutor()
        with pytest.raises(ValueError, match="循环依赖"):
            executor.execute_sync(make_cycle_tasks())

    def test_cycle_raises_value_error_async(self):
        """execute 异步入口检测到循环依赖抛出 ValueError"""
        executor = DagExecutor()
        with pytest.raises(ValueError, match="循环依赖"):
            asyncio.run(executor.execute(make_cycle_tasks()))


# ============================================================
# 澄清阻塞测试
# ============================================================


class TestClarificationBlocking:
    """需要用户澄清的任务阻塞测试"""

    def test_clarification_blocks_task(self):
        """requires_clarification 的任务标记为 BLOCKED"""
        executor = DagExecutor()
        tasks = [
            DagTask(task_id="t1", description="前置任务"),
            DagTask(
                task_id="t2",
                description="需要澄清的任务",
                dependencies=["t1"],
                requires_clarification=True,
            ),
        ]
        state = executor.execute_sync(tasks, executor_func=make_blocking_executor())
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t1"].status == TaskStatus.COMPLETED
        assert task_map["t2"].status == TaskStatus.BLOCKED
        assert task_map["t2"].failure_reason == "需要用户澄清输入"

    def test_blocked_task_not_complete(self):
        """存在 BLOCKED 任务时 is_complete=False"""
        executor = DagExecutor()
        tasks = [
            DagTask(
                task_id="t1",
                description="需澄清",
                requires_clarification=True,
            ),
        ]
        state = executor.execute_sync(tasks, executor_func=make_blocking_executor())
        assert state.is_complete is False
        # BLOCKED 不计入 has_failure
        assert state.has_failure is False

    def test_normal_task_ignored_by_blocking_executor(self):
        """普通任务在阻塞执行器下正常完成"""
        executor = DagExecutor()
        tasks = [DagTask(task_id="t1", description="普通任务")]
        state = executor.execute_sync(tasks, executor_func=make_blocking_executor())
        assert state.tasks[0].status == TaskStatus.COMPLETED
        assert state.is_complete is True


# ============================================================
# 自定义 executor_func 测试
# ============================================================


class TestCustomExecutor:
    """自定义执行函数测试"""

    def test_custom_executor_invoked(self):
        """自定义执行器被调用并写入结果"""
        executor = DagExecutor()
        tasks = [DagTask(task_id="t1", description="t")]
        state = executor.execute_sync(tasks, executor_func=completing_executor)
        assert state.tasks[0].status == TaskStatus.COMPLETED
        assert state.tasks[0].result == {"executed": "t1"}

    def test_custom_executor_full_dag(self):
        """自定义执行器在完整 DAG 上工作"""
        executor = DagExecutor()
        state = executor.execute_sync(
            make_diamond_tasks(), executor_func=completing_executor
        )
        assert state.is_complete is True
        for task in state.tasks:
            assert task.result == {"executed": task.task_id}

    def test_custom_executor_simulates_failure(self):
        """自定义执行器模拟指定任务失败"""
        executor = DagExecutor()
        tasks = make_linear_tasks()
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t2")
        )
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t2"].status == TaskStatus.FAILED
        assert task_map["t3"].status == TaskStatus.CANCELLED


# ============================================================
# ParallelScheduler 测试
# ============================================================


class TestParallelScheduler:
    """并行调度器测试"""

    def test_execute_parallel_empty(self):
        """空任务列表返回空结果"""
        scheduler = ParallelScheduler()
        result = asyncio.run(scheduler.execute_parallel([], completing_executor))
        assert result == []

    def test_execute_parallel_concurrent(self):
        """多个任务并发执行全部完成"""
        scheduler = ParallelScheduler()
        tasks = make_parallel_tasks()
        result = asyncio.run(scheduler.execute_parallel(tasks, completing_executor))
        assert len(result) == 2
        for task in result:
            assert task.status == TaskStatus.COMPLETED

    def test_execute_parallel_marks_running(self):
        """执行前任务被标记为 RUNNING 并记录开始时间"""
        scheduler = ParallelScheduler()
        tasks = [DagTask(task_id="t1", description="t")]

        async def _capture(task: DagTask) -> DagTask:
            # 执行函数内可见任务已进入 RUNNING
            assert task.status == TaskStatus.RUNNING
            assert task.started_at is not None
            task.status = TaskStatus.COMPLETED
            return task

        result = asyncio.run(scheduler.execute_parallel(tasks, _capture))
        assert result[0].status == TaskStatus.COMPLETED

    def test_execute_parallel_exception_isolation(self):
        """单任务异常不影响同层其他任务，异常转化为 FAILED"""

        async def _mixed(task: DagTask) -> DagTask:
            if task.task_id == "t1":
                raise ValueError("t1 出错")
            task.status = TaskStatus.COMPLETED
            return task

        scheduler = ParallelScheduler()
        tasks = make_parallel_tasks()
        result = asyncio.run(scheduler.execute_parallel(tasks, _mixed))
        result_map = {t.task_id: t for t in result}
        assert result_map["t1"].status == TaskStatus.FAILED
        assert "ValueError" in result_map["t1"].failure_reason
        assert result_map["t2"].status == TaskStatus.COMPLETED

    def test_execute_parallel_preserves_order(self):
        """返回结果顺序与输入一致"""
        scheduler = ParallelScheduler()
        tasks = [
            DagTask(task_id="t3", description=""),
            DagTask(task_id="t1", description=""),
            DagTask(task_id="t2", description=""),
        ]
        result = asyncio.run(scheduler.execute_parallel(tasks, completing_executor))
        assert [t.task_id for t in result] == ["t3", "t1", "t2"]


class TestExecuteWithTimeout:
    """单任务超时控制测试"""

    def test_completes_within_timeout(self):
        """未超时任务正常完成"""
        scheduler = ParallelScheduler()
        task = DagTask(task_id="t1", description="快速任务")
        result = asyncio.run(
            scheduler.execute_with_timeout(task, completing_executor, timeout_ms=1000)
        )
        assert result.status == TaskStatus.COMPLETED
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_timeout_marks_failed(self):
        """超时任务标记为 FAILED 并记录原因"""
        scheduler = ParallelScheduler()
        task = DagTask(task_id="t1", description="慢任务")
        result = asyncio.run(
            scheduler.execute_with_timeout(task, slow_executor, timeout_ms=50)
        )
        assert result.status == TaskStatus.FAILED
        assert "超时" in result.failure_reason

    def test_exception_marks_failed(self):
        """执行函数抛出异常时标记为 FAILED"""
        scheduler = ParallelScheduler()
        task = DagTask(task_id="t1", description="异常任务")
        result = asyncio.run(
            scheduler.execute_with_timeout(task, raising_executor, timeout_ms=1000)
        )
        assert result.status == TaskStatus.FAILED
        assert "RuntimeError" in result.failure_reason

    def test_timeout_sets_running_before_execute(self):
        """执行前任务被标记为 RUNNING"""

        async def _check(task: DagTask) -> DagTask:
            assert task.status == TaskStatus.RUNNING
            task.status = TaskStatus.COMPLETED
            return task

        scheduler = ParallelScheduler()
        task = DagTask(task_id="t1", description="t")
        result = asyncio.run(
            scheduler.execute_with_timeout(task, _check, timeout_ms=1000)
        )
        assert result.status == TaskStatus.COMPLETED

    def test_zero_timeout_fails(self):
        """零超时阈值导致任务失败"""
        scheduler = ParallelScheduler()
        task = DagTask(task_id="t1", description="t")
        result = asyncio.run(
            scheduler.execute_with_timeout(task, slow_executor, timeout_ms=0)
        )
        assert result.status == TaskStatus.FAILED


# ============================================================
# 集成测试
# ============================================================


class TestDagExecutorIntegration:
    """DagExecutor 与各组件集成测试"""

    def test_state_serializable_after_execution(self):
        """执行后产生的 DagState 可序列化往返"""
        executor = DagExecutor()
        state = executor.execute_sync(make_diamond_tasks())
        restored = DagState.from_dict(state.to_dict())
        assert restored.is_complete == state.is_complete
        assert restored.has_failure == state.has_failure
        assert restored.execution_order == state.execution_order
        assert len(restored.tasks) == len(state.tasks)

    def test_mixed_failure_and_success(self):
        """复杂场景：部分成功部分失败，状态正确汇总"""
        executor = DagExecutor()
        # t1 → t2(失败) → t3(取消)；t4 独立完成
        tasks = [
            DagTask(task_id="t1", description="前置"),
            DagTask(task_id="t2", description="失败", dependencies=["t1"]),
            DagTask(task_id="t3", description="取消", dependencies=["t2"]),
            DagTask(task_id="t4", description="独立完成"),
        ]
        state = executor.execute_sync(
            tasks, executor_func=make_failing_executor("t2")
        )
        task_map = {t.task_id: t for t in state.tasks}
        assert task_map["t1"].status == TaskStatus.COMPLETED
        assert task_map["t2"].status == TaskStatus.FAILED
        assert task_map["t3"].status == TaskStatus.CANCELLED
        assert task_map["t4"].status == TaskStatus.COMPLETED
        assert state.has_failure is True
        assert state.is_complete is False
        # t4 与 t1 被执行，t3 被取消未执行
        assert "t3" not in state.execution_order

    def test_executor_reusable_across_calls(self):
        """同一 DagExecutor 实例可重复使用"""
        executor = DagExecutor()
        state1 = executor.execute_sync(make_linear_tasks())
        state2 = executor.execute_sync(make_diamond_tasks())
        assert state1.is_complete is True
        assert state2.is_complete is True
        assert state1.execution_order == ["t1", "t2", "t3"]
        assert state2.execution_order == ["t1", "t2", "t3", "t4"]
