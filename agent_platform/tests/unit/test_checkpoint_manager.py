"""
检查点管理器单元测试（M4.4）

测试用例（对应开发计划测试用例表）:
  - 保存检查点: save() → 返回 checkpoint_id
  - 恢复检查点: load_latest() → 状态正确恢复
  - 多版本: 保存5次 → 只保留最近3个
  - 恢复后重放: recover() → 返回已完成和待执行任务
  - 无检查点: 新会话恢复 → success=False
  - 检查点损坏: recover_from_corrupt → 降级到上一版本
  - 删除全部: delete_all() → 无检查点
  - 序列化往返: to_dict/from_dict → 数据一致
  - 统计信息: get_stats() → 返回正确统计

额外测试用例:
  - 版本号递增
  - cleanup_old_versions 清理
  - load_by_version 指定版本加载
  - get_resume_tasks 已完成/待执行区分
  - Redis 后端保存加载
  - Redis 异常降级到内存
"""

import pytest

from agent_platform.orchestration.checkpoint_manager import (
    Checkpoint,
    CheckpointManager,
    RecoveryManager,
    RecoveryResult,
)


# ============================================================
# 工厂函数与测试桩
# ============================================================


def make_checkpoint(
    session_id="s1",
    request_id="r1",
    state="INIT",
    query_spec=None,
    query_plan=None,
    dag_state=None,
    evidence_bundle=None,
    budget_consumed=None,
    version=1,
):
    """构造检查点，缺省字段使用安全默认值"""
    return Checkpoint(
        checkpoint_id="",
        session_id=session_id,
        request_id=request_id,
        state=state,
        query_spec=query_spec if query_spec is not None else {"query": "default"},
        query_plan=query_plan if query_plan is not None else {"path": "P2"},
        dag_state=dag_state,
        evidence_bundle=evidence_bundle if evidence_bundle is not None else {},
        budget_consumed=budget_consumed if budget_consumed is not None else {"tokens": 0},
        version=version,
    )


def make_dag_state(tasks):
    """构造 DAG 任务状态快照"""
    return {"tasks": tasks}


class FakeRedis:
    """简易 Redis 客户端，用于测试 Redis 后端"""

    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def delete(self, key):
        self._data.pop(key, None)


class FailingRedis:
    """始终抛异常的 Redis 客户端，用于测试降级"""

    def get(self, key):
        raise RuntimeError("redis down")

    def set(self, key, value):
        raise RuntimeError("redis down")

    def delete(self, key):
        raise RuntimeError("redis down")


# ============================================================
# Checkpoint 数据模型测试
# ============================================================


class TestCheckpoint:
    """检查点数据模型测试"""

    def test_to_dict_from_dict_roundtrip(self):
        """序列化往返: to_dict/from_dict 数据一致"""
        cp = make_checkpoint(
            state="RETRIEVING",
            query_spec={"query": "保险条款"},
            query_plan={"path": "P3"},
            dag_state={"tasks": [{"task_id": "t1", "status": "completed"}]},
            evidence_bundle={"evidence": ["e1"]},
            budget_consumed={"tokens": 1000},
            version=2,
        )
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.checkpoint_id == cp.checkpoint_id
        assert restored.session_id == cp.session_id
        assert restored.request_id == cp.request_id
        assert restored.state == cp.state
        assert restored.query_spec == cp.query_spec
        assert restored.query_plan == cp.query_plan
        assert restored.dag_state == cp.dag_state
        assert restored.evidence_bundle == cp.evidence_bundle
        assert restored.budget_consumed == cp.budget_consumed
        assert restored.timestamp == cp.timestamp
        assert restored.version == cp.version

    def test_post_init_generates_id_and_timestamp(self):
        """__post_init__ 自动生成 checkpoint_id 和 timestamp"""
        cp = Checkpoint(
            checkpoint_id="",
            session_id="s1",
            request_id="r1",
            state="INIT",
            query_spec={},
            query_plan={},
        )
        assert cp.checkpoint_id  # 非空且以 cp- 开头
        assert cp.checkpoint_id.startswith("cp-")
        assert cp.timestamp  # 非空

    def test_post_init_preserves_given_id_and_timestamp(self):
        """已提供的 checkpoint_id 和 timestamp 不被覆盖"""
        cp = Checkpoint(
            checkpoint_id="cp-custom",
            session_id="s1",
            request_id="r1",
            state="INIT",
            query_spec={},
            query_plan={},
            timestamp="1234567890.0",
        )
        assert cp.checkpoint_id == "cp-custom"
        assert cp.timestamp == "1234567890.0"

    def test_from_dict_missing_fields_defaults(self):
        """from_dict 缺失字段使用安全默认值"""
        restored = Checkpoint.from_dict({"session_id": "s1", "request_id": "r1"})
        assert restored.query_spec == {}
        assert restored.query_plan == {}
        assert restored.dag_state is None
        assert restored.evidence_bundle == {}
        assert restored.budget_consumed == {}
        assert restored.version == 1
        # checkpoint_id 缺失 → __post_init__ 自动生成
        assert restored.checkpoint_id

    def test_to_dict_all_json_serializable(self):
        """to_dict 所有字段为 JSON 可序列化类型"""
        import json

        cp = make_checkpoint(
            state="GENERATING",
            dag_state={"tasks": [{"task_id": "t1", "status": "completed"}]},
        )
        d = cp.to_dict()
        # 能正常 JSON 序列化即说明类型安全
        json.dumps(d, ensure_ascii=False)

    def test_repr(self):
        """__repr__ 包含关键字段"""
        cp = make_checkpoint(state="RETRIEVING", version=3)
        r = repr(cp)
        assert "Checkpoint" in r
        assert "RETRIEVING" in r
        assert "version=3" in r


# ============================================================
# CheckpointManager 测试
# ============================================================


class TestCheckpointManager:
    """检查点管理器测试"""

    def test_save_returns_checkpoint_id(self):
        """保存检查点: save() 返回 checkpoint_id"""
        manager = CheckpointManager()
        cp = make_checkpoint()
        cp_id = manager.save(cp)
        assert cp_id  # 非空
        assert cp_id.startswith("cp-")
        # 版本号被分配为 1
        assert cp.version == 1

    def test_load_latest_restores_state(self):
        """恢复检查点: load_latest() 状态正确恢复"""
        manager = CheckpointManager()
        cp = make_checkpoint(
            state="RETRIEVING", query_spec={"q": "保险条款"}
        )
        manager.save(cp)
        loaded = manager.load_latest("s1", "r1")
        assert loaded is not None
        assert loaded.state == "RETRIEVING"
        assert loaded.query_spec == {"q": "保险条款"}
        assert loaded.version == 1
        assert loaded.session_id == "s1"
        assert loaded.request_id == "r1"

    def test_max_versions_retention(self):
        """多版本: 保存5次 → 只保留最近3个"""
        manager = CheckpointManager()
        for i in range(5):
            manager.save(make_checkpoint(state=f"S{i}"))
        versions = manager.list_versions("s1", "r1")
        assert len(versions) == CheckpointManager.MAX_VERSIONS
        assert versions == sorted(versions)
        # 保留版本 3, 4, 5
        assert versions == [3, 4, 5]
        # 最新版本状态正确
        loaded = manager.load_latest("s1", "r1")
        assert loaded.state == "S4"

    def test_save_increments_version(self):
        """多次保存版本号递增"""
        manager = CheckpointManager()
        for _ in range(3):
            cp = make_checkpoint()
            manager.save(cp)
        versions = manager.list_versions("s1", "r1")
        assert versions == [1, 2, 3]

    def test_list_versions_empty(self):
        """无检查点 → list_versions 返回空列表"""
        manager = CheckpointManager()
        assert manager.list_versions("s1", "r1") == []

    def test_load_latest_none_when_empty(self):
        """无检查点 → load_latest 返回 None"""
        manager = CheckpointManager()
        assert manager.load_latest("s1", "r1") is None

    def test_load_by_version(self):
        """load_by_version 加载指定版本"""
        manager = CheckpointManager()
        for i in range(3):
            manager.save(make_checkpoint(state=f"S{i}"))
        v1 = manager.load_by_version("s1", "r1", 1)
        assert v1 is not None
        assert v1.state == "S0"
        v3 = manager.load_by_version("s1", "r1", 3)
        assert v3 is not None
        assert v3.state == "S2"

    def test_load_by_version_nonexistent_returns_none(self):
        """加载不存在的版本 → 返回 None"""
        manager = CheckpointManager()
        manager.save(make_checkpoint())
        assert manager.load_by_version("s1", "r1", 99) is None

    def test_cleanup_old_versions(self):
        """cleanup_old_versions 保留最近 MAX_VERSIONS 个"""
        manager = CheckpointManager()
        for _ in range(5):
            manager.save(make_checkpoint())
        # save 已自动清理，显式调用应保持不变
        manager.cleanup_old_versions("s1", "r1")
        versions = manager.list_versions("s1", "r1")
        assert len(versions) == CheckpointManager.MAX_VERSIONS
        assert versions == [3, 4, 5]
        # 旧版本数据已删除
        assert manager.load_by_version("s1", "r1", 1) is None
        assert manager.load_by_version("s1", "r1", 2) is None

    def test_delete_all(self):
        """删除全部: delete_all() 后无检查点"""
        manager = CheckpointManager()
        for i in range(3):
            manager.save(make_checkpoint(state=f"S{i}"))
        manager.delete_all("s1", "r1")
        assert manager.list_versions("s1", "r1") == []
        assert manager.load_latest("s1", "r1") is None

    def test_get_stats_memory(self):
        """统计信息: get_stats() 返回正确统计（内存后端）"""
        manager = CheckpointManager()
        manager.save(make_checkpoint())
        stats = manager.get_stats()
        assert stats["backend"] == "memory"
        assert stats["max_versions"] == CheckpointManager.MAX_VERSIONS
        assert stats["checkpoint_count"] == 1
        assert stats["session_request_count"] == 1

    def test_get_stats_memory_multiple_checkpoints(self):
        """多次保存后统计信息正确"""
        manager = CheckpointManager()
        for _ in range(5):
            manager.save(make_checkpoint())
        stats = manager.get_stats()
        assert stats["backend"] == "memory"
        # 清理后只保留 3 个检查点
        assert stats["checkpoint_count"] == CheckpointManager.MAX_VERSIONS
        # 仅 1 个 session+request
        assert stats["session_request_count"] == 1

    def test_get_stats_memory_multiple_sessions(self):
        """多个会话的统计信息正确"""
        manager = CheckpointManager()
        manager.save(make_checkpoint(session_id="s1", request_id="r1"))
        manager.save(make_checkpoint(session_id="s2", request_id="r1"))
        stats = manager.get_stats()
        assert stats["session_request_count"] == 2

    def test_isolation_between_sessions(self):
        """不同 session/request 的检查点互相隔离"""
        manager = CheckpointManager()
        manager.save(make_checkpoint(session_id="s1", request_id="r1", state="A"))
        manager.save(make_checkpoint(session_id="s2", request_id="r1", state="B"))
        loaded1 = manager.load_latest("s1", "r1")
        loaded2 = manager.load_latest("s2", "r1")
        assert loaded1.state == "A"
        assert loaded2.state == "B"

    def test_redis_backend_save_load(self):
        """Redis 后端: 保存与加载正常工作"""
        redis_client = FakeRedis()
        manager = CheckpointManager(redis_client=redis_client)
        cp = make_checkpoint(state="RETRIEVING")
        manager.save(cp)
        loaded = manager.load_latest("s1", "r1")
        assert loaded is not None
        assert loaded.state == "RETRIEVING"
        stats = manager.get_stats()
        assert stats["backend"] == "redis"
        assert stats["max_versions"] == CheckpointManager.MAX_VERSIONS

    def test_redis_backend_max_versions(self):
        """Redis 后端版本保留策略生效"""
        redis_client = FakeRedis()
        manager = CheckpointManager(redis_client=redis_client)
        for i in range(5):
            manager.save(make_checkpoint(state=f"S{i}"))
        versions = manager.list_versions("s1", "r1")
        assert len(versions) == CheckpointManager.MAX_VERSIONS
        assert manager.load_latest("s1", "r1").state == "S4"

    def test_redis_failure_falls_back_to_memory(self):
        """Redis 异常时降级到内存存储"""
        redis_client = FailingRedis()
        manager = CheckpointManager(redis_client=redis_client)
        cp = make_checkpoint(state="RETRIEVING")
        # 首次 _store_get 抛异常 → 永久降级到内存
        manager.save(cp)
        # 降级后从内存加载
        loaded = manager.load_latest("s1", "r1")
        assert loaded is not None
        assert loaded.state == "RETRIEVING"
        stats = manager.get_stats()
        assert stats["backend"] == "memory"

    def test_repr(self):
        """__repr__ 包含后端信息"""
        mem_manager = CheckpointManager()
        assert "memory" in repr(mem_manager)
        redis_manager = CheckpointManager(redis_client=FakeRedis())
        assert "redis" in repr(redis_manager)


# ============================================================
# RecoveryManager 测试
# ============================================================


class TestRecoveryManager:
    """故障恢复管理器测试"""

    def test_recover_returns_completed_and_pending(self):
        """恢复后重放: recover() 返回已完成和待执行任务"""
        manager = CheckpointManager()
        cp = make_checkpoint(state="RETRIEVING")
        cp.dag_state = make_dag_state(
            [
                {"task_id": "t1", "status": "completed"},
                {"task_id": "t2", "status": "pending"},
                {"task_id": "t3", "status": "running"},
            ]
        )
        manager.save(cp)
        recovery = RecoveryManager(manager)
        result = recovery.recover("s1", "r1")
        assert result.success is True
        assert result.checkpoint is not None
        assert result.recovered_state == "RETRIEVING"
        assert result.completed_task_ids == ["t1"]
        assert result.pending_task_ids == ["t2", "t3"]

    def test_recover_no_checkpoint(self):
        """无检查点: 新会话恢复 → success=False"""
        manager = CheckpointManager()
        recovery = RecoveryManager(manager)
        result = recovery.recover("s_new", "r_new")
        assert result.success is False
        assert result.error is not None
        assert result.checkpoint is None
        assert result.completed_task_ids == []
        assert result.pending_task_ids == []

    def test_recover_with_no_dag_state(self):
        """无 dag_state → completed/pending 均为空"""
        manager = CheckpointManager()
        cp = make_checkpoint(state="INIT")
        cp.dag_state = None
        manager.save(cp)
        recovery = RecoveryManager(manager)
        result = recovery.recover("s1", "r1")
        assert result.success is True
        assert result.completed_task_ids == []
        assert result.pending_task_ids == []

    def test_recover_from_corrupt_falls_back(self):
        """检查点损坏: recover_from_corrupt 降级到上一版本"""
        manager = CheckpointManager()
        # 保存两个版本
        cp1 = make_checkpoint(state="S0")
        manager.save(cp1)
        cp2 = make_checkpoint(state="S1")
        manager.save(cp2)
        # 破坏最新版本（版本 2）的存储数据
        corrupt_key = manager._make_key("s1", "r1", 2)
        manager._store_set(corrupt_key, "{not valid json")
        recovery = RecoveryManager(manager)
        result = recovery.recover_from_corrupt("s1", "r1")
        assert result.success is True
        assert result.checkpoint is not None
        assert result.checkpoint.state == "S0"  # 回退到版本 1
        assert result.error is not None
        assert "回退" in result.error
        assert "1" in result.error

    def test_recover_from_corrupt_all_corrupt(self):
        """所有版本均损坏 → success=False"""
        manager = CheckpointManager()
        manager.save(make_checkpoint(state="S0"))
        manager.save(make_checkpoint(state="S1"))
        # 破坏所有版本
        for v in [1, 2]:
            manager._store_set(
                manager._make_key("s1", "r1", v), "{bad json"
            )
        recovery = RecoveryManager(manager)
        result = recovery.recover_from_corrupt("s1", "r1")
        assert result.success is False
        assert result.checkpoint is None
        assert result.error is not None

    def test_recover_from_corrupt_no_versions(self):
        """无可用版本 → success=False"""
        manager = CheckpointManager()
        recovery = RecoveryManager(manager)
        result = recovery.recover_from_corrupt("s1", "r1")
        assert result.success is False
        assert result.error is not None

    def test_recover_from_corrupt_latest_valid(self):
        """最新版本未损坏 → 直接使用最新版本（无回退）"""
        manager = CheckpointManager()
        manager.save(make_checkpoint(state="S0"))
        manager.save(make_checkpoint(state="S1"))
        recovery = RecoveryManager(manager)
        result = recovery.recover_from_corrupt("s1", "r1")
        assert result.success is True
        assert result.checkpoint.state == "S1"
        assert result.error is not None
        assert "回退" in result.error

    def test_get_resume_tasks_completed_and_pending(self):
        """get_resume_tasks 区分已完成与待执行"""
        recovery = RecoveryManager()
        dag_state = make_dag_state(
            [
                {"task_id": "t1", "status": "completed"},
                {"task_id": "t2", "status": "done"},
                {"task_id": "t3", "status": "success"},
                {"task_id": "t4", "status": "pending"},
                {"task_id": "t5", "status": "running"},
                {"task_id": "t6", "status": "failed"},
            ]
        )
        completed, pending = recovery.get_resume_tasks(dag_state)
        assert completed == ["t1", "t2", "t3"]
        assert pending == ["t4", "t5", "t6"]

    def test_get_resume_tasks_empty(self):
        """空 dag_state → 返回空列表"""
        recovery = RecoveryManager()
        completed, pending = recovery.get_resume_tasks({})
        assert completed == []
        assert pending == []

    def test_get_resume_tasks_none(self):
        """dag_state 为 None → 返回空列表"""
        recovery = RecoveryManager()
        completed, pending = recovery.get_resume_tasks(None)
        assert completed == []
        assert pending == []

    def test_get_resume_tasks_id_field_alias(self):
        """task_id 与 id 字段均支持"""
        recovery = RecoveryManager()
        dag_state = make_dag_state(
            [
                {"id": "t1", "status": "completed"},
                {"task_id": "t2", "status": "pending"},
            ]
        )
        completed, pending = recovery.get_resume_tasks(dag_state)
        assert completed == ["t1"]
        assert pending == ["t2"]

    def test_get_resume_tasks_status_case_insensitive(self):
        """任务状态大小写不敏感"""
        recovery = RecoveryManager()
        dag_state = make_dag_state(
            [
                {"task_id": "t1", "status": "COMPLETED"},
                {"task_id": "t2", "status": "Done"},
                {"task_id": "t3", "status": "Pending"},
            ]
        )
        completed, pending = recovery.get_resume_tasks(dag_state)
        assert completed == ["t1", "t2"]
        assert pending == ["t3"]

    def test_get_resume_tasks_skips_invalid_entries(self):
        """跳过无 task_id 或非字典的任务条目"""
        recovery = RecoveryManager()
        dag_state = {
            "tasks": [
                {"status": "completed"},  # 无 task_id
                {"task_id": "t2", "status": "pending"},
                "not a dict",
            ]
        }
        completed, pending = recovery.get_resume_tasks(dag_state)
        assert completed == []
        assert pending == ["t2"]

    def test_recovery_result_defaults(self):
        """RecoveryResult 默认值"""
        r = RecoveryResult(success=False)
        assert r.checkpoint is None
        assert r.recovered_state == ""
        assert r.completed_task_ids == []
        assert r.pending_task_ids == []
        assert r.error is None

    def test_recovery_manager_default_checkpoint_manager(self):
        """RecoveryManager 无参时创建默认 CheckpointManager"""
        recovery = RecoveryManager()
        assert recovery._checkpoint_manager is not None
        assert isinstance(recovery._checkpoint_manager, CheckpointManager)
