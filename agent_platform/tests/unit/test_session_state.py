"""
会话状态管理单元测试 — M2.5 记忆/会话状态模块

测试用例覆盖:
  - 创建会话并检索
  - 获取不存在的会话
  - 更新会话状态
  - 添加对话轮次 / recent_queries / mentioned_entities
  - 检查点保存与恢复
  - 幂等键缓存
  - TTL 过期（mock 手动触发）
  - 序列化/反序列化
  - 最大轮次限制

所有测试使用 mock 模式，无需真实 Redis。
"""

import json
import time

import pytest

from agent_platform.memory.session_state import (
    RedisSessionManager,
    SessionCheckpoint,
    SessionState,
    SessionTurn,
)


@pytest.fixture
def manager():
    """内存模式会话管理器（max_turns=3 便于测试裁剪）"""
    return RedisSessionManager(mock=True, ttl_seconds=3600, max_turns=3)


# ============================================================
# 会话创建与检索
# ============================================================


class TestSessionCreate:
    """会话创建与检索测试"""

    def test_create_session_returns_id(self, manager):
        """创建会话返回非空 session_id"""
        session_id = manager.create_session()
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_create_and_retrieve(self, manager):
        """创建会话后可检索，初始状态正确"""
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.current_state == "RECEIVED"
        assert session.turns == []
        assert session.created_at > 0
        assert session.updated_at > 0

    def test_get_nonexistent_session(self, manager):
        """获取不存在的会话返回 None"""
        result = manager.get_session("nonexistent-id-12345")
        assert result is None

    def test_create_multiple_sessions(self, manager):
        """创建多个会话，ID 互不相同"""
        id1 = manager.create_session()
        id2 = manager.create_session()
        assert id1 != id2
        assert manager.get_session(id1) is not None
        assert manager.get_session(id2) is not None


# ============================================================
# 状态更新
# ============================================================


class TestStateUpdate:
    """状态更新测试"""

    def test_update_state(self, manager):
        """更新状态机状态"""
        session_id = manager.create_session()
        manager.update_state(session_id, "NORMALIZED")

        session = manager.get_session(session_id)
        assert session is not None
        assert session.current_state == "NORMALIZED"

    def test_update_state_multiple_times(self, manager):
        """多次更新状态"""
        session_id = manager.create_session()
        for state in ["NORMALIZED", "CONTEXT_RESOLVED", "ANALYZED", "ROUTED"]:
            manager.update_state(session_id, state)

        session = manager.get_session(session_id)
        assert session.current_state == "ROUTED"

    def test_update_state_nonexistent_no_error(self, manager):
        """更新不存在的会话不抛异常"""
        # 不应抛出异常
        manager.update_state("nonexistent", "NORMALIZED")

    def test_updated_at_changes(self, manager):
        """更新状态后 updated_at 变化"""
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        old_updated = session.updated_at

        time.sleep(0.01)
        manager.update_state(session_id, "NORMALIZED")

        session = manager.get_session(session_id)
        assert session.updated_at > old_updated


# ============================================================
# 对话轮次管理
# ============================================================


class TestTurnManagement:
    """对话轮次管理测试"""

    def test_add_turn_basic(self, manager):
        """添加对话轮次，返回 SessionTurn"""
        session_id = manager.create_session()
        turn = manager.add_turn(
            session_id,
            "什么是 GDPR？",
            "GDPR 是通用数据保护条例",
            metadata={"intent": "factual", "complexity": "L1"},
        )

        assert isinstance(turn, SessionTurn)
        assert turn.turn_id  # 非空
        assert turn.query == "什么是 GDPR？"
        assert turn.answer == "GDPR 是通用数据保护条例"
        assert turn.intent == "factual"
        assert turn.complexity == "L1"
        assert turn.timestamp > 0

    def test_add_turn_persisted(self, manager):
        """添加的轮次持久化到存储"""
        session_id = manager.create_session()
        manager.add_turn(session_id, "问题1", "答案1")

        session = manager.get_session(session_id)
        assert len(session.turns) == 1
        assert session.turns[0].query == "问题1"
        assert session.turns[0].answer == "答案1"

    def test_add_turn_without_metadata(self, manager):
        """不传 metadata 时 intent/complexity 为空字符串"""
        session_id = manager.create_session()
        turn = manager.add_turn(session_id, "问题", "答案")

        assert turn.intent == ""
        assert turn.complexity == ""

    def test_add_turn_nonexistent_raises(self, manager):
        """向不存在的会话添加轮次抛出 ValueError"""
        with pytest.raises(ValueError, match="会话不存在"):
            manager.add_turn("nonexistent", "query", "answer")

    def test_recent_queries(self, manager):
        """recent_queries 返回最近 n 轮查询（旧 -> 新）"""
        session_id = manager.create_session()
        manager.add_turn(session_id, "问题1", "答案1")
        manager.add_turn(session_id, "问题2", "答案2")
        manager.add_turn(session_id, "问题3", "答案3")

        session = manager.get_session(session_id)
        recent = session.recent_queries(2)
        assert recent == ["问题2", "问题3"]

    def test_recent_queries_all(self, manager):
        """请求超过已有数量时返回全部"""
        session_id = manager.create_session()
        manager.add_turn(session_id, "问题1", "答案1")

        session = manager.get_session(session_id)
        recent = session.recent_queries(5)
        assert recent == ["问题1"]

    def test_recent_queries_zero(self, manager):
        """n=0 返回空列表"""
        session_id = manager.create_session()
        manager.add_turn(session_id, "问题1", "答案1")

        session = manager.get_session(session_id)
        assert session.recent_queries(0) == []

    def test_recent_queries_empty_session(self, manager):
        """无对话历史时返回空列表"""
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        assert session.recent_queries(3) == []

    def test_mentioned_entities(self, manager):
        """mentioned_entities 提取历史轮次中的实体"""
        session_id = manager.create_session()
        manager.add_turn(
            session_id, "问题1", "答案1",
            metadata={"entities": [{"name": "GDPR", "type": "regulation"}]},
        )
        manager.add_turn(
            session_id, "问题2", "答案2",
            metadata={"entities": [{"name": "CCPA", "type": "regulation"}]},
        )

        session = manager.get_session(session_id)
        entities = session.mentioned_entities()
        assert len(entities) == 2
        assert entities[0]["name"] == "GDPR"
        assert entities[1]["name"] == "CCPA"

    def test_mentioned_entities_empty(self, manager):
        """无实体时返回空列表"""
        session_id = manager.create_session()
        manager.add_turn(session_id, "问题", "答案")
        session = manager.get_session(session_id)
        assert session.mentioned_entities() == []

    def test_max_turns_limit(self, manager):
        """最大轮次限制：超过 max_turns=3 时裁剪最早轮次"""
        session_id = manager.create_session()
        for i in range(5):
            manager.add_turn(session_id, f"问题{i}", f"答案{i}")

        session = manager.get_session(session_id)
        assert len(session.turns) == 3
        # 保留最后 3 轮（问题2、问题3、问题4）
        assert session.turns[0].query == "问题2"
        assert session.turns[1].query == "问题3"
        assert session.turns[2].query == "问题4"

    def test_max_turns_boundary(self, manager):
        """恰好等于 max_turns 时不裁剪"""
        session_id = manager.create_session()
        for i in range(3):
            manager.add_turn(session_id, f"问题{i}", f"答案{i}")

        session = manager.get_session(session_id)
        assert len(session.turns) == 3
        assert session.turns[0].query == "问题0"


# ============================================================
# 检查点管理
# ============================================================


class TestCheckpoint:
    """检查点保存与恢复测试"""

    def test_save_and_restore_checkpoint(self, manager):
        """保存检查点后可恢复"""
        session_id = manager.create_session()
        checkpoint = SessionCheckpoint(
            checkpoint_id="cp-001",
            session_id=session_id,
            state_machine_state="RETRIEVING",
            events=[
                {
                    "event_id": "e1",
                    "from_state": "ROUTED",
                    "to_state": "RETRIEVING",
                    "timestamp": 1234567890.0,
                    "metadata": {"step": "retrieve"},
                }
            ],
            timestamp=time.time(),
            metadata={"round": 1},
        )
        manager.save_checkpoint(session_id, checkpoint)

        restored = manager.restore_checkpoint(session_id)
        assert restored is not None
        assert restored.checkpoint_id == "cp-001"
        assert restored.session_id == session_id
        assert restored.state_machine_state == "RETRIEVING"
        assert len(restored.events) == 1
        assert restored.events[0]["to_state"] == "RETRIEVING"
        assert restored.metadata == {"round": 1}

    def test_restore_nonexistent_checkpoint(self, manager):
        """恢复不存在的检查点返回 None"""
        result = manager.restore_checkpoint("nonexistent")
        assert result is None

    def test_save_checkpoint_overwrites(self, manager):
        """多次保存检查点，后者覆盖前者"""
        session_id = manager.create_session()

        cp1 = SessionCheckpoint(
            checkpoint_id="cp-001",
            session_id=session_id,
            state_machine_state="RETRIEVING",
        )
        manager.save_checkpoint(session_id, cp1)

        cp2 = SessionCheckpoint(
            checkpoint_id="cp-002",
            session_id=session_id,
            state_machine_state="GENERATING",
        )
        manager.save_checkpoint(session_id, cp2)

        restored = manager.restore_checkpoint(session_id)
        assert restored.checkpoint_id == "cp-002"
        assert restored.state_machine_state == "GENERATING"


# ============================================================
# 幂等键缓存
# ============================================================


class TestIdempotency:
    """幂等键缓存测试"""

    def test_first_check_returns_none(self, manager):
        """首次检查幂等键返回 None"""
        result = manager.check_idempotency("req-001")
        assert result is None

    def test_cache_and_check(self, manager):
        """缓存响应后检查返回缓存数据"""
        response = {"answer": "测试答案", "status": "ok"}
        manager.cache_response("req-001", response)

        result = manager.check_idempotency("req-001")
        assert result is not None
        assert result["response"] == response
        assert "cached_at" in result
        assert "ttl" in result

    def test_different_keys_independent(self, manager):
        """不同幂等键互相独立"""
        manager.cache_response("req-001", {"answer": "A"})
        manager.cache_response("req-002", {"answer": "B"})

        assert manager.check_idempotency("req-001")["response"]["answer"] == "A"
        assert manager.check_idempotency("req-002")["response"]["answer"] == "B"

    def test_idempotency_flow(self, manager):
        """完整幂等流程：首次无缓存 -> 缓存 -> 二次命中"""
        key = "req-flow-001"

        # 首次检查：无缓存
        assert manager.check_idempotency(key) is None

        # 执行业务逻辑后缓存响应
        manager.cache_response(key, {"answer": "结果", "evidence_count": 3})

        # 二次检查：命中缓存
        cached = manager.check_idempotency(key)
        assert cached is not None
        assert cached["response"]["answer"] == "结果"


# ============================================================
# TTL 过期与删除
# ============================================================


class TestTTLExpiration:
    """TTL 过期与删除测试"""

    def test_expire_session_sets_ttl(self, manager):
        """expire_session 正确设置 TTL"""
        session_id = manager.create_session()
        manager.expire_session(session_id, ttl=100)

        key = manager._session_key(session_id)
        ttl = manager.client.ttl(key)
        assert 0 < ttl <= 100

    def test_expired_session_returns_none(self, manager):
        """过期会话返回 None（手动设置过期时间）"""
        session_id = manager.create_session()
        assert manager.get_session(session_id) is not None

        # 手动将会话过期时间设为过去（模拟 TTL 到期）
        key = manager._session_key(session_id)
        manager.client._expiry[key] = time.time() - 1

        assert manager.get_session(session_id) is None

    def test_delete_session(self, manager):
        """删除会话后不可检索"""
        session_id = manager.create_session()
        manager.delete_session(session_id)
        assert manager.get_session(session_id) is None

    def test_delete_session_removes_checkpoint(self, manager):
        """删除会话同时删除关联检查点"""
        session_id = manager.create_session()
        checkpoint = SessionCheckpoint(
            checkpoint_id="cp-001",
            session_id=session_id,
            state_machine_state="RECEIVED",
        )
        manager.save_checkpoint(session_id, checkpoint)
        assert manager.restore_checkpoint(session_id) is not None

        manager.delete_session(session_id)
        assert manager.restore_checkpoint(session_id) is None

    def test_delete_nonexistent_session_no_error(self, manager):
        """删除不存在的会话不抛异常"""
        manager.delete_session("nonexistent")  # 不应抛异常

    def test_default_ttl_on_create(self, manager):
        """创建会话时设置默认 TTL"""
        session_id = manager.create_session()
        key = manager._session_key(session_id)
        ttl = manager.client.ttl(key)
        # ttl_seconds=3600（fixture 配置）
        assert 0 < ttl <= 3600


# ============================================================
# 序列化/反序列化
# ============================================================


class TestSerialization:
    """数据模型序列化/反序列化测试"""

    def test_session_state_roundtrip(self):
        """SessionState to_dict / from_dict 往返一致"""
        original = SessionState(
            session_id="test-001",
            current_state="RETRIEVING",
            turns=[
                SessionTurn(
                    turn_id="t1",
                    query="Q1",
                    answer="A1",
                    intent="factual",
                    complexity="L1",
                    timestamp=1234567890.0,
                    metadata={"key": "value"},
                ),
            ],
            created_at=1234567890.0,
            updated_at=1234567891.0,
            query_spec={"intent": "factual"},
            budget_consumed={"tokens": 500},
            metadata={"user": "test"},
        )
        d = original.to_dict()
        restored = SessionState.from_dict(d)

        assert restored.session_id == original.session_id
        assert restored.current_state == original.current_state
        assert len(restored.turns) == 1
        assert restored.turns[0].query == "Q1"
        assert restored.turns[0].intent == "factual"
        assert restored.query_spec == original.query_spec
        assert restored.budget_consumed == original.budget_consumed
        assert restored.metadata == original.metadata

    def test_session_state_roundtrip_empty_turns(self):
        """空轮次列表的 SessionState 往返"""
        original = SessionState(
            session_id="test-002",
            current_state="RECEIVED",
        )
        d = original.to_dict()
        restored = SessionState.from_dict(d)

        assert restored.session_id == "test-002"
        assert restored.turns == []
        assert restored.query_spec is None
        assert restored.budget_consumed == {}

    def test_session_turn_roundtrip(self):
        """SessionTurn to_dict / from_dict 往返一致"""
        original = SessionTurn(
            turn_id="t1",
            query="Q",
            answer="A",
            intent="procedural",
            complexity="L2",
            timestamp=1234567890.0,
            metadata={"entities": [{"name": "GDPR"}]},
        )
        d = original.to_dict()
        restored = SessionTurn.from_dict(d)

        assert restored.turn_id == original.turn_id
        assert restored.query == original.query
        assert restored.answer == original.answer
        assert restored.intent == original.intent
        assert restored.complexity == original.complexity
        assert restored.metadata == original.metadata

    def test_checkpoint_roundtrip(self):
        """SessionCheckpoint to_dict / from_dict 往返一致"""
        original = SessionCheckpoint(
            checkpoint_id="cp-1",
            session_id="sess-1",
            state_machine_state="GENERATING",
            events=[{"event_id": "e1", "to_state": "GENERATING"}],
            timestamp=1234567890.0,
            metadata={"round": 2},
        )
        d = original.to_dict()
        restored = SessionCheckpoint.from_dict(d)

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.session_id == original.session_id
        assert restored.state_machine_state == original.state_machine_state
        assert len(restored.events) == 1
        assert restored.metadata == {"round": 2}

    def test_session_state_json_serializable(self):
        """SessionState.to_dict() 可 JSON 序列化（Redis 存储要求）"""
        session = SessionState(
            session_id="test-003",
            current_state="RECEIVED",
            turns=[
                SessionTurn(
                    turn_id="t1",
                    query="Q",
                    answer="A",
                    intent="",
                    complexity="",
                    timestamp=1234567890.0,
                ),
            ],
        )
        d = session.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        restored = json.loads(json_str)
        assert restored["session_id"] == "test-003"
        assert len(restored["turns"]) == 1

    def test_persistence_through_redis(self, manager):
        """会话状态通过 Redis 存储后完整恢复"""
        session_id = manager.create_session()
        manager.update_state(session_id, "RETRIEVING")
        manager.add_turn(
            session_id, "查询", "回答",
            metadata={"intent": "factual", "complexity": "L1",
                      "entities": [{"name": "GDPR"}]},
        )

        # 重新加载（模拟从 Redis 恢复）
        session = manager.get_session(session_id)
        assert session is not None
        assert session.current_state == "RETRIEVING"
        assert len(session.turns) == 1
        assert session.turns[0].intent == "factual"
        assert session.recent_queries(1) == ["查询"]
        assert len(session.mentioned_entities()) == 1


# ============================================================
# 管理器初始化
# ============================================================


class TestManagerInit:
    """管理器初始化测试"""

    def test_mock_mode_flag(self):
        """mock 模式下 is_mock 为 True"""
        mgr = RedisSessionManager(mock=True)
        assert mgr.is_mock is True

    def test_mock_client_is_mock_redis(self):
        """mock 模式下 client 为 _MockRedis 实例"""
        from agent_platform.memory.session_state.session_manager import _MockRedis

        mgr = RedisSessionManager(mock=True)
        assert isinstance(mgr.client, _MockRedis)

    def test_redis_unavailable_falls_back_to_mock(self):
        """Redis 不可用时自动降级为 mock 模式"""
        # 指定一个不可达的 Redis URL，应自动降级
        mgr = RedisSessionManager(
            redis_url="redis://localhost:19999/0",  # 不存在的端口
            mock=False,
        )
        assert mgr.is_mock is True

    def test_default_max_turns(self):
        """默认 max_turns 从环境变量或默认值读取"""
        import os

        old_val = os.environ.get("SESSION_MAX_TURNS")
        try:
            os.environ["SESSION_MAX_TURNS"] = "15"
            mgr = RedisSessionManager(mock=True)
            assert mgr._max_turns == 15
        finally:
            if old_val is not None:
                os.environ["SESSION_MAX_TURNS"] = old_val
            else:
                os.environ.pop("SESSION_MAX_TURNS", None)
