"""
Redis 会话状态管理器 — M2.5 记忆/会话状态模块

基于 Redis 的会话状态持久化管理，替代 Phase 1 的纯内存实现。
支持 Redis 主从 + Sentinel 部署，开发环境可降级为内存存储。

职责:
  - 创建/获取/删除会话
  - 更新会话状态机状态
  - 保存/恢复状态机检查点
  - 管理会话 TTL（过期自动清理）
  - 对话轮次管理（含最大轮次限制）
  - 幂等键缓存

Redis Key 设计（参考 runtime/redis/keyspace.md）:
  - ace-rag:session:{session_id}     → Hash: 会话状态
  - ace-rag:checkpoint:{session_id}  → String(JSON): 状态机检查点
  - ace-rag:idempotency:{key}        → String(JSON): 幂等响应缓存

部署说明:
  生产环境使用 Redis 主从 + Sentinel，通过 redis_url 指向 Sentinel 发现的 Master。
  Docker Compose 开发环境使用单节点 Redis（见 docker-compose.yml）。
  当 Redis 不可用或 mock=True 时，自动降级为内存存储（_MockRedis）。
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from .session_models import SessionCheckpoint, SessionState, SessionTurn

logger = logging.getLogger(__name__)

# ============================================================
# Redis Key 前缀（与 runtime/redis/keyspace.md 一致）
# ============================================================
_SESSION_PREFIX = "ace-rag:session:"
_CHECKPOINT_PREFIX = "ace-rag:checkpoint:"
_IDEMPOTENCY_PREFIX = "ace-rag:idempotency:"

# 默认配置（构造时从环境变量读取，允许运行时动态配置）
_DEFAULT_TTL = 3600
_DEFAULT_MAX_TURNS = 20
_IDEMPOTENCY_TTL = 600  # 幂等键缓存 10 分钟（参考 ttl-policy.md）


class _MockRedis:
    """
    模拟 Redis 客户端 — 用于开发/测试环境

    使用内存字典模拟 Redis 的 Hash、String、TTL 等操作。
    实现 redis-py 常用命令的子集，方法签名与 redis-py 保持一致，
    使 RedisSessionManager 可以透明切换真实 Redis 和内存存储。

    注意:
      - 非线程安全，仅用于单线程开发/测试
      - TTL 过期为惰性删除（访问时检查），与 Redis 的惰性删除行为一致
    """

    def __init__(self):
        # _data: key -> value（string 类型为 str，hash 类型为 Dict[str, str]）
        self._data: Dict[str, Any] = {}
        # _expiry: key -> 过期时间戳（None 表示永不过期）
        self._expiry: Dict[str, Optional[float]] = {}
        # _type: key -> "string" | "hash"
        self._type: Dict[str, str] = {}

    def _check_expired(self, key: str) -> bool:
        """检查 key 是否过期，过期则删除并返回 True"""
        exp = self._expiry.get(key)
        if exp is not None and time.time() >= exp:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            self._type.pop(key, None)
            return True
        return False

    def ping(self) -> bool:
        """模拟 PING 命令"""
        return True

    # ----------------------------------------------------------
    # String 操作
    # ----------------------------------------------------------

    def set(self, name: str, value: str, ex: Optional[int] = None) -> bool:
        """
        SET 命令

        Args:
            name: key
            value: 值（字符串）
            ex: 过期时间（秒），None 表示永不过期
        """
        self._data[name] = value
        self._type[name] = "string"
        if ex is not None:
            self._expiry[name] = time.time() + ex
        else:
            self._expiry[name] = None
        return True

    def get(self, name: str) -> Optional[str]:
        """GET 命令"""
        if self._check_expired(name):
            return None
        if self._type.get(name) != "string":
            return None
        return self._data.get(name)

    # ----------------------------------------------------------
    # Hash 操作
    # ----------------------------------------------------------

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        HSET 命令

        支持单字段（key/value）和批量（mapping）两种模式。
        返回新增字段数（与 redis-py 行为一致）。
        """
        if self._check_expired(name) or name not in self._data:
            self._data[name] = {}
            self._type[name] = "hash"

        if self._type[name] != "hash":
            # 类型不匹配（key 原为 string），与 Redis 行为一致：拒绝操作
            return 0

        added = 0
        if mapping:
            for k, v in mapping.items():
                if k not in self._data[name]:
                    added += 1
                self._data[name][k] = v
        if key is not None and value is not None:
            if key not in self._data[name]:
                added += 1
            self._data[name][key] = value
        return added

    def hget(self, name: str, key: str) -> Optional[str]:
        """HGET 命令"""
        if self._check_expired(name):
            return None
        if self._type.get(name) != "hash":
            return None
        return self._data.get(name, {}).get(key)

    def hgetall(self, name: str) -> Dict[str, str]:
        """HGETALL 命令"""
        if self._check_expired(name):
            return {}
        if self._type.get(name) != "hash":
            return {}
        return dict(self._data.get(name, {}))

    # ----------------------------------------------------------
    # 通用操作
    # ----------------------------------------------------------

    def expire(self, name: str, time_seconds: Optional[int]) -> bool:
        """
        EXPIRE 命令

        Args:
            name: key
            time_seconds: 过期秒数，None 或负数表示移除过期时间（永久）
        """
        if self._check_expired(name) or name not in self._data:
            return False
        if time_seconds is None or time_seconds < 0:
            self._expiry[name] = None
        else:
            self._expiry[name] = time.time() + time_seconds
        return True

    def delete(self, *names: str) -> int:
        """DEL 命令，返回实际删除的 key 数量"""
        count = 0
        for name in names:
            if name in self._data:
                self._data.pop(name, None)
                self._expiry.pop(name, None)
                self._type.pop(name, None)
                count += 1
        return count

    def exists(self, name: str) -> bool:
        """EXISTS 命令"""
        if self._check_expired(name):
            return False
        return name in self._data

    def ttl(self, name: str) -> int:
        """
        TTL 命令

        Returns:
            剩余秒数；-1 表示永久；-2 表示 key 不存在
        """
        if self._check_expired(name) or name not in self._data:
            return -2
        exp = self._expiry.get(name)
        if exp is None:
            return -1
        remaining = int(exp - time.time())
        return max(0, remaining)


class RedisSessionManager:
    """
    Redis 会话状态管理器

    基于 Redis 持久化会话状态，支持故障恢复和水平扩展。
    当 Redis 不可用或 mock=True 时，自动降级为内存存储（_MockRedis）。

    与 Phase 1 的 gateway/session_handler/SessionManager 区别:
      - 基于 Redis 持久化，支持多实例共享会话
      - 新增检查点保存/恢复（save_checkpoint / restore_checkpoint）
      - 新增最大轮次限制（max_turns），自动裁剪旧轮次
      - 幂等键缓存支持 TTL（Redis 原生过期）

    使用方式:
        # 生产环境（自动从环境变量读取配置）
        manager = RedisSessionManager()

        # 显式指定 Redis URL
        manager = RedisSessionManager(redis_url="redis://:password@host:6379/0")

        # 开发/测试环境（内存模式，无需 Redis）
        manager = RedisSessionManager(mock=True)

        # 创建会话
        session_id = manager.create_session()

        # 添加对话轮次
        turn = manager.add_turn(
            session_id, "什么是 GDPR？", "GDPR 是...",
            metadata={"intent": "factual", "complexity": "L1"},
        )

        # 保存状态机检查点
        checkpoint = SessionCheckpoint(
            checkpoint_id="cp-001", session_id=session_id,
            state_machine_state="RETRIEVING", events=[...],
        )
        manager.save_checkpoint(session_id, checkpoint)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_turns: Optional[int] = None,
        mock: bool = False,
    ):
        """
        初始化会话管理器

        Args:
            redis_url: Redis 连接 URL（如 redis://:pass@host:6379/0）。
                       为 None 时从环境变量（REDIS_HOST/PORT/DB/PASSWORD）构造。
            ttl_seconds: 会话 TTL（秒），None 时读取 SESSION_TTL_SECONDS 环境变量（默认 3600）
            max_turns: 最大保留对话轮次，None 时读取 SESSION_MAX_TURNS 环境变量（默认 20）
            mock: 强制使用内存模式（开发/测试），不连接真实 Redis
        """
        # 从环境变量读取默认值（构造时读取，允许运行时动态配置）
        self._ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else int(os.getenv("SESSION_TTL_SECONDS", str(_DEFAULT_TTL)))
        )
        self._max_turns = (
            max_turns
            if max_turns is not None
            else int(os.getenv("SESSION_MAX_TURNS", str(_DEFAULT_MAX_TURNS)))
        )
        self._mock = mock
        self._client = None

        if not mock:
            # 尝试连接真实 Redis
            url = redis_url or self._build_redis_url()
            try:
                import redis  # 延迟导入，未安装时降级为 mock

                self._client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self._client.ping()
                logger.info("Redis 会话管理器已连接: %s", _safe_url(url))
            except Exception as e:
                logger.warning("Redis 连接失败，降级为内存模式: %s", e)
                self._client = _MockRedis()
                self._mock = True
        else:
            self._client = _MockRedis()
            logger.info("会话管理器运行在内存模式（mock）")

    @staticmethod
    def _build_redis_url() -> str:
        """从环境变量构造 Redis 连接 URL"""
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        db = os.getenv("REDIS_DB", "0")
        password = os.getenv("REDIS_PASSWORD", "")
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

    # ============================================================
    # 内部 Key 构造
    # ============================================================

    @staticmethod
    def _session_key(session_id: str) -> str:
        """构造会话状态 Redis Key"""
        return f"{_SESSION_PREFIX}{session_id}"

    @staticmethod
    def _checkpoint_key(session_id: str) -> str:
        """构造检查点 Redis Key"""
        return f"{_CHECKPOINT_PREFIX}{session_id}"

    @staticmethod
    def _idempotency_key(key: str) -> str:
        """构造幂等键 Redis Key"""
        return f"{_IDEMPOTENCY_PREFIX}{key}"

    # ============================================================
    # 会话生命周期
    # ============================================================

    def create_session(self) -> str:
        """
        创建新会话

        创建 SessionState（初始状态 RECEIVED）并持久化到 Redis，设置 TTL。

        Returns:
            新会话的 session_id（UUID）
        """
        session_id = str(uuid.uuid4())
        now = time.time()
        session = SessionState(
            session_id=session_id,
            current_state="RECEIVED",
            created_at=now,
            updated_at=now,
        )
        self._save_session(session)
        logger.debug("创建会话: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        获取会话状态

        从 Redis Hash 读取会话数据并反序列化。
        若会话不存在或已过期（Redis TTL 自动清理），返回 None。

        Args:
            session_id: 会话 ID

        Returns:
            SessionState 对象，不存在或已过期返回 None
        """
        key = self._session_key(session_id)
        data = self._client.hgetall(key)
        if not data:
            return None
        return self._deserialize_session(data)

    def update_state(self, session_id: str, state: str) -> None:
        """
        更新会话的状态机状态

        读取当前会话，更新 current_state 和 updated_at，重新持久化。
        若会话不存在则忽略（记录警告日志）。

        Args:
            session_id: 会话 ID
            state: 新状态（AgentState 枚举值字符串，如 "NORMALIZED"）
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning("更新状态失败，会话不存在: %s", session_id)
            return
        session.current_state = state
        session.updated_at = time.time()
        self._save_session(session)

    def delete_session(self, session_id: str) -> None:
        """
        删除会话

        同时删除会话状态和关联的检查点。
        幂等键不随会话删除（独立 TTL 管理）。

        Args:
            session_id: 会话 ID
        """
        self._client.delete(
            self._session_key(session_id),
            self._checkpoint_key(session_id),
        )
        logger.debug("删除会话: %s", session_id)

    def expire_session(self, session_id: str, ttl: Optional[int] = None) -> None:
        """
        设置会话过期时间

        同时更新会话状态和检查点的 TTL。
        用于主动过期（如用户登出）或续期（如用户活跃时刷新 TTL）。

        Args:
            session_id: 会话 ID
            ttl: 过期秒数，None 则使用管理器默认 TTL
        """
        ttl_val = ttl if ttl is not None else self._ttl
        self._client.expire(self._session_key(session_id), ttl_val)
        self._client.expire(self._checkpoint_key(session_id), ttl_val)
        logger.debug("设置会话 %s 过期时间: %ds", session_id, ttl_val)

    # ============================================================
    # 对话轮次管理
    # ============================================================

    def add_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionTurn:
        """
        添加一轮对话记录

        读取会话状态，调用 SessionState.add_turn 添加轮次，
        执行最大轮次限制（超过 max_turns 时裁剪最早轮次），重新持久化。

        Args:
            session_id: 会话 ID
            query: 用户查询
            answer: Agent 回答
            metadata: 附加元数据（可含 intent、complexity、entities）

        Returns:
            新创建的 SessionTurn

        Raises:
            ValueError: 会话不存在
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"会话不存在: {session_id}")

        turn = session.add_turn(query, answer, metadata)

        # 执行最大轮次限制：保留最近 max_turns 轮
        if len(session.turns) > self._max_turns:
            session.turns = session.turns[-self._max_turns:]

        self._save_session(session)
        logger.debug("会话 %s 添加轮次 %s", session_id, turn.turn_id)
        return turn

    # ============================================================
    # 检查点管理
    # ============================================================

    def save_checkpoint(self, session_id: str, checkpoint: SessionCheckpoint) -> None:
        """
        保存状态机检查点

        将检查点序列化为 JSON 存储到 Redis String，设置与会话相同的 TTL。

        Args:
            session_id: 会话 ID
            checkpoint: 检查点对象
        """
        key = self._checkpoint_key(session_id)
        self._client.set(
            key,
            json.dumps(checkpoint.to_dict(), ensure_ascii=False),
            ex=self._ttl,
        )
        logger.debug("会话 %s 保存检查点 %s", session_id, checkpoint.checkpoint_id)

    def restore_checkpoint(self, session_id: str) -> Optional[SessionCheckpoint]:
        """
        恢复状态机检查点

        从 Redis 读取检查点 JSON 并反序列化。

        Args:
            session_id: 会话 ID

        Returns:
            SessionCheckpoint 对象，不存在返回 None
        """
        key = self._checkpoint_key(session_id)
        raw = self._client.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return SessionCheckpoint.from_dict(data)

    # ============================================================
    # 幂等键缓存
    # ============================================================

    def check_idempotency(self, key: str) -> Optional[dict]:
        """
        检查幂等键是否已有缓存响应

        用于防止重复请求的重复执行。
        幂等键由调用方生成（如基于请求体哈希），TTL 为 10 分钟。

        Args:
            key: 幂等键

        Returns:
            缓存的响应字典（含 response、cached_at、ttl），不存在返回 None
        """
        redis_key = self._idempotency_key(key)
        raw = self._client.get(redis_key)
        if raw is None:
            return None
        return json.loads(raw)

    def cache_response(self, key: str, response_dict: dict) -> None:
        """
        缓存幂等响应

        将响应数据包装后缓存到 Redis，设置幂等 TTL（10 分钟）。

        Args:
            key: 幂等键
            response_dict: 响应数据
        """
        redis_key = self._idempotency_key(key)
        payload = {
            "response": response_dict,
            "cached_at": time.time(),
            "ttl": _IDEMPOTENCY_TTL,
        }
        self._client.set(
            redis_key,
            json.dumps(payload, ensure_ascii=False),
            ex=_IDEMPOTENCY_TTL,
        )

    # ============================================================
    # 内部序列化方法
    # ============================================================

    def _save_session(self, session: SessionState) -> None:
        """
        将会话状态序列化存储到 Redis Hash

        标量字段直接存储为字符串，复杂字段（turns/query_spec/budget_consumed/metadata）
        序列化为 JSON 字符串存储。设置会话 TTL。

        Redis Hash 结构:
            session_id     → str
            current_state  → str
            created_at     → str (float)
            updated_at     → str (float)
            turns          → JSON str (List[SessionTurn])
            query_spec     → JSON str | "" (None)
            budget_consumed→ JSON str
            metadata       → JSON str
        """
        key = self._session_key(session.session_id)
        mapping = {
            "session_id": session.session_id,
            "current_state": session.current_state,
            "created_at": str(session.created_at),
            "updated_at": str(session.updated_at),
            "turns": json.dumps(
                [t.to_dict() for t in session.turns], ensure_ascii=False
            ),
            "query_spec": (
                json.dumps(session.query_spec, ensure_ascii=False)
                if session.query_spec is not None
                else ""
            ),
            "budget_consumed": json.dumps(
                session.budget_consumed, ensure_ascii=False
            ),
            "metadata": json.dumps(session.metadata, ensure_ascii=False),
        }
        self._client.hset(key, mapping=mapping)
        self._client.expire(key, self._ttl)

    @staticmethod
    def _deserialize_session(data: Dict[str, str]) -> SessionState:
        """
        从 Redis Hash 数据反序列化 SessionState

        Args:
            data: Redis HGETALL 返回的字段字典（所有值为字符串）

        Returns:
            SessionState 对象
        """
        # turns: JSON 字符串 -> List[SessionTurn]
        turns_raw = data.get("turns", "[]")
        turns = (
            [SessionTurn.from_dict(t) for t in json.loads(turns_raw)]
            if turns_raw
            else []
        )

        # query_spec: JSON 字符串或空字符串 -> dict | None
        query_spec_raw = data.get("query_spec", "")
        query_spec = json.loads(query_spec_raw) if query_spec_raw else None

        # budget_consumed: JSON 字符串 -> dict
        budget_raw = data.get("budget_consumed", "{}")
        budget_consumed = json.loads(budget_raw) if budget_raw else {}

        # metadata: JSON 字符串 -> dict
        metadata_raw = data.get("metadata", "{}")
        metadata = json.loads(metadata_raw) if metadata_raw else {}

        return SessionState(
            session_id=data.get("session_id", ""),
            current_state=data.get("current_state", "RECEIVED"),
            turns=turns,
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            query_spec=query_spec,
            budget_consumed=budget_consumed,
            metadata=metadata,
        )

    # ============================================================
    # 辅助属性
    # ============================================================

    @property
    def is_mock(self) -> bool:
        """是否运行在内存模式（mock）"""
        return self._mock

    @property
    def client(self):
        """
        底层 Redis 客户端（用于高级操作或测试）

        mock 模式下返回 _MockRedis 实例。
        """
        return self._client


def _safe_url(url: str) -> str:
    """脱敏 Redis URL（隐藏密码）用于日志输出"""
    if "@" in url:
        scheme, rest = url.split("://", 1)
        if ":" in rest.split("@", 1)[0]:
            # redis://:password@host:port/db -> redis://***@host:port/db
            creds, host_part = rest.split("@", 1)
            return f"{scheme}://***@{host_part}"
    return url
