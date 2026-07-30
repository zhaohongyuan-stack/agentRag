"""
检查点管理器 — 保存、加载、版本管理与清理

支持多版本检查点（按版本号保留最近 N 个），并提供基于 Redis 或
内存字典的存储后端。Redis 不可用时自动回退到内存字典，保证核心
流程不硬依赖外部存储。

存储键格式:
  - 检查点:  checkpoint:{session_id}:{request_id}:{version}
  - 版本索引: checkpoint:{session_id}:{request_id}:versions

设计要点:
  1. 版本递增: 每次保存版本号 +1，最新版本号由当前已有版本决定
  2. 自动清理: 保存后清理超过 MAX_VERSIONS 的旧版本
  3. 存储无关: 通过统一接口屏蔽 Redis / 内存差异，均序列化为 JSON
  4. 健壮回退: Redis 客户端为 None 或运行时出错时回退到内存字典
"""

import json
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from .checkpoint_models import Checkpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    检查点管理器 — 保存、加载、版本管理与清理

    保存策略:
      - 每次状态迁移时保存
      - DAG 每个任务完成时保存
      - 检查点按版本号递增，保留最近 N 个版本（默认 3 个）

    存储后端:
      - Redis（优先）
      - 内存字典（Redis 不可用时回退）
    """

    MAX_VERSIONS = 3  # 保留最近 3 个版本

    # 存储键前缀与版本索引后缀
    _KEY_PREFIX = "checkpoint"
    _VERSIONS_SUFFIX = "versions"

    def __init__(self, redis_client: Any = None):
        """
        初始化检查点管理器

        Args:
            redis_client: Redis 客户端实例，为 None 时使用内存字典。
                          运行时若 Redis 操作异常，也会降级到内存字典。
        """
        self._redis = redis_client
        # 内存存储后端: {key: json_str}
        self._memory: Dict[str, str] = {}
        self._lock = threading.Lock()

        if self._redis is not None:
            logger.debug("检查点管理器使用 Redis 存储后端")
        else:
            logger.debug("检查点管理器使用内存字典存储后端（Redis 不可用）")

    # ============================================================
    # 存储键构造
    # ============================================================

    def _make_key(self, session_id: str, request_id: str, version: int) -> str:
        """构造检查点存储键: checkpoint:{session_id}:{request_id}:{version}"""
        return f"{self._KEY_PREFIX}:{session_id}:{request_id}:{version}"

    def _make_versions_key(self, session_id: str, request_id: str) -> str:
        """构造版本索引键: checkpoint:{session_id}:{request_id}:versions"""
        return f"{self._KEY_PREFIX}:{session_id}:{request_id}:{self._VERSIONS_SUFFIX}"

    # ============================================================
    # 原始存储读写（屏蔽 Redis / 内存差异，含降级）
    # ============================================================

    def _store_get(self, key: str) -> Optional[str]:
        """读取原始字符串值，不存在返回 None"""
        if self._redis is not None:
            try:
                return self._redis.get(key)
            except Exception as e:
                logger.warning("Redis 读取失败，降级到内存存储: %s", e)
                self._redis = None  # 永久降级，保证后续一致性
        with self._lock:
            return self._memory.get(key)

    def _store_set(self, key: str, value: str) -> None:
        """写入原始字符串值"""
        if self._redis is not None:
            try:
                self._redis.set(key, value)
                return
            except Exception as e:
                logger.warning("Redis 写入失败，降级到内存存储: %s", e)
                self._redis = None
        with self._lock:
            self._memory[key] = value

    def _store_delete(self, key: str) -> None:
        """删除原始键"""
        if self._redis is not None:
            try:
                self._redis.delete(key)
                return
            except Exception as e:
                logger.warning("Redis 删除失败，降级到内存存储: %s", e)
                self._redis = None
        with self._lock:
            self._memory.pop(key, None)

    # ============================================================
    # 版本索引管理
    # ============================================================

    def _get_versions(self, session_id: str, request_id: str) -> List[int]:
        """获取某 session+request 的所有版本号（升序）"""
        raw = self._store_get(self._make_versions_key(session_id, request_id))
        if not raw:
            return []
        try:
            versions = json.loads(raw)
            if not isinstance(versions, list):
                return []
            return sorted(int(v) for v in versions)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(
                "版本索引反序列化失败 sid=%s rid=%s: %s", session_id, request_id, e
            )
            return []

    def _set_versions(self, session_id: str, request_id: str, versions: List[int]) -> None:
        """写入版本索引（去重并升序）"""
        unique_sorted = sorted(set(int(v) for v in versions))
        self._store_set(
            self._make_versions_key(session_id, request_id),
            json.dumps(unique_sorted),
        )

    def _next_version(self, session_id: str, request_id: str) -> int:
        """计算下一个版本号（已有最大版本号 +1，无则从 1 开始）"""
        versions = self._get_versions(session_id, request_id)
        if not versions:
            return 1
        return max(versions) + 1

    # ============================================================
    # 公共 API
    # ============================================================

    def save(self, checkpoint: Checkpoint) -> str:
        """
        保存检查点

        保存逻辑:
          1. 版本号递增（覆盖传入的 version）
          2. 序列化为 JSON 写入 Redis/内存
          3. 更新版本索引
          4. 清理超过 MAX_VERSIONS 的旧版本

        Args:
            checkpoint: 待保存的检查点

        Returns:
            checkpoint_id
        """
        session_id = checkpoint.session_id
        request_id = checkpoint.request_id

        # 1. 分配递增版本号
        version = self._next_version(session_id, request_id)
        checkpoint.version = version

        # 2. 确保检查点 ID（__post_init__ 通常已生成，此处兜底）
        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = f"cp-{uuid.uuid4().hex[:8]}"

        # 3. 写入存储
        key = self._make_key(session_id, request_id, version)
        payload = json.dumps(checkpoint.to_dict(), ensure_ascii=False)
        self._store_set(key, payload)

        # 4. 更新版本索引
        versions = self._get_versions(session_id, request_id)
        if version not in versions:
            versions.append(version)
            self._set_versions(session_id, request_id, versions)

        # 5. 清理旧版本
        self.cleanup_old_versions(session_id, request_id)

        logger.info(
            "保存检查点 sid=%s rid=%s version=%d cp_id=%s",
            session_id, request_id, version, checkpoint.checkpoint_id,
        )
        return checkpoint.checkpoint_id

    def load_latest(self, session_id: str, request_id: str) -> Optional[Checkpoint]:
        """
        加载最新的检查点

        Args:
            session_id: 会话 ID
            request_id: 请求 ID

        Returns:
            最新版本的 Checkpoint，无检查点时返回 None
        """
        versions = self._get_versions(session_id, request_id)
        if not versions:
            return None
        return self.load_by_version(session_id, request_id, max(versions))

    def load_by_version(
        self, session_id: str, request_id: str, version: int
    ) -> Optional[Checkpoint]:
        """
        加载指定版本的检查点

        Args:
            session_id: 会话 ID
            request_id: 请求 ID
            version: 版本号

        Returns:
            对应版本的 Checkpoint，不存在或反序列化失败时返回 None
        """
        raw = self._store_get(self._make_key(session_id, request_id, version))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "检查点反序列化失败 sid=%s rid=%s version=%d: %s",
                session_id, request_id, version, e,
            )
            return None

    def list_versions(self, session_id: str, request_id: str) -> List[int]:
        """
        列出所有可用版本号

        Args:
            session_id: 会话 ID
            request_id: 请求 ID

        Returns:
            版本号列表（升序），无检查点时返回空列表
        """
        return self._get_versions(session_id, request_id)

    def cleanup_old_versions(self, session_id: str, request_id: str) -> None:
        """
        清理超过 MAX_VERSIONS 的旧版本

        保留最新的 MAX_VERSIONS 个版本，删除其余版本的检查点数据与索引。
        """
        versions = self._get_versions(session_id, request_id)
        if len(versions) <= self.MAX_VERSIONS:
            return

        # 保留最新的 MAX_VERSIONS 个，删除其余
        sorted_desc = sorted(versions, reverse=True)
        keep = set(sorted_desc[: self.MAX_VERSIONS])
        to_delete = [v for v in versions if v not in keep]

        for v in to_delete:
            self._store_delete(self._make_key(session_id, request_id, v))

        self._set_versions(session_id, request_id, sorted(keep))

        logger.debug(
            "清理旧版本 sid=%s rid=%s 删除=%s 保留=%s",
            session_id, request_id, to_delete, sorted(keep),
        )

    def delete_all(self, session_id: str, request_id: str) -> None:
        """
        删除指定会话+请求的所有检查点

        清除全部版本检查点数据与版本索引。
        """
        versions = self._get_versions(session_id, request_id)
        for v in versions:
            self._store_delete(self._make_key(session_id, request_id, v))
        self._store_delete(self._make_versions_key(session_id, request_id))

        logger.info(
            "删除所有检查点 sid=%s rid=%s 共%d个", session_id, request_id, len(versions)
        )

    def get_stats(self) -> dict:
        """
        获取检查点统计信息

        Returns:
            统计字典，含 backend / max_versions，以及内存后端下的
            checkpoint_count / session_request_count
        """
        if self._redis is not None:
            return {
                "backend": "redis",
                "max_versions": self.MAX_VERSIONS,
            }

        with self._lock:
            checkpoint_count = sum(
                1
                for k in self._memory
                if k.startswith(f"{self._KEY_PREFIX}:")
                and not k.endswith(f":{self._VERSIONS_SUFFIX}")
            )
            session_request_count = sum(
                1 for k in self._memory if k.endswith(f":{self._VERSIONS_SUFFIX}")
            )

        return {
            "backend": "memory",
            "max_versions": self.MAX_VERSIONS,
            "checkpoint_count": checkpoint_count,
            "session_request_count": session_request_count,
        }

    def __repr__(self) -> str:
        backend = "redis" if self._redis is not None else "memory"
        return f"CheckpointManager(backend={backend}, max_versions={self.MAX_VERSIONS})"
