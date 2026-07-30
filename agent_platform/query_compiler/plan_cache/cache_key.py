"""
缓存键生成 — 为物理计划生成稳定的缓存键

职责:
  1. 将 QueryIR 与缓存上下文规范化为稳定字符串
  2. 取 SHA256 哈希作为缓存键，保证等价查询命中同一计划

缓存键组成:
  规范化问题（intent + claims + entities + constraints + risk_level）
  + 用户权限范围（user_permissions）
  + 知识库版本（index_epoch）
  + 查询约束哈希（constraints_hash）

只要上述任一要素变化（如知识库更新、权限变化），
缓存键即不同，从而避免命中过期/越权计划。

模式参考: query_ir/ir_builder.py 的 dataclass + to_dict 风格
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from ..query_ir.ir_builder import QueryIR


logger = logging.getLogger(__name__)


@dataclass
class CacheContext:
    """
    缓存上下文 — 影响计划可复用性的环境因素

    Attributes:
        user_permissions: 用户权限范围（如角色/可见知识域），不同权限不复用
        index_epoch: 知识库版本标识（如 "kb-2026-07"），版本更新需失效
        constraints_hash: 查询约束的哈希（补充约束维度的指纹）
    """

    user_permissions: str = ""
    index_epoch: str = ""
    constraints_hash: str = ""


class CacheKeyGenerator:
    """
    缓存键生成器

    根据 QueryIR 与 CacheContext 生成稳定的 SHA256 缓存键。

    用法:
        gen = CacheKeyGenerator()
        key = gen.make_key(query_ir, CacheContext(index_epoch="kb-2026-07"))
        plan_cache.put(key, physical_plan)
    """

    def make_key(self, query_ir: QueryIR, context: CacheContext) -> str:
        """
        生成缓存键

        拼接 intent / claims / entities / constraints / risk_level /
        permissions / epoch / constraints_hash，取 SHA256 哈希。

        Args:
            query_ir: 查询中间表示
            context: 缓存上下文

        Returns:
            64 字符的 SHA256 十六进制缓存键
        """
        canonical = self._canonicalize(query_ir, context)
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        logger.debug(
            "生成缓存键: intent=%s, risk=%s, epoch=%s, key=%s",
            query_ir.intent,
            query_ir.risk_level,
            context.index_epoch,
            key[:12] + "...",
        )
        return key

    # ============================================================
    # 内部方法
    # ============================================================

    def _canonicalize(self, query_ir: QueryIR, context: CacheContext) -> str:
        """
        规范化为稳定字符串

        对列表类要素排序，保证元素顺序不影响哈希。
        """
        parts = []
        parts.append(f"intent={query_ir.intent}")
        parts.append(f"risk={query_ir.risk_level}")

        # 声明槽位：按 claim_id 排序，提取关键属性
        claim_keys = sorted(
            self._claim_key(c) for c in query_ir.claims
        )
        parts.append("claims=" + "|".join(claim_keys))

        # 实体：按 JSON 规范化排序
        entity_keys = sorted(
            self._dumps(e) for e in query_ir.entities
        )
        parts.append("entities=" + "|".join(entity_keys))

        # 约束：按 JSON 规范化排序
        constraint_keys = sorted(
            self._dumps(c) for c in query_ir.constraints
        )
        parts.append("constraints=" + "|".join(constraint_keys))

        # 缓存上下文
        parts.append(f"perms={context.user_permissions}")
        parts.append(f"epoch={context.index_epoch}")
        parts.append(f"chash={context.constraints_hash}")

        return "\n".join(parts)

    @staticmethod
    def _claim_key(claim: Any) -> str:
        """
        提取声明槽位的关键指纹

        兼容 ClaimSlot 对象与 dict，取 claim_id / slot_type / description。
        """
        if hasattr(claim, "to_dict"):
            data = claim.to_dict()
        elif isinstance(claim, dict):
            data = claim
        else:
            data = {"claim_id": str(claim)}

        claim_id = data.get("claim_id", "")
        slot_type = data.get("slot_type", "")
        description = data.get("description", "")
        return f"{claim_id}:{slot_type}:{description}"

    @staticmethod
    def _dumps(obj: Any) -> str:
        """
        将对象规范化为 JSON 字符串（键排序，确保稳定）

        不可序列化对象回退为 str()。
        """
        try:
            return json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(obj)
