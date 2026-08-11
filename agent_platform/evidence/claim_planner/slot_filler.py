"""
槽位填充器 — 用证据填充声明槽位

职责:
  1. 对每个声明槽位，从证据项中匹配相关证据
  2. 将匹配的证据 ID 绑定到槽位
  3. 更新槽位状态（supported / missing / pending）

填充策略（Phase 1 纯规则版）:
  - 从声明描述中提取关键词
  - 在证据内容中做关键词匹配
  - 每个槽位最多绑定 3 条证据（按得分降序取前 3）
  - 必填槽位无匹配证据时标记为 missing
  - 可选槽位无匹配证据时标记为 pending
"""

import logging
import re
from typing import Any, List

from ..evidence_assembler.builder import ClaimSlot


logger = logging.getLogger(__name__)


class SlotFiller:
    """
    槽位填充器

    使用关键词匹配策略，将证据项绑定到声明槽位。
    匹配后更新每个槽位的 status 和 evidence_ids。

    兼容 EvidenceItem 对象和 dict 两种证据项格式。
    """

    def __init__(self, max_evidence_per_slot: int = 3, min_keyword_length: int = 2):
        """
        Args:
            max_evidence_per_slot: 每个槽位最多绑定的证据数量
            min_keyword_length: 关键词最小长度，短于此值的词不作为匹配关键词
        """
        self._max_evidence = max_evidence_per_slot
        self._min_keyword_length = min_keyword_length

    def fill(
        self,
        slots: List[ClaimSlot],
        evidence_items: List[Any],
    ) -> List[ClaimSlot]:
        """
        用证据填充声明槽位

        填充逻辑:
          1. 若无证据项，所有必填槽位标记为 missing，可选槽位标记为 pending
          2. 对每个槽位，从 description 提取关键词
          3. 在证据内容中匹配关键词
          4. 有匹配 → 绑定证据（最多 max_evidence_per_slot 条），status = supported
          5. 无匹配但存在证据 → 必填槽位 missing，可选槽位 pending

        Args:
            slots: 声明槽位列表
            evidence_items: 证据项列表（EvidenceItem 对象或 dict）

        Returns:
            填充后的声明槽位列表（原地更新，同时返回引用）
        """
        if not slots:
            return slots

        # 无证据时的处理
        if not evidence_items:
            for slot in slots:
                required = self._is_required(slot)
                slot.status = "missing" if required else "pending"
                slot.evidence_ids = []
            logger.debug("无证据项，%d 个槽位已标记状态", len(slots))
            return slots

        for slot in slots:
            matched = self._match_evidence(slot, evidence_items)

            if matched:
                # 有匹配证据 → supported
                slot.evidence_ids = [
                    self._get_attr(e, "evidence_id", "") for e in matched[: self._max_evidence]
                ]
                slot.status = "supported"
            else:
                # 无匹配证据
                required = self._is_required(slot)
                slot.status = "missing" if required else "pending"
                slot.evidence_ids = []

            logger.debug(
                "槽位 '%s' 填充完成: status=%s, evidence=%d",
                slot.claim_id,
                slot.status,
                len(slot.evidence_ids),
            )

        return slots

    # ============================================================
    # 内部方法
    # ============================================================

    def _match_evidence(
        self,
        slot: ClaimSlot,
        evidence_items: List[Any],
    ) -> List[Any]:
        """
        为单个槽位匹配证据

        策略:
          1. 从槽位描述提取关键词
          2. 在每条证据的 content + evidence_snippet 中匹配
          3. 按得分降序排序匹配结果

        Args:
            slot: 声明槽位
            evidence_items: 证据项列表

        Returns:
            匹配到的证据项列表（按得分降序）
        """
        keywords = self._extract_keywords(slot.description)

        matched: List[Any] = []
        for ev in evidence_items:
            ev_text = self._get_evidence_text(ev).lower()
            if not keywords:
                # 无关键词时（如描述为空），所有证据都视为候选
                matched.append(ev)
            elif any(kw in ev_text for kw in keywords):
                matched.append(ev)

        # 按得分降序排序
        matched.sort(key=lambda e: self._get_score(e), reverse=True)
        return matched

    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词

        按空格和标点分割，过滤过短的词。

        Args:
            text: 输入文本

        Returns:
            关键词列表（小写）
        """
        if not text:
            return []
        words = re.split(r"[\s,，。、；;：:（）()\/|]+", text)
        return [w.lower() for w in words if len(w) >= self._min_keyword_length]

    def _is_required(self, slot: ClaimSlot) -> bool:
        """
        判断槽位是否为必填

        通过解析 slot_type 中的编码判断:
          - 格式为 "{template_key}|required" 或 "{template_key}|optional"
          - 默认值为 True（无法解析时视为必填）

        Args:
            slot: 声明槽位

        Returns:
            True 表示必填，False 表示可选
        """
        if not slot.slot_type:
            return True
        if "|" in slot.slot_type:
            parts = slot.slot_type.split("|", 1)
            return parts[1].strip().lower() != "optional"
        return True

    # ============================================================
    # 证据项属性访问工具 — 兼容对象和 dict
    # ============================================================

    @staticmethod
    def _get_attr(ev: Any, name: str, default: Any = "") -> Any:
        """
        从证据项获取属性值，兼容对象和字典

        Args:
            ev: 证据项（EvidenceItem 对象或 dict）
            name: 属性名/键名
            default: 默认值

        Returns:
            属性值
        """
        if isinstance(ev, dict):
            return ev.get(name, default)
        return getattr(ev, name, default)

    def _get_evidence_text(self, ev: Any) -> str:
        """
        获取证据项的文本内容（content + evidence_snippet）

        Args:
            ev: 证据项

        Returns:
            拼接后的文本
        """
        content = self._get_attr(ev, "content", "") or ""
        snippet = self._get_attr(ev, "evidence_snippet", "") or ""
        return f"{content} {snippet}"

    def _get_score(self, ev: Any) -> float:
        """
        获取证据项的得分

        Args:
            ev: 证据项

        Returns:
            得分（float），无得分时返回 0.0
        """
        score = self._get_attr(ev, "score", 0.0)
        return float(score) if score is not None else 0.0
