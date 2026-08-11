"""
声明槽位规划器 — 根据问题意图生成声明槽位

职责:
  1. 将意图分类器的 intent 值映射到预定义模板
  2. 根据模板生成 ClaimSlot 列表
  3. 支持自定义模板注册

ClaimSlot 复用 evidence_assembler/builder.py 中的定义，
保证与证据组装器的数据结构一致。
"""

import logging
from typing import Any, Dict, List, Optional

from ..evidence_assembler.builder import ClaimSlot
from .templates import CLAIM_TEMPLATES, GENERIC_TEMPLATE


logger = logging.getLogger(__name__)


# ============================================================
# 意图 → 模板 key 映射
# 同时支持 intent_classifier 的原始输出值和带 _query 后缀的变体
# ============================================================
INTENT_TO_TEMPLATE: Dict[str, str] = {
    # 阈值查询
    "threshold_query": "threshold",
    "threshold": "threshold",
    # 定义查询
    "definition_query": "definition",
    "definition": "definition",
    # 表格取数
    "table_lookup": "table_lookup",
    # 条款查询
    "clause_query": "clause_query",
    # 比较查询
    "comparison": "comparison",
}


class ClaimPlanner:
    """
    声明槽位规划器

    根据查询规格（QuerySpec）中的意图类型，从预定义模板生成声明槽位列表。
    每个槽位对应回答中需要覆盖的一个声明点，后续由 SlotFiller 填充证据。

    支持通过 register_template() 注册自定义模板，扩展新的意图类型。
    """

    def __init__(self, templates: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """
        Args:
            templates: 自定义模板字典，为 None 时使用默认 CLAIM_TEMPLATES。
                       传入时会与默认模板合并（自定义模板覆盖同名 key）。
        """
        # 复制默认模板，避免修改全局常量
        self._templates: Dict[str, List[Dict[str, Any]]] = {
            k: list(v) for k, v in CLAIM_TEMPLATES.items()
        }
        if templates:
            for key, tmpl in templates.items():
                self._templates[key] = list(tmpl)
        logger.debug("ClaimPlanner 初始化完成，已加载 %d 个模板", len(self._templates))

    def plan(self, query_spec: Dict[str, Any]) -> List[ClaimSlot]:
        """
        根据查询规格生成声明槽位列表

        流程:
          1. 从 query_spec 提取 intent
          2. 通过 INTENT_TO_TEMPLATE 映射到模板 key
          3. 从模板生成 ClaimSlot 列表
          4. 若意图无法映射，使用通用模板

        Args:
            query_spec: 查询规格字典，需包含 "intent" 字段。
                       也可包含 "template_key" 字段直接指定模板（优先于 intent 映射）。

        Returns:
            声明槽位列表（List[ClaimSlot]）
        """
        # 优先使用 query_spec 中显式指定的 template_key
        template_key = query_spec.get("template_key")
        if template_key and template_key in self._templates:
            logger.debug("使用显式指定的模板: %s", template_key)
            template = self._templates[template_key]
            return self._build_slots_from_template(template, template_key)

        # 通过 intent 映射到模板 key
        intent = query_spec.get("intent", "unknown")
        template_key = INTENT_TO_TEMPLATE.get(intent)
        if template_key and template_key in self._templates:
            logger.debug("意图 '%s' 映射到模板 '%s'", intent, template_key)
            template = self._templates[template_key]
            return self._build_slots_from_template(template, template_key)

        # 无法映射，使用通用模板
        logger.debug("意图 '%s' 无匹配模板，使用通用模板", intent)
        return self._build_slots_from_template(GENERIC_TEMPLATE, "generic")

    def register_template(
        self, key: str, template: List[Dict[str, Any]]
    ) -> None:
        """
        注册自定义模板

        注册后可通过 template_key 直接引用，也可在 INTENT_TO_TEMPLATE 中
        添加 intent → key 的映射以支持自动匹配。

        Args:
            key: 模板标识 key
            template: 槽位定义列表，每个元素包含 slot/description/required 字段
        """
        self._templates[key] = list(template)
        logger.info("已注册自定义模板: %s（%d 个槽位）", key, len(template))

    def get_template(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        获取指定模板的副本

        Args:
            key: 模板标识 key

        Returns:
            模板定义列表的副本，不存在时返回 None
        """
        template = self._templates.get(key)
        return list(template) if template is not None else None

    # ============================================================
    # 内部方法
    # ============================================================

    def _build_slots_from_template(
        self,
        template: List[Dict[str, Any]],
        template_key: str,
    ) -> List[ClaimSlot]:
        """
        从模板定义构建 ClaimSlot 列表

        将模板中的 slot/description/required 字段映射到 ClaimSlot:
          - claim_id   ← slot（槽位标识，作为唯一 key）
          - description ← description（中文描述，用于关键词匹配）
          - slot_type  ← "{template_key}|required" 或 "{template_key}|optional"
                        （编码模板来源和必填属性，便于 SlotFiller 解析）
          - status     ← "pending"（初始状态，等待填充）

        Args:
            template: 模板槽位定义列表
            template_key: 模板标识 key

        Returns:
            ClaimSlot 列表
        """
        slots: List[ClaimSlot] = []
        for slot_def in template:
            required = slot_def.get("required", True)
            slot = ClaimSlot(
                claim_id=slot_def.get("slot", ""),
                description=slot_def.get("description", ""),
                slot_type=f"{template_key}|{'required' if required else 'optional'}",
                status="pending",
            )
            slots.append(slot)
        return slots
