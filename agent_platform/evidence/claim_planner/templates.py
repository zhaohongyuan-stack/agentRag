"""
声明槽位模板 — 预定义各类问题的声明槽位结构

每种问题类型对应一组声明槽位（claim slots），每个槽位定义:
  - slot: 槽位标识（英文 key）
  - description: 槽位描述（中文，用于关键词匹配）
  - required: 是否为必填槽位

模板用于 ClaimPlanner 根据问题意图生成对应的声明槽位列表。
"""

from typing import Any, Dict, List


# ============================================================
# 预定义声明槽位模板
# 每个模板 key 对应一种问题类型，值为槽位定义列表
# ============================================================
CLAIM_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    # ── 阈值查询 ── 指标阈值/比例要求类问题
    "threshold": [
        {"slot": "applicable_entity", "description": "适用主体", "required": True},
        {"slot": "metric_name", "description": "指标名称", "required": True},
        {"slot": "minimum_value", "description": "最低比例/数值", "required": True},
        {"slot": "effective_date", "description": "生效时间", "required": True},
        {"slot": "exceptions", "description": "例外或附加要求", "required": False},
        {"slot": "legal_basis", "description": "法规依据", "required": True},
    ],
    # ── 定义查询 ── 术语定义类问题
    "definition": [
        {"slot": "term", "description": "被定义术语", "required": True},
        {"slot": "definition", "description": "定义内容", "required": True},
        {"slot": "source", "description": "定义来源", "required": True},
    ],
    # ── 表格取数 ── 表格数据查询类问题
    "table_lookup": [
        {"slot": "table_name", "description": "表名", "required": True},
        {"slot": "row_key", "description": "行标题", "required": True},
        {"slot": "column_key", "description": "列标题/期间", "required": True},
        {"slot": "value", "description": "单元格值", "required": True},
        {"slot": "unit", "description": "单位", "required": False},
        {"slot": "source", "description": "来源文件和页码", "required": True},
    ],
    # ── 条款查询 ── 法规条款内容查询类问题
    "clause_query": [
        {"slot": "clause_content", "description": "条款内容", "required": True},
        {"slot": "clause_no", "description": "条款编号", "required": True},
        {"slot": "source", "description": "来源文件", "required": True},
        {"slot": "normative_level", "description": "规范强度", "required": False},
    ],
    # ── 比较查询 ── 对象对比类问题
    "comparison": [
        {"slot": "entity_a", "description": "比较对象A", "required": True},
        {"slot": "entity_b", "description": "比较对象B", "required": True},
        {"slot": "dimensions", "description": "比较维度", "required": True},
        {"slot": "differences", "description": "差异点", "required": True},
        {"slot": "sources", "description": "双方来源", "required": True},
    ],
}


# ============================================================
# 通用槽位模板 — 用于无法匹配特定模板的意图
# 至少包含主要回答和来源
# ============================================================
GENERIC_TEMPLATE: List[Dict[str, Any]] = [
    {"slot": "main_answer", "description": "主要回答内容", "required": True},
    {"slot": "source", "description": "来源依据", "required": True},
]
