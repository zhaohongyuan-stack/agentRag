"""
版本检查器适配器 — M5.3 工具模块

检查法规版本在指定日期是否有效。
"""

import logging
from datetime import datetime
from typing import Any, Dict

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)


def version_checker_handler(input_data: Dict[str, Any]) -> Any:
    """
    版本检查处理函数

    Args:
        input_data:
            - effective_date: str — 版本生效日期（YYYY-MM-DD）
            - superseded_date: str — 版本被替代日期（可选）
            - query_date: str — 查询日期（YYYY-MM-DD）

    Returns:
        dict: {is_effective, status}
    """
    effective_date = input_data.get("effective_date", "")
    superseded_date = input_data.get("superseded_date", "")
    query_date = input_data.get("query_date", "")

    if not effective_date or not query_date:
        raise ValueError("需要 effective_date 和 query_date 参数")

    try:
        effective = datetime.strptime(effective_date, "%Y-%m-%d")
        query = datetime.strptime(query_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式错误（需 YYYY-MM-DD）: {e}")

    if query < effective:
        return {"is_effective": False, "status": "not_yet_effective"}

    if superseded_date:
        try:
            superseded = datetime.strptime(superseded_date, "%Y-%m-%d")
            if query >= superseded:
                return {"is_effective": False, "status": "superseded"}
        except ValueError:
            pass

    return {"is_effective": True, "status": "active"}


VERSION_CHECKER_MANIFEST = ToolManifest(
    name="version_checker",
    version="1.0.0",
    description="检查法规版本在指定日期是否有效",
    input_schema={
        "type": "object",
        "properties": {
            "effective_date": {"type": "string", "description": "生效日期 YYYY-MM-DD"},
            "superseded_date": {"type": "string", "description": "被替代日期"},
            "query_date": {"type": "string", "description": "查询日期 YYYY-MM-DD"},
        },
        "required": ["effective_date", "query_date"],
    },
    capabilities=["read_only"],
    permission_level="public",
    is_read_only=True,
    timeout_ms=1000,
    idempotent=True,
    cost_level="low",
    result_trust_level="verified",
)
