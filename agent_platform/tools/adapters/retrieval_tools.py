"""
检索工具适配器 — 将检索服务 API 封装为可被 LLM Function Calling 调用的工具

提供 5 个检索工具:
  1. search_chunks      — 统一检索（调用 RetrievalClient.search）
  2. get_chunk_content  — 获取单个 chunk 原文
  3. search_table       — 表格搜索（按 table_name 或 query）
  4. get_cell_value     — 获取单元格值（chunk_type=cell_fact）
  5. list_documents     — 列出文档

设计要点:
  1. RetrievalToolFactory 接收 RetrievalClient 实例与可选 base_url，
     通过 create_all() 返回 [(manifest, handler), ...] 列表
  2. search_chunks 复用 RetrievalClient.search()（HTTP/In-Process 双模式）
  3. 其余工具通过 urllib.request 直接调用检索服务（与 retrieval_client.py 风格一致）
  4. handler 签名: handler(input_data: dict) -> dict
  5. 错误处理: 网络错误统一返回 {success: False, error: "..."}
  6. content 截断: search_chunks 截断 200 字符，search_table 截断 300 字符

检索服务 API（运行在 http://127.0.0.1:8000）:
  POST /api/v1/search          — 统一检索入口
  GET  /api/v1/chunks/{id}     — 获取单个 chunk
  GET  /api/v1/documents       — 列出文档
  POST /api/v1/chunks/search   — 按元数据搜索 chunks
"""

import json
import logging
import urllib.error
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..tool_models import ToolManifest

logger = logging.getLogger(__name__)

# 默认检索服务地址
_DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# content 截断长度
_CHUNK_CONTENT_LIMIT = 200
_TABLE_CONTENT_LIMIT = 300


# ============================================================
# 检索工具工厂
# ============================================================
class RetrievalToolFactory:
    """
    检索工具工厂

    将检索服务 API 封装为 5 个可被 LLM Function Calling 调用的工具。
    每个 handler 通过闭包（类绑定）持有 retrieval_client 与 base_url。

    用法:
        from agent_platform.gateway.request_handler.retrieval_client import RetrievalClient
        from agent_platform.tools.adapters.retrieval_tools import RetrievalToolFactory

        client = RetrievalClient(base_url="http://127.0.0.1:8000")
        factory = RetrievalToolFactory(retrieval_client=client)

        # 注册到工具注册表
        for manifest, handler in factory.create_all():
            registry.register(manifest, handler)

        # 或一次性注册
        factory.register_all(registry)
    """

    def __init__(
        self,
        retrieval_client: Any,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            retrieval_client: RetrievalClient 实例（用于 search_chunks）
            base_url: 检索服务基础 URL，为 None 时从 retrieval_client 推断
        """
        self._retrieval_client = retrieval_client

        if base_url:
            self._base_url = base_url.rstrip("/")
        else:
            # 优先复用 retrieval_client 已配置的 base_url，保持一致性
            client_base = getattr(retrieval_client, "_base_url", None)
            self._base_url = (client_base or _DEFAULT_BASE_URL).rstrip("/")

        # 复用 retrieval_client 的超时配置
        self._timeout_ms = getattr(retrieval_client, "_timeout_ms", 5000)
        self._timeout = self._timeout_ms / 1000.0

    # ------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------
    def create_all(self) -> List[Tuple[ToolManifest, Callable]]:
        """
        创建全部 5 个检索工具

        Returns:
            [(ToolManifest, handler), ...] 列表
        """
        return [
            (self._build_search_chunks_manifest(), self._search_chunks_handler),
            (self._build_get_chunk_content_manifest(), self._get_chunk_content_handler),
            (self._build_search_table_manifest(), self._search_table_handler),
            (self._build_get_cell_value_manifest(), self._get_cell_value_handler),
            (self._build_list_documents_manifest(), self._list_documents_handler),
        ]

    def register_all(self, registry: Any) -> List[str]:
        """
        便捷方法：将全部检索工具注册到 ToolRegistry

        Args:
            registry: ToolRegistry 实例

        Returns:
            已注册的工具名称列表
        """
        tools = self.create_all()
        registered = []
        for manifest, handler in tools:
            if registry.exists(manifest.name):
                registry.unregister(manifest.name)
            registry.register(manifest, handler)
            registered.append(manifest.name)
        logger.info("注册 %d 个检索工具: %s", len(registered), registered)
        return registered

    # ============================================================
    # 工具 1: search_chunks — 统一检索
    # ============================================================
    def _search_chunks_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一检索工具 handler

        调用 RetrievalClient.search()，返回截断后的命中列表。

        Args:
            input_data:
                - query: str — 检索查询文本（必填）
                - strategy: str — 检索策略（默认 hybrid）
                - top_k: int — 返回结果数（默认 10）
                - filters: dict — 元数据过滤条件（可选）

        Returns:
            {success, hit_count, hits, latency_ms} 或 {success: False, error}
        """
        query = input_data.get("query")
        if not query:
            return {"success": False, "error": "缺少必填参数: query"}

        strategy = input_data.get("strategy") or "hybrid"
        top_k = _coerce_int(input_data.get("top_k"), default=10)
        filters = input_data.get("filters")

        try:
            result = self._retrieval_client.search(
                query=query,
                strategy=strategy,
                top_k=top_k,
                filters=filters,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("search_chunks 调用异常")
            return {"success": False, "error": f"检索异常: {e}"}

        if not result.success:
            return {"success": False, "error": result.error or "检索失败"}

        hits = []
        for hit in result.hits or []:
            content = hit.get("content", "") or ""
            hits.append({
                "chunk_id": hit.get("chunk_id", ""),
                "content": content[:_CHUNK_CONTENT_LIMIT],
                "score": hit.get("score", 0.0),
                "citation": hit.get("citation", ""),
                "source_doc": (
                    hit.get("source_doc")
                    or hit.get("doc_name")
                    or hit.get("doc_id", "")
                ),
            })

        return {
            "success": True,
            "hit_count": len(hits),
            "hits": hits,
            "latency_ms": result.latency_ms,
        }

    def _build_search_chunks_manifest(self) -> ToolManifest:
        return ToolManifest(
            name="search_chunks",
            version="1.0.0",
            description="统一检索工具：基于查询文本检索知识库 chunks，支持多种检索策略",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询文本",
                    },
                    "strategy": {
                        "type": "string",
                        "description": (
                            "检索策略: hybrid(默认)/bm25/dense/exact/"
                            "metadata/relation/table"
                        ),
                        "default": "hybrid",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数",
                        "default": 10,
                    },
                    "filters": {
                        "type": "object",
                        "description": "元数据过滤条件（如 doc_id、table_name 等）",
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "hit_count": {"type": "integer"},
                    "hits": {"type": "array"},
                    "latency_ms": {"type": "number"},
                    "error": {"type": "string"},
                },
            },
            capabilities=["read_only"],
            permission_level="public",
            is_read_only=True,
            timeout_ms=5000,
            idempotent=True,
            cost_level="low",
            result_trust_level="verified",
        )

    # ============================================================
    # 工具 2: get_chunk_content — 获取 chunk 原文
    # ============================================================
    def _get_chunk_content_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取单个 chunk 原文

        调用 GET /api/v1/chunks/{chunk_id}。

        Args:
            input_data:
                - chunk_id: str — chunk 唯一标识（必填）

        Returns:
            {success, chunk_id, content, chunk_type, doc_name,
             hierarchy_path, metadata} 或 {success: False, error}
        """
        chunk_id = input_data.get("chunk_id")
        if not chunk_id:
            return {"success": False, "error": "缺少必填参数: chunk_id"}

        try:
            data = self._http_get(
                f"/api/v1/chunks/{urllib.parse.quote(str(chunk_id), safe='')}"
            )
        except _RetrievalHttpError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            logger.exception("get_chunk_content 调用异常")
            return {"success": False, "error": f"获取 chunk 异常: {e}"}

        if not isinstance(data, dict):
            return {"success": False, "error": "检索服务返回数据格式异常"}

        return {
            "success": True,
            "chunk_id": chunk_id,
            "content": data.get("content", ""),
            "chunk_type": data.get("chunk_type", ""),
            "doc_name": data.get("doc_name") or data.get("source_doc", ""),
            "hierarchy_path": data.get("hierarchy_path", []),
            "metadata": data.get("metadata", {}),
        }

    def _build_get_chunk_content_manifest(self) -> ToolManifest:
        return ToolManifest(
            name="get_chunk_content",
            version="1.0.0",
            description="获取指定 chunk 的完整原文内容及元数据",
            input_schema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "chunk 唯一标识",
                    },
                },
                "required": ["chunk_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "chunk_id": {"type": "string"},
                    "content": {"type": "string"},
                    "chunk_type": {"type": "string"},
                    "doc_name": {"type": "string"},
                    "hierarchy_path": {"type": "array"},
                    "metadata": {"type": "object"},
                    "error": {"type": "string"},
                },
            },
            capabilities=["read_only"],
            permission_level="public",
            is_read_only=True,
            timeout_ms=5000,
            idempotent=True,
            cost_level="low",
            result_trust_level="verified",
        )

    # ============================================================
    # 工具 3: search_table — 表格搜索
    # ============================================================
    def _search_table_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        表格搜索

        优先用 POST /api/v1/chunks/search 按 table_name 过滤；
        若仅有 query，则用 POST /api/v1/search（strategy=table）。

        Args:
            input_data:
                - table_name: str — 表格名称（可选）
                - query: str — 查询文本（可选）
                - sheet_name: str — 工作表名称（可选，客户端过滤）
                - limit: int — 返回结果数（默认 20）

        Returns:
            {success, hit_count, hits} 或 {success: False, error}
        """
        table_name = input_data.get("table_name")
        query = input_data.get("query")
        sheet_name = input_data.get("sheet_name")
        limit = _coerce_int(input_data.get("limit"), default=20)

        raw_hits: List[dict] = []

        try:
            if table_name:
                # 优先按 table_name 过滤
                body = {"table_name": table_name, "limit": limit}
                resp = self._http_post("/api/v1/chunks/search", body=body)
                raw_hits = resp if isinstance(resp, list) else []
            elif query:
                # 仅有 query，使用统一检索 strategy=table
                filters: Dict[str, Any] = {}
                if sheet_name:
                    filters["sheet_name"] = sheet_name
                result = self._retrieval_client.search(
                    query=query,
                    strategy="table",
                    top_k=limit,
                    filters=filters or None,
                )
                if not result.success:
                    return {"success": False, "error": result.error or "表格检索失败"}
                raw_hits = list(result.hits or [])
            else:
                return {"success": False, "error": "需要 table_name 或 query 参数"}
        except _RetrievalHttpError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            logger.exception("search_table 调用异常")
            return {"success": False, "error": f"表格检索异常: {e}"}

        # sheet_name 客户端过滤（chunks/search 不支持 sheet_name 字段）
        if sheet_name:
            raw_hits = [h for h in raw_hits if h.get("sheet_name") == sheet_name]

        hits = []
        for hit in raw_hits:
            content = hit.get("content", "") or ""
            hits.append({
                "chunk_id": hit.get("chunk_id", ""),
                "content": content[:_TABLE_CONTENT_LIMIT],
                "table_name": hit.get("table_name", ""),
                "sheet_name": hit.get("sheet_name", ""),
                "cell_ref": hit.get("cell_ref", ""),
            })

        return {
            "success": True,
            "hit_count": len(hits),
            "hits": hits,
        }

    def _build_search_table_manifest(self) -> ToolManifest:
        return ToolManifest(
            name="search_table",
            version="1.0.0",
            description="表格搜索工具：按表格名称或查询文本检索表格类 chunk",
            input_schema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "表格名称（优先按名称过滤）",
                    },
                    "query": {
                        "type": "string",
                        "description": "查询文本（无 table_name 时使用 table 策略检索）",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "工作表名称（可选，对结果做过滤）",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数",
                        "default": 20,
                    },
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "hit_count": {"type": "integer"},
                    "hits": {"type": "array"},
                    "error": {"type": "string"},
                },
            },
            capabilities=["read_only"],
            permission_level="public",
            is_read_only=True,
            timeout_ms=5000,
            idempotent=True,
            cost_level="low",
            result_trust_level="verified",
        )

    # ============================================================
    # 工具 4: get_cell_value — 获取单元格值
    # ============================================================
    def _get_cell_value_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取单元格值

        调用 POST /api/v1/chunks/search（chunk_type=cell_fact, table_name），
        可按 cell_ref / row_label / column_label 过滤。

        Args:
            input_data:
                - table_name: str — 表格名称（必填）
                - cell_ref: str — 单元格引用，如 "C5"（可选）
                - row_label: str — 行标签（可选）
                - column_label: str — 列标签（可选）
                - limit: int — 返回结果数（默认 10）

        Returns:
            {success, cell_count, cells} 或 {success: False, error}
        """
        table_name = input_data.get("table_name")
        if not table_name:
            return {"success": False, "error": "缺少必填参数: table_name"}

        cell_ref = input_data.get("cell_ref")
        row_label = input_data.get("row_label")
        column_label = input_data.get("column_label")
        limit = _coerce_int(input_data.get("limit"), default=10)

        try:
            body = {
                "chunk_type": "cell_fact",
                "table_name": table_name,
                "limit": limit,
            }
            resp = self._http_post("/api/v1/chunks/search", body=body)
            raw_hits = resp if isinstance(resp, list) else []
        except _RetrievalHttpError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            logger.exception("get_cell_value 调用异常")
            return {"success": False, "error": f"单元格检索异常: {e}"}

        # 按 cell_ref / row_label / column_label 过滤
        if cell_ref:
            raw_hits = [h for h in raw_hits if h.get("cell_ref") == cell_ref]
        if row_label:
            raw_hits = [h for h in raw_hits if h.get("row_label") == row_label]
        if column_label:
            raw_hits = [h for h in raw_hits if h.get("column_label") == column_label]

        cells = []
        for hit in raw_hits:
            cells.append({
                "chunk_id": hit.get("chunk_id", ""),
                "content": hit.get("content", ""),
                "cell_ref": hit.get("cell_ref", ""),
                "row_label": hit.get("row_label", ""),
                "column_label": hit.get("column_label", ""),
            })

        return {
            "success": True,
            "cell_count": len(cells),
            "cells": cells,
        }

    def _build_get_cell_value_manifest(self) -> ToolManifest:
        return ToolManifest(
            name="get_cell_value",
            version="1.0.0",
            description="获取表格单元格值：按表格名称检索 cell_fact 类型 chunk，支持按单元格引用/行列标签过滤",
            input_schema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "表格名称",
                    },
                    "cell_ref": {
                        "type": "string",
                        "description": "单元格引用，如 'C5'",
                    },
                    "row_label": {
                        "type": "string",
                        "description": "行标签",
                    },
                    "column_label": {
                        "type": "string",
                        "description": "列标签",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数",
                        "default": 10,
                    },
                },
                "required": ["table_name"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "cell_count": {"type": "integer"},
                    "cells": {"type": "array"},
                    "error": {"type": "string"},
                },
            },
            capabilities=["read_only"],
            permission_level="public",
            is_read_only=True,
            timeout_ms=5000,
            idempotent=True,
            cost_level="low",
            result_trust_level="verified",
        )

    # ============================================================
    # 工具 5: list_documents — 列出文档
    # ============================================================
    def _list_documents_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        列出文档

        调用 GET /api/v1/documents?limit=N。

        Args:
            input_data:
                - limit: int — 返回文档数（默认 20）

        Returns:
            {success, doc_count, documents} 或 {success: False, error}
        """
        limit = _coerce_int(input_data.get("limit"), default=20)

        try:
            resp = self._http_get("/api/v1/documents", params={"limit": limit})
            docs = resp if isinstance(resp, list) else []
        except _RetrievalHttpError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            logger.exception("list_documents 调用异常")
            return {"success": False, "error": f"列出文档异常: {e}"}

        documents = []
        for doc in docs:
            documents.append({
                "doc_id": doc.get("doc_id", ""),
                "doc_name": doc.get("doc_name", ""),
                "doc_title": doc.get("doc_title", ""),
                "parser_type": doc.get("parser_type", ""),
            })

        return {
            "success": True,
            "doc_count": len(documents),
            "documents": documents,
        }

    def _build_list_documents_manifest(self) -> ToolManifest:
        return ToolManifest(
            name="list_documents",
            version="1.0.0",
            description="列出检索服务中的文档清单",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回文档数",
                        "default": 20,
                    },
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "doc_count": {"type": "integer"},
                    "documents": {"type": "array"},
                    "error": {"type": "string"},
                },
            },
            capabilities=["read_only"],
            permission_level="public",
            is_read_only=True,
            timeout_ms=5000,
            idempotent=True,
            cost_level="low",
            result_trust_level="verified",
        )

    # ============================================================
    # HTTP 工具方法（与 retrieval_client.py 风格一致）
    # ============================================================
    def _http_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """发起 GET 请求，返回解析后的 JSON"""
        url = f"{self._base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = Request(url, method="GET")
        req.add_header("Accept", "application/json")
        return self._do_request(req)

    def _http_post(
        self,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """发起 POST 请求，返回解析后的 JSON"""
        url = f"{self._base_url}{path}"
        data = json.dumps(body or {}).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        return self._do_request(req)

    def _do_request(self, req: Request) -> Any:
        """执行请求并解析响应，失败时抛出 _RetrievalHttpError"""
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as e:
            raise _RetrievalHttpError(
                f"检索服务返回错误: HTTP {e.code}"
            ) from e
        except URLError as e:
            raise _RetrievalHttpError(
                f"检索服务连接失败: {e}"
            ) from e
        except _RetrievalHttpError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _RetrievalHttpError(f"检索请求异常: {e}") from e


# ============================================================
# 辅助类型与函数
# ============================================================
class _RetrievalHttpError(Exception):
    """检索服务 HTTP 调用错误（统一封装，便于 handler 捕获）"""


def _coerce_int(value: Any, default: int) -> int:
    """安全地将值转为 int，失败时返回默认值"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["RetrievalToolFactory"]
