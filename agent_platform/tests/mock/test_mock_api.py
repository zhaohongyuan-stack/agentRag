"""
Mock API 单元测试 — 使用 pytest + FastAPI TestClient

测试 Mock Retrieval API 的5个接口：
  1. test_normal_search:      正常检索返回结果
  2. test_empty_result:        空结果场景返回 []
  3. test_timeout:             超时场景返回 504
  4. test_get_chunk:           按ID获取chunk
  5. test_list_documents:      列出文档
  6. test_search_chunks:       多字段组合查询
  7. test_get_chunk_not_found: chunk不存在时返回空对象
  8. test_version_conflict:    版本冲突场景返回带冲突标记的结果
  9. test_partial_failure:     部分失败场景返回带失败标记的结果

测试不需要启动真实服务，使用 FastAPI TestClient（基于 httpx）。

运行方式：
    pytest agent_platform/tests/mock/test_mock_api.py -v
    或
    python -m pytest agent_platform/tests/mock/test_mock_api.py -v
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中，以便导入 agent_platform 包
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from fastapi.testclient import TestClient

from agent_platform.tests.mock.mock_retrieval_api import (
    MOCK_CHUNKS,
    MOCK_DOCUMENTS,
    app,
)

# 创建 TestClient 实例（不需要启动真实服务）
client = TestClient(app)


# ============================================================
# 测试用例
# ============================================================

class TestSearchAPI:
    """POST /api/v1/search 接口测试"""

    def test_normal_search(self):
        """正常检索返回结果"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "核心一级资本充足率",
                "strategy": "hybrid",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # 验证返回的 hit 结构
        hit = data[0]
        assert "chunk_id" in hit
        assert "chunk_type" in hit
        assert "doc_id" in hit
        assert "content" in hit
        assert "score" in hit
        assert "rank" in hit
        assert "matched_by" in hit
        assert "metadata" in hit

    def test_normal_search_with_scenario_field(self):
        """通过显式 scenario 字段指定 normal 场景"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "商业银行资本管理办法",
                "scenario": "normal",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_empty_result(self):
        """空结果场景返回空数组"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "查不到的内容",
                "scenario": "empty",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_empty_result_by_query_match(self):
        """通过 query 模糊匹配触发空结果场景"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "空结果测试",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_timeout(self):
        """超时场景返回504状态码"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "超时测试查询",
                "scenario": "timeout",
            },
        )
        assert response.status_code == 504
        data = response.json()
        assert data["error"] == "timeout"
        assert "message" in data
        assert data["scenario"] == "timeout"

    def test_timeout_by_query_match(self):
        """通过 query 模糊匹配触发超时场景"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "这是一个timeout请求",
            },
        )
        assert response.status_code == 504

    def test_version_conflict(self):
        """版本冲突场景返回带冲突标记的结果"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "资本充足率最低要求",
                "scenario": "version_conflict",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # 验证版本冲突标记
        for hit in data:
            meta = hit.get("metadata", {})
            assert meta.get("version_conflict") is True
            assert "version_status" in meta

    def test_partial_failure(self):
        """部分失败场景返回带失败标记的结果"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "系统重要性银行附加资本",
                "scenario": "partial_failure",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # 验证部分结果标记了失败
        has_partial = False
        for hit in data:
            trace = hit.get("trace", {})
            if trace.get("retrieval_status") == "partial_failure":
                has_partial = True
                assert "failed_channels" in trace
        assert has_partial, "应至少有一条结果标记为 partial_failure"

    def test_search_with_filters(self):
        """带 filters 的检索请求"""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "资本充足率",
                "filters": {"chunk_type": "clause"},
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestGetChunkAPI:
    """GET /api/v1/chunks/{chunk_id} 接口测试"""

    def test_get_chunk(self):
        """按ID获取chunk"""
        # 使用内置的Mock chunk
        chunk_id = MOCK_CHUNKS[0]["chunk_id"]
        response = client.get(f"/api/v1/chunks/{chunk_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["chunk_id"] == chunk_id
        assert "chunk_type" in data
        assert "content" in data
        assert "metadata" in data

    def test_get_chunk_clause_type(self):
        """获取 clause 类型的chunk"""
        # 找一个 clause 类型的chunk
        clause_chunk = next(c for c in MOCK_CHUNKS if c["chunk_type"] == "clause")
        response = client.get(f"/api/v1/chunks/{clause_chunk['chunk_id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["chunk_type"] == "clause"

    def test_get_chunk_cell_fact_type(self):
        """获取 cell_fact 类型的chunk"""
        cell_chunk = next(c for c in MOCK_CHUNKS if c["chunk_type"] == "cell_fact")
        response = client.get(f"/api/v1/chunks/{cell_chunk['chunk_id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["chunk_type"] == "cell_fact"

    def test_get_chunk_not_found(self):
        """chunk不存在时返回空对象"""
        response = client.get("/api/v1/chunks/nonexistent_chunk_id")
        assert response.status_code == 200
        data = response.json()
        assert data == {}


class TestDocumentAPI:
    """GET /api/v1/documents 和 GET /api/v1/documents/{doc_id} 接口测试"""

    def test_list_documents(self):
        """列出文档"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # 验证文档结构
        doc = data[0]
        assert "doc_id" in doc
        assert "doc_name" in doc
        assert "parser_type" in doc

    def test_list_documents_with_limit(self):
        """列出文档带 limit 参数"""
        response = client.get("/api/v1/documents?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 1

    def test_get_document(self):
        """按ID获取文档"""
        doc_id = MOCK_DOCUMENTS[0]["doc_id"]
        response = client.get(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == doc_id
        assert "doc_name" in data
        assert "parser_type" in data

    def test_get_document_not_found(self):
        """文档不存在时返回空对象"""
        response = client.get("/api/v1/documents/nonexistent_doc_id")
        assert response.status_code == 200
        data = response.json()
        assert data == {}


class TestSearchChunksAPI:
    """POST /api/v1/chunks/search 接口测试"""

    def test_search_chunks(self):
        """多字段组合查询"""
        response = client.post(
            "/api/v1/chunks/search",
            json={
                "doc_id": "doc-400",
                "chunk_type": "clause",
                "limit": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # 验证所有结果都满足过滤条件
        for chunk in data:
            assert chunk["doc_id"] == "doc-400"
            assert chunk["chunk_type"] == "clause"

    def test_search_chunks_by_table_name(self):
        """按 table_name 查询 cell_fact"""
        response = client.post(
            "/api/v1/chunks/search",
            json={
                "chunk_type": "cell_fact",
                "table_name": "1. 银行业金融机构",
                "limit": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        for chunk in data:
            assert chunk["chunk_type"] == "cell_fact"
            assert chunk["table_name"] == "1. 银行业金融机构"

    def test_search_chunks_by_clause_number(self):
        """按 clause_number 查询"""
        response = client.post(
            "/api/v1/chunks/search",
            json={
                "clause_number": "第四十三条",
                "limit": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        for chunk in data:
            assert chunk["clause_number"] == "第四十三条"

    def test_search_chunks_no_match(self):
        """无匹配结果时返回空数组"""
        response = client.post(
            "/api/v1/chunks/search",
            json={
                "doc_id": "nonexistent",
                "limit": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_search_chunks_empty_body(self):
        """空请求体返回所有chunk（受limit限制）"""
        response = client.post(
            "/api/v1/chunks/search",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # 默认 limit=20，应返回全部内置 chunk
        assert len(data) == len(MOCK_CHUNKS)


class TestHealthCheck:
    """健康检查端点测试"""

    def test_health_check(self):
        """健康检查端点返回正常状态"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "mock-retrieval-api"
