"""
Retrieval HTTP API — FastAPI server
启动: python -m retrieval_service.server

行业分区模式（固定）：数据在 银行业/ 保险业/ 其他/ 三个子目录下，各自独立建库检索。
（已移除单库模式，每次启动均为三分库）
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, Query
from pydantic import BaseModel

from .industry_partition import IndustryPartitionAPI
from .retrieval_request import RetrievalRequest, RetrievalStrategy

# ── 加载 .env 环境变量 ──
try:
    from dotenv import load_dotenv
    # 从项目根目录加载 .env（server.py 在 knowledge_platform/retrieval/retrieval_service/ 下，向上 3 层）
    from pathlib import Path
    _env_path = Path(__file__).resolve().parents[3] / ".env"
    if _env_path.exists():
        load_dotenv(str(_env_path))
except ImportError:
    pass


def _build_retrieval_api() -> IndustryPartitionAPI:
    """构建行业分区检索 API（固定三分库：银行业/保险业/其他，不支持单库）"""
    use_embed_api = os.environ.get("USE_EMBED_API", "").lower() in ("true", "1", "yes")
    embed_api_key = os.environ.get("SILICONFLOW_EMBED_API_KEY") or None
    embed_api_model = os.environ.get("SILICONFLOW_EMBED_MODEL") or None

    use_reranker = os.environ.get("USE_RERANKER", "").lower() in ("true", "1", "yes")
    reranker_api_key = os.environ.get("SILICONFLOW_RERANK_API_KEY") or None
    reranker_model = os.environ.get("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    return IndustryPartitionAPI(
        base_dir="regulatory_docs",
        use_embed_api=use_embed_api,
        embed_api_key=embed_api_key,
        embed_api_model=embed_api_model,
        use_reranker=use_reranker,
        reranker_api_key=reranker_api_key,
        reranker_model=reranker_model,
    )


# ── 启动时加载 ──
_api: Optional[IndustryPartitionAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _api
    _api = _build_retrieval_api()
    _api.load()
    yield


app = FastAPI(title="Retrieval API", version="1.0", lifespan=lifespan)


# ── 健康检查 ──
@app.get("/health")
def health_check():
    """健康检查（行业分区模式）"""
    return {
        "status": "ok" if _api and _api.is_loaded else "loading",
        "service": "retrieval-api",
        "version": "1.0",
        "mode": "industry_partition",
        "docs": _api.doc_count if _api else 0,
        "chunks": _api.chunk_count if _api else 0,
        "partitions": _api.stats if _api else {},
    }


# ── 请求体模型 ──
class SearchRequest(BaseModel):
    query: str
    strategy: str = "hybrid"
    top_k: int = 10
    filters: dict = {}
    expand_context: bool = False
    target_industries: Optional[List[str]] = None
    """行业分区过滤: ["银行业"] | ["银行业", "保险业"] | None=自动推断"""


class ChunkSearchRequest(BaseModel):
    doc_id: Optional[str] = None
    chunk_type: Optional[str] = None
    table_name: Optional[str] = None
    clause_number: Optional[str] = None
    chapter_number: Optional[str] = None
    limit: int = 20


# ── 接口一：统一 Retrieval API ──
@app.post("/api/v1/search")
def search(req: SearchRequest):
    """统一检索入口，返回 RetrievalHit"""
    r = RetrievalRequest(
        query=req.query,
        strategy=RetrievalStrategy(req.strategy),
        top_k=req.top_k,
        filters=req.filters,
        bm25_k=20,
        vector_k=20,
        expand_context=req.expand_context,
        target_industries=req.target_industries,
    )
    hits = _api.search_request(r)
    return [h.to_dict() for h in hits]


# ── 接口二：文档与 chunk 查询 API ──
@app.get("/api/v1/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    return _api.get_chunk(chunk_id) or {}


@app.get("/api/v1/documents/{doc_id}")
def get_document(doc_id: str):
    return _api.get_document(doc_id) or {}


@app.get("/api/v1/documents")
def list_documents(limit: int = Query(default=20, ge=1, le=200)):
    docs = _api.list_documents()
    return docs[:limit]


@app.post("/api/v1/chunks/search")
def search_chunks(req: ChunkSearchRequest):
    filters = {k: v for k, v in req.model_dump().items() if v is not None and k != "limit"}
    results = _api.search_chunks(**filters, limit=req.limit)
    return results


# ── 直接启动 ──
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)