"""
Retrieval HTTP API — FastAPI server
启动: python -m retrieval_service.server
"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel

from .retrieval_api import RetrievalAPI
from .retrieval_request import RetrievalRequest, RetrievalStrategy

# ── 启动时加载 ──
_api: Optional[RetrievalAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _api
    _api = RetrievalAPI()
    _api.load("regulatory_docs/")
    yield


app = FastAPI(title="Retrieval API", version="1.0", lifespan=lifespan)


# ── 请求体模型 ──
class SearchRequest(BaseModel):
    query: str
    strategy: str = "hybrid"
    top_k: int = 10
    filters: dict = {}
    expand_context: bool = False


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
    uvicorn.run(app, host="127.0.0.1", port=8000)
