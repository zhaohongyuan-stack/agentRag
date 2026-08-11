"""
检索测试服务 — 轻量 FastAPI，与 server.py 接口兼容

启动:
    python test_retrieval.py
    或: uvicorn test_retrieval:app --port 8000

其他模块调用:
    from test_retrieval import app
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval_service.retrieval_api import RetrievalAPI
from retrieval_service.retrieval_request import RetrievalRequest, RetrievalStrategy


class SearchRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    strategy: str = Field(default="hybrid", description="bm25|dense|hybrid|exact|metadata|relation|table")
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict = Field(default_factory=dict)
    expand_context: bool = False


_api: Optional[RetrievalAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _api
    data_dir = Path(__file__).resolve().parent / "regulatory_docs"
    _api = RetrievalAPI()
    _api.load(str(data_dir), lightweight=True)
    yield


app = FastAPI(title="Retrieval Test API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if _api and _api.is_loaded else "loading",
        "docs": _api.doc_count if _api else 0,
        "chunks": _api.chunk_count if _api else 0,
    }


@app.post("/api/v1/search")
def search(req: SearchRequest):
    r = RetrievalRequest(
        query=req.query,
        strategy=RetrievalStrategy(req.strategy),
        top_k=req.top_k,
        filters=req.filters,
        expand_context=req.expand_context,
    )
    return [h.to_dict() for h in _api.search_request(r)]


@app.get("/api/v1/search")
def search_get(query: str, strategy: str = "hybrid", top_k: int = 10):
    """GET 版检索，参数放 URL 避免编码问题"""
    r = RetrievalRequest(
        query=query,
        strategy=RetrievalStrategy(strategy),
        top_k=top_k,
    )
    return [h.to_dict() for h in _api.search_request(r)]


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent)

    if len(sys.argv) > 1:
        # 命令行模式: python test_retrieval.py "原保险保费收入" --strategy table --topk 3
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("query", nargs="?", default="")
        p.add_argument("--strategy", "-s", default="hybrid")
        p.add_argument("--topk", "-k", type=int, default=5)
        args = p.parse_args()

        data_dir = Path(__file__).resolve().parent / "regulatory_docs"
        api = RetrievalAPI()
        api.load(str(data_dir), lightweight=True)

        if args.query:
            req = RetrievalRequest(query=args.query, strategy=RetrievalStrategy(args.strategy), top_k=args.topk)
            for h in api.search_request(req):
                d = h.to_dict()
                print(f"\n[{d['rank']}] {d['chunk_type']} | {d['doc_name']} | score={d['score']:.4f}")
                print(f"  {d['content'][:300]}")
        else:
            # 交互模式
            print("输入查询文本，Ctrl+C 退出\n")
            api = api  # noqa
            while True:
                try:
                    q = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    continue
                req = RetrievalRequest(query=q, strategy=RetrievalStrategy("hybrid"), top_k=5)
                for h in api.search_request(req):
                    d = h.to_dict()
                    print(f"  [{d['rank']}] {d['chunk_type']} | score={d['score']:.4f} | {d['content'][:200]}")
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8001)
