"""
Agent 平台 FastAPI 服务 — HTTP API 入口

启动方式:
    python -m agent_platform.server
    或设置环境变量 AGENT_PORT=8000 python -m agent_platform.server

默认端口: 8000（通过环境变量 AGENT_PORT 配置）

API 接口:
    POST /api/v1/query   — 用户查询入口
    GET  /health         — 健康检查
    GET  /api/v1/sessions/{session_id} — 查询会话状态
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .gateway.request_handler import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RequestHandler,
)
from .gateway.session_handler import SessionManager

# ============================================================
# 初始化
# ============================================================
app = FastAPI(
    title="ACE-RAG Agent Platform",
    description="B组 Agent 执行平台 — 自适应编译式证据检索问答",
    version="1.0.0-phase1",
)

# CORS（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
_handler = RequestHandler()
_session_manager = _handler._session_manager


# ============================================================
# 接口一: POST /api/v1/query — 用户查询入口
# ============================================================
@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    用户查询入口

    接收用户问题，经过查询理解、路由、检索、证据组装、回答生成，
    返回带引用的回答。
    """
    response = _handler.handle_query(request)
    return response


# ============================================================
# 接口二: GET /health — 健康检查
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        service="agent-platform",
        version="1.0.0-phase1",
    )


# ============================================================
# 接口三: GET /api/v1/sessions/{session_id} — 查询会话状态
# ============================================================
@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str):
    """查询会话状态和历史"""
    session = _session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "last_active_at": session.last_active_at,
        "turn_count": session.turn_count,
        "current_state": session.state_machine.current_state.value if session.state_machine.current_state else None,
        "state_trace": session.state_machine.get_state_trace(),
        "history": session.history,
    }


# ============================================================
# 接口四: GET /api/v1/sessions — 列出活跃会话
# ============================================================
@app.get("/api/v1/sessions")
def list_sessions():
    """列出所有活跃会话"""
    return {
        "active_count": _session_manager.active_count,
        "session_ids": list(_session_manager._sessions.keys()),
    }


# ============================================================
# 直接启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AGENT_PORT", 8000))
    host = os.environ.get("AGENT_HOST", "0.0.0.0")
    print("=" * 60)
    print("  ACE-RAG Agent Platform 启动中...")
    print(f"  地址: http://{host}:{port}")
    print(f"  文档: http://{host}:{port}/docs")
    print(f"  健康检查: http://{host}:{port}/health")
    print(f"  查询接口: POST http://{host}:{port}/api/v1/query")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
