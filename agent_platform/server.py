"""
Agent 平台 FastAPI 服务 — HTTP API 入口

启动方式:
    python -m agent_platform.server
    或设置环境变量 AGENT_PORT=8000 python -m agent_platform.server

默认端口: 8000（通过环境变量 AGENT_PORT 配置）

API 接口:
    POST /api/v1/query                      — 用户查询入口
    GET  /health                            — 健康检查
    GET  /api/v1/sessions/{session_id}      — 查询会话状态
    GET  /api/v1/sessions                   — 列出活跃会话
    POST /api/v1/auth/login                 — 登录
    GET  /api/v1/auth/me                    — 获取当前用户信息
    POST /api/v1/conversations              — 保存对话
    GET  /api/v1/conversations              — 列出对话
    GET  /api/v1/conversations/{id}         — 查看对话详情
    DELETE /api/v1/conversations/{id}       — 删除对话
    POST /api/v1/conversations/{id}/review  — 提交审查
    GET  /api/v1/conversations/{id}/review  — 查看审查
    POST /api/v1/conversations/{id}/suggestions  — 添加建议
    GET  /api/v1/conversations/{id}/suggestions  — 查看建议
    DELETE /api/v1/suggestions/{id}         — 删除建议
    POST /api/v1/conversations/{id}/unanswered  — 标记未回答
    GET  /api/v1/admin/conversations        — 管理员查看所有对话
    GET  /api/v1/admin/users                — 管理员查看用户统计
    GET  /api/v1/admin/reviews              — 管理员查看审查
    GET  /api/v1/admin/suggestions          — 管理员查看建议
    POST /api/v1/admin/conversations/{id}/qa  — 管理员质检标注
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import shutil
import glob as _glob
import base64

# ── 加载 .env 环境变量 ──
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(str(_env_path))
except ImportError:
    pass

from .gateway.auth.jwt_handler import create_access_token, verify_token
from .gateway.auth.user_store import get_user_by_id, verify_user
from .gateway.conversation_store import (
    add_suggestion,
    admin_delete_conversation,
    admin_get_conversations,
    admin_get_reviews,
    admin_get_suggestions,
    admin_get_users,
    admin_save_qa,
    cleanup_unreviewed_conversations,
    delete_conversation,
    delete_suggestion,
    get_admin_stats,
    get_conversation,
    get_conversations,
    get_review,
    get_suggestions,
    get_unanswered,
    get_user_marked_conversations,
    save_conversation,
    save_review,
    toggle_unanswered,
)
from .gateway.request_handler import (
    ConversationCreate,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    QARequest,
    QueryRequest,
    QueryResponse,
    RequestHandler,
    ReviewCreate,
    SuggestionCreate,
)
from .gateway.request_handler.event_callback import SSEEventCallback
from .gateway.request_handler.retrieval_client import RetrievalClient
from .gateway.request_handler.trace_enhancer import enhance_trace
from .gateway.session_handler.session_state import SessionManager
from .file_service import _register_router as _register_file_router
logger = logging.getLogger(__name__)

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

# 注册文件服务路由（真实业务文件浏览/预览/下载）
_register_file_router(app)


def _build_request_handler() -> RequestHandler:
    """
    构建 RequestHandler，根据环境变量决定检索模式:
      - RETRIEVAL_SERVICE_URL 设置时 → HTTP 模式（连接真实检索服务）
      - 未设置时 → In-Process Mock 模式（用于测试）
    """
    retrieval_url = os.environ.get("RETRIEVAL_SERVICE_URL", "")

    if retrieval_url:
        retrieval_client = RetrievalClient(
            base_url=retrieval_url,
            timeout_ms=int(os.environ.get("RETRIEVAL_TIMEOUT_MS", "10000")),
            in_process=False,
        )
        logger.info(f"[Agent] 检索模式: HTTP → {retrieval_url}")
        print(f"[Agent] 检索模式: HTTP → {retrieval_url}")
    else:
        retrieval_client = RetrievalClient(in_process=True)
        logger.info("[Agent] 检索模式: In-Process Mock")
        print("[Agent] 检索模式: In-Process Mock")

    return RequestHandler(retrieval_client=retrieval_client)


# 全局实例
_handler = _build_request_handler()
_session_manager = _handler._session_manager

# ============================================================
# 后台定时清理：每天0点删除无审查记录的对话
# ============================================================
import threading

def _cleanup_worker():
    """每天0点清理无审查记录的对话"""
    import time as _time
    from datetime import datetime, timedelta
    while True:
        now = datetime.now()
        # 计算下一个0点
        tomorrow = now + timedelta(days=1)
        next_midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        sleep_seconds = (next_midnight - now).total_seconds()
        logger.info("[Cleanup] 下次清理时间: %s (等待 %.0f 秒)", next_midnight, sleep_seconds)
        _time.sleep(sleep_seconds)
        try:
            deleted = cleanup_unreviewed_conversations()
            logger.info("[Cleanup] 清理完成，删除 %d 条对话", deleted)
        except Exception as e:
            logger.error("[Cleanup] 清理失败: %s", e, exc_info=True)

# 启动清理线程（守护线程）
_cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True, name="cleanup-worker")
_cleanup_thread.start()


# ============================================================
# JWT 认证依赖
# ============================================================
_security_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security_scheme),
) -> Optional[Dict[str, Any]]:
    """
    可选认证依赖（游客可跳过）

    从 Bearer token 中解析当前用户信息，无效或缺失时返回 None。
    """
    if credentials is None or not credentials.credentials:
        return None

    payload = verify_token(credentials.credentials)
    if payload is None:
        return None

    return {
        "user_id": payload.get("user_id"),
        "username": payload.get("username"),
        "role": payload.get("role"),
    }


def require_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    必需认证依赖（必须登录）

    未登录或 token 无效时返回 401。
    """
    if current_user is None:
        raise HTTPException(
            status_code=401,
            detail="未登录或登录已过期，请重新登录",
        )
    return current_user


def require_admin(
    current_user: Dict[str, Any] = Depends(require_user),
) -> Dict[str, Any]:
    """
    管理员认证依赖（必须管理员）

    非管理员返回 403。
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="权限不足，需要管理员权限",
        )
    return current_user


def _enhance_query_response(response: QueryResponse) -> QueryResponse:
    """为 QueryResponse 附加状态轨迹增强详情"""
    try:
        # 从会话中提取事件（StateEvent 对象需转为 dict）
        events = []
        session = _session_manager.get_session(response.session_id)
        if session is not None and hasattr(session, "state_machine"):
            raw_events = session.state_machine.events or []
            for ev in raw_events:
                if hasattr(ev, "to_dict"):
                    events.append(ev.to_dict())
                elif isinstance(ev, dict):
                    events.append(ev)
        response.state_trace_detail = enhance_trace(response.state_trace, events)
    except Exception as e:
        logger.warning("[Query] 状态轨迹增强失败: %s", e)
    return response


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
    # 附加状态轨迹增强详情
    response = _enhance_query_response(response)
    return response


# ============================================================
# 接口一B: GET /api/v1/query/stream — SSE 流式查询
# ============================================================
@app.get("/api/v1/query/stream")
async def query_stream(
    query: str = Query(..., description="用户问题"),
    session_id: str = Query(None, description="会话 ID"),
):
    """
    SSE 流式查询入口

    通过 Server-Sent Events 实时推送 Agent 执行过程事件。
    事件格式: data: {"type": "agent:start", "agent": "Planner", ...}\n\n

    事件类型:
      agent:start      — Agent 开始执行
      agent:thinking   — Agent 推理过程
      agent:result     — Agent 输出决策
      tool:call        — Agent 调用工具
      tool:result      — 工具返回结果
      loop:round       — 检索-评估 Loop 轮次
      answer:token     — 逐 token 输出回答
      error            — 执行异常
      done             — 流程结束，包含完整响应
    """
    queue: asyncio.Queue = asyncio.Queue()
    # 捕获当前事件循环，供 worker 线程通过 call_soon_threadsafe 安全推送事件
    loop = asyncio.get_running_loop()
    callback = SSEEventCallback(queue, loop=loop)

    # 构建请求对象
    req = QueryRequest(query=query, session_id=session_id)

    async def _run_handler():
        """在线程池中执行同步 handler，不影响事件循环"""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, _handler.handle_query, req, callback
            )
            # 确保 done 事件已发送（handler 内部已发送，这里做兜底）
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"SSE handler 异常: {e}", exc_info=True)
            callback.on_error("System", str(e))
            callback.on_done({"error": str(e)})

    async def event_generator():
        """生成 SSE 事件流"""
        # 启动 handler 任务（不等待完成）
        handler_task = asyncio.create_task(_run_handler())

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] == "done":
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': '请求超时'}}, ensure_ascii=False)}\n\n"
                break

        # 等待 handler 完成（清理资源）
        await handler_task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 接口二: GET /health — 健康检查
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查（含知识库文档数和分块数）"""
    docs = None
    chunks = None
    try:
        retrieval_url = os.environ.get("RETRIEVAL_SERVICE_URL", "")
        if retrieval_url:
            from urllib.request import urlopen
            import json as _json
            with urlopen(f"{retrieval_url.rstrip('/')}/health", timeout=3) as resp:
                data = _json.loads(resp.read().decode())
                docs = data.get("docs")
                chunks = data.get("chunks")
    except Exception as e:
        logger.debug("[Health] 获取检索服务统计失败: %s", e)
    return HealthResponse(
        status="ok",
        service="agent-platform",
        version="1.0.0-phase1",
        docs=docs,
        chunks=chunks,
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
# 认证接口
# ============================================================
# ============================================================
# 接口五: POST /api/v1/auth/login — 登录
# ============================================================
@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """
    用户登录

    验证用户名密码，成功后返回 JWT token 和用户信息。
    """
    user = verify_user(request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
        )

    token = create_access_token(
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
    )

    user_info = {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "display_name": user["display_name"],
    }

    logger.info("[Auth] 用户登录成功: %s (role=%s)", user["username"], user["role"])
    return LoginResponse(token=token, user=user_info)


# ============================================================
# 接口六: GET /api/v1/auth/me — 获取当前用户信息
# ============================================================
@app.get("/api/v1/auth/me")
def get_me(current_user: Dict[str, Any] = Depends(require_user)):
    """获取当前登录用户信息"""
    user = get_user_by_id(current_user["user_id"])
    if user is None:
        raise HTTPException(status_code=404, detail="用户信息不存在")
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }


# ============================================================
# 对话持久化接口
# ============================================================
# ============================================================
# 接口七: POST /api/v1/conversations — 保存对话
# ============================================================
@app.post("/api/v1/conversations")
def create_conversation(
    body: ConversationCreate,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """保存对话记录"""
    conversation_id = save_conversation(
        user_id=current_user["user_id"],
        session_id=body.session_id,
        query=body.query,
        response_dict=body.response,
    )
    return {"id": conversation_id, "message": "对话保存成功"}


# ============================================================
# 接口八: GET /api/v1/conversations — 列出对话
# ============================================================
@app.get("/api/v1/conversations")
def list_conversations(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    start_date: Optional[str] = Query(None, description="起始日期（ISO）"),
    end_date: Optional[str] = Query(None, description="结束日期（ISO）"),
    current_user: Dict[str, Any] = Depends(require_user),
):
    """分页查询当前用户的对话列表"""
    return get_conversations(
        user_id=current_user["user_id"],
        page=page,
        page_size=page_size,
        start_date=start_date,
        end_date=end_date,
    )


# ============================================================
# 接口九: GET /api/v1/conversations/{id} — 查看对话详情
# ============================================================
@app.get("/api/v1/conversations/{conversation_id}")
def get_conversation_detail(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """查看对话详情"""
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    # 普通用户只能查看自己的对话，管理员可查看所有
    if conv["user_id"] != current_user["user_id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="无权查看此对话")
    return conv


# ============================================================
# 接口十: DELETE /api/v1/conversations/{id} — 删除对话
# ============================================================
@app.delete("/api/v1/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """删除对话（仅所有者可删除）"""
    success = delete_conversation(conversation_id, current_user["user_id"])
    if not success:
        raise HTTPException(
            status_code=404,
            detail="对话不存在或无权删除",
        )
    return {"message": "对话已删除"}


# ============================================================
# 接口十半: GET /api/v1/my/marked — 用户工作台（我的标记）
# ============================================================
@app.get("/api/v1/my/marked")
def my_marked_conversations(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    current_user: Dict[str, Any] = Depends(require_user),
):
    """获取当前用户标记过的对话（有审查记录或管理员质检反馈的）"""
    return get_user_marked_conversations(
        user_id=current_user["user_id"],
        page=page,
        page_size=page_size,
    )


# ============================================================
# 接口十一: POST /api/v1/conversations/{id}/review — 提交审查
# ============================================================
@app.post("/api/v1/conversations/{conversation_id}/review")
def create_review(
    conversation_id: str,
    body: ReviewCreate,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """提交对话审查（审查评论最少 10 字符）"""
    # 校验对话存在
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 校验审查状态
    valid_statuses = {"correct", "incorrect", "needs_improvement"}
    if body.review_status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"无效的审查状态，可选值: {', '.join(valid_statuses)}",
        )

    review_id = save_review(
        conversation_id=conversation_id,
        user_id=current_user["user_id"],
        review_status=body.review_status,
        review_comment=body.review_comment,
    )
    return {"id": review_id, "message": "审查已提交"}


# ============================================================
# 接口十二: GET /api/v1/conversations/{id}/review — 查看审查
# ============================================================
@app.get("/api/v1/conversations/{conversation_id}/review")
def get_conversation_review(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """查看对话的审查记录"""
    review = get_review(conversation_id)
    if review is None:
        raise HTTPException(status_code=404, detail="暂无审查记录")
    return review


# ============================================================
# 接口十三: POST /api/v1/conversations/{id}/suggestions — 添加建议
# ============================================================
@app.post("/api/v1/conversations/{conversation_id}/suggestions")
def create_suggestion(
    conversation_id: str,
    body: SuggestionCreate,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """添加改进建议"""
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    suggestion_id = add_suggestion(
        conversation_id=conversation_id,
        user_id=current_user["user_id"],
        suggestion_text=body.suggestion_text,
    )
    return {"id": suggestion_id, "message": "建议已添加"}


# ============================================================
# 接口十四: GET /api/v1/conversations/{id}/suggestions — 查看建议
# ============================================================
@app.get("/api/v1/conversations/{conversation_id}/suggestions")
def get_conversation_suggestions(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """查看对话的所有改进建议"""
    return get_suggestions(conversation_id)


# ============================================================
# 接口十五: DELETE /api/v1/suggestions/{id} — 删除建议
# ============================================================
@app.delete("/api/v1/suggestions/{suggestion_id}")
def remove_suggestion(
    suggestion_id: str,
    current_user: Dict[str, Any] = Depends(require_user),
):
    """删除改进建议（仅建议所有者可删除）"""
    success = delete_suggestion(suggestion_id, current_user["user_id"])
    if not success:
        raise HTTPException(
            status_code=404,
            detail="建议不存在或无权删除",
        )
    return {"message": "建议已删除"}


# ============================================================
# 接口十六: POST /api/v1/conversations/{id}/unanswered — 标记未回答
# ============================================================
@app.post("/api/v1/conversations/{conversation_id}/unanswered")
def mark_unanswered(
    conversation_id: str,
    body: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(require_user),
):
    """标记/取消标记对话为未回答（JSON body: {"is_marked": true/false}）"""
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    is_marked = body.get("is_marked", True)
    toggle_unanswered(
        conversation_id=conversation_id,
        user_id=current_user["user_id"],
        is_marked=is_marked,
    )
    return {"message": "标记已更新", "is_marked": is_marked}


# ============================================================
# 管理员接口
# ============================================================
# ============================================================
# 接口十七: GET /api/v1/admin/conversations — 管理员查看所有对话
# ============================================================
@app.get("/api/v1/admin/conversations")
def admin_list_conversations(
    user_id: Optional[str] = Query(None, description="筛选用户 ID"),
    review_status: Optional[str] = Query(None, description="筛选审查状态"),
    is_unanswered: Optional[bool] = Query(None, description="筛选未回答"),
    has_suggestion: Optional[bool] = Query(None, description="筛选有建议"),
    start_date: Optional[str] = Query(None, description="起始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    intent: Optional[str] = Query(None, description="意图筛选"),
    is_refusal: Optional[bool] = Query(None, description="是否拒答"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    admin: Dict[str, Any] = Depends(require_admin),
):
    """管理员查看所有对话（支持多维筛选，附带用户名）"""
    result = admin_get_conversations(
        user_id=user_id,
        review_status=review_status,
        is_unanswered=is_unanswered,
        has_suggestion=has_suggestion,
        start_date=start_date,
        end_date=end_date,
        intent=intent,
        is_refusal=is_refusal,
        page=page,
        page_size=page_size,
    )
    # 在接口层补充用户名信息（确保即使 conversation_store 未更新也能工作）
    try:
        from .gateway.auth.user_store import list_users
        user_map = {u["id"]: u for u in list_users()}
        for item in result.get("items", []):
            uid = item.get("user_id", "")
            uinfo = user_map.get(uid, {})
            if "username" not in item:
                item["username"] = uinfo.get("username", "未知")
            if "display_name" not in item:
                item["display_name"] = uinfo.get("display_name", "未知")
    except Exception as e:
        logger.warning("[Admin] 补充用户名失败: %s", e)
    return result


# ============================================================
# 接口十八: GET /api/v1/admin/users — 管理员查看用户统计
# ============================================================
@app.get("/api/v1/admin/users")
def admin_list_users(admin: Dict[str, Any] = Depends(require_admin)):
    """管理员查看用户统计信息"""
    return admin_get_users()


# ============================================================
# 接口十九: GET /api/v1/admin/reviews — 管理员查看审查
# ============================================================
@app.get("/api/v1/admin/reviews")
def admin_list_reviews(
    review_status: Optional[str] = Query(None, description="筛选审查状态"),
    user_id: Optional[str] = Query(None, description="筛选用户 ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    admin: Dict[str, Any] = Depends(require_admin),
):
    """管理员查看审查记录"""
    return admin_get_reviews(
        review_status=review_status,
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# ============================================================
# 接口十九半: GET /api/v1/admin/stats — 质检中心统计数据
# ============================================================
@app.get("/api/v1/admin/stats")
def admin_stats(admin: Dict[str, Any] = Depends(require_admin)):
    """获取质检中心准确统计数据"""
    return get_admin_stats()


# ============================================================
# 接口二十: GET /api/v1/admin/suggestions — 管理员查看建议
# ============================================================
@app.get("/api/v1/admin/suggestions")
def admin_list_suggestions(
    user_id: Optional[str] = Query(None, description="筛选用户 ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    admin: Dict[str, Any] = Depends(require_admin),
):
    """管理员查看改进建议"""
    return admin_get_suggestions(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# ============================================================
# 接口二十一: POST /api/v1/admin/conversations/{id}/qa — 管理员质检标注
# ============================================================
@app.post("/api/v1/admin/conversations/{conversation_id}/qa")
def admin_create_qa(
    conversation_id: str,
    body: QARequest,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """管理员质检标注（pass / fail / rework）"""
    conv = get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    valid_results = {"pass", "fail", "rework"}
    if body.qa_result not in valid_results:
        raise HTTPException(
            status_code=422,
            detail=f"无效的质检结果，可选值: {', '.join(valid_results)}",
        )

    qa_id = admin_save_qa(
        conversation_id=conversation_id,
        admin_id=admin["user_id"],
        qa_result=body.qa_result,
        qa_comment=body.qa_comment,
    )
    return {"id": qa_id, "message": "质检标注已保存"}


# ============================================================
# 接口二十一半: DELETE /api/v1/admin/conversations/{conversation_id} — 管理员删除任意对话
# ============================================================
@app.delete("/api/v1/admin/conversations/{conversation_id}")
def admin_delete_conv(
    conversation_id: str,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """管理员删除任意对话（不验证所有权）"""
    success = admin_delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在")
    logger.info("[Admin] 管理员删除对话: conv=%s, admin=%s", conversation_id, admin.get("username"))
    return {"message": "对话已删除", "conversation_id": conversation_id}


# ============================================================
# 知识库管理接口
# ============================================================
_KB_DIR = os.environ.get(
    "KB_DIR",
    "/app/knowledge_platform/retrieval/regulatory_docs",
)


def _resolve_kb_dir() -> str:
    """解析知识库目录路径，容器路径不可用时回退到本地"""
    kb_dir = _KB_DIR
    if not os.path.isdir(kb_dir):
        # 回退到本地项目路径
        local_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "knowledge_platform", "retrieval", "regulatory_docs",
        )
        if os.path.isdir(local_dir):
            kb_dir = local_dir
        else:
            os.makedirs(kb_dir, exist_ok=True)
    return kb_dir


# ============================================================
# 接口二十二: GET /api/v1/admin/kb/files — 列出知识库文件
# ============================================================
@app.get("/api/v1/admin/kb/files")
def admin_list_kb_files(admin: Dict[str, Any] = Depends(require_admin)):
    """列出知识库目录下的所有 JSONL 文件"""
    kb_dir = _resolve_kb_dir()
    files = []
    try:
        for f in sorted(os.listdir(kb_dir)):
            if f.endswith(".jsonl") or f.endswith(".json"):
                fpath = os.path.join(kb_dir, f)
                if os.path.isfile(fpath):
                    stat = os.stat(fpath)
                    files.append({
                        "name": f,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
    except OSError as e:
        logger.warning("[KB] 列出文件失败: %s", e)

    # 获取文档/分块统计
    docs = None
    chunks = None
    try:
        retrieval_url = os.environ.get("RETRIEVAL_SERVICE_URL", "")
        if retrieval_url:
            from urllib.request import urlopen
            import json as _json
            with urlopen(f"{retrieval_url.rstrip('/')}/health", timeout=3) as resp:
                data = _json.loads(resp.read().decode())
                docs = data.get("docs")
                chunks = data.get("chunks")
    except Exception:
        pass

    return {
        "files": files,
        "kb_dir": kb_dir,
        "doc_count": docs,
        "chunk_count": chunks,
    }


# ============================================================
# 接口二十三: POST /api/v1/admin/kb/upload — 上传 JSONL 文件
# ============================================================
class KBUploadRequest(BaseModel):
    filename: str
    content: str  # base64-encoded file content


@app.post("/api/v1/admin/kb/upload")
def admin_upload_kb_file(
    body: KBUploadRequest,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """上传 JSONL 文件到知识库目录（base64 编码方式，无需 python-multipart）"""
    filename = body.filename
    if not (filename.endswith(".jsonl") or filename.endswith(".json")):
        raise HTTPException(status_code=422, detail="仅支持 .jsonl 或 .json 格式文件")

    # 安全检查：文件名不能包含路径分隔符
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="文件名包含非法字符")

    kb_dir = _resolve_kb_dir()
    file_path = os.path.join(kb_dir, filename)

    # 安全检查：防止路径穿越
    if not os.path.abspath(file_path).startswith(os.path.abspath(kb_dir)):
        raise HTTPException(status_code=400, detail="非法文件路径")

    try:
        file_data = base64.b64decode(body.content)
        with open(file_path, "wb") as f:
            f.write(file_data)
        logger.info("[KB] 文件上传成功: %s (%d bytes, admin=%s)",
                     filename, len(file_data), admin.get("username"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    return {
        "message": f"文件 {filename} 上传成功",
        "filename": filename,
        "size": len(file_data),
        "hint": "请点击「重启嵌入服务」使新文件生效",
    }


# ============================================================
# 接口二十四: POST /api/v1/admin/kb/rebuild — 触发索引重建
# ============================================================
@app.post("/api/v1/admin/kb/rebuild")
def admin_rebuild_kb(admin: Dict[str, Any] = Depends(require_admin)):
    """触发知识库索引重建（通过调用检索服务的重新加载接口）"""
    retrieval_url = os.environ.get("RETRIEVAL_SERVICE_URL", "")
    if not retrieval_url:
        raise HTTPException(status_code=500, detail="检索服务 URL 未配置")

    try:
        from urllib.request import urlopen, Request
        import json as _json

        # 尝试调用检索服务的 reload 接口
        req = Request(
            f"{retrieval_url.rstrip('/')}/api/v1/reload",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=_json.dumps({"force": True}).encode(),
        )
        with urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())

        logger.info("[KB] 索引重建触发成功 (admin=%s)", admin.get("username"))
        return {
            "message": "索引重建已触发",
            "result": data,
        }
    except Exception as e:
        logger.warning("[KB] 索引重建触发失败: %s", e)
        # 如果 reload 接口不存在，返回提示
        return {
            "message": "索引重建触发失败，请手动重启检索服务容器",
            "error": str(e),
            "hint": "可在服务器执行: docker restart ace-rag-retrieval",
        }


# ============================================================
# 接口二十五: DELETE /api/v1/admin/kb/files/{filename} — 删除知识库文件
# ============================================================
@app.delete("/api/v1/admin/kb/files/{filename}")
def admin_delete_kb_file(
    filename: str,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """删除知识库目录下的指定文件"""
    if not (filename.endswith(".jsonl") or filename.endswith(".json")):
        raise HTTPException(status_code=422, detail="仅支持删除 .jsonl 或 .json 文件")

    kb_dir = _resolve_kb_dir()
    file_path = os.path.join(kb_dir, filename)

    # 安全检查
    if not os.path.abspath(file_path).startswith(os.path.abspath(kb_dir)):
        raise HTTPException(status_code=400, detail="非法文件路径")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        os.remove(file_path)
        logger.info("[KB] 文件删除成功: %s (admin=%s)", filename, admin.get("username"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件删除失败: {e}")

    return {"message": f"文件 {filename} 已删除"}


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
