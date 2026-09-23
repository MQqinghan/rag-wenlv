"""

查询服务 HTTP 入口模块，承载查询接口与流式 SSE 推送。

"""

import asyncio

import sys

import uuid

from contextlib import asynccontextmanager

from mimetypes import guess_type

from pathlib import Path



# 兼容直接以 `python query_server.py` 方式启动，提前把项目根目录加入模块搜索路径

if __package__ in (None, ""):

    bootstrap_root = Path(__file__).resolve().parents[3]

    if str(bootstrap_root) not in sys.path:

        sys.path.insert(0, str(bootstrap_root))



from fastapi import BackgroundTasks, FastAPI, Request

from fastapi.responses import FileResponse, Response, StreamingResponse

from starlette.middleware.cors import CORSMiddleware



from app.shared.schemas.tourism_query import (

    HistoryItem,

    HistoryResponse,

    QueryRequest,

    QueryResponse,

    SessionListResponse,

    SessionSummary,

)

from app.shared.runtime.data_version_service import maybe_refresh_query_caches

from app.shared.runtime.loop_attribution import detect_implicit_feedback, record_implicit_feedback

from app.shared.runtime.logger import logger

# R12（2026-09-11）：静态页面/图片根目录必须锚定本文件所在工作空间。
# 不可用共享层 logger.PROJECT_ROOT——其 Path.resolve() 会展开 junction 指向 RAG_shared_infra（那里没有 app/web），拆分后 /html 等页面 500。
STATIC_ROOT = Path(__file__).resolve().parents[3]

from app.shared.runtime.warmup import warmup_all

import app.api.bootstrap  # noqa: F401  触发预热/缓存失效钩子注册

from app.infra.config import settings

from app.shared.config.role_config import ROLE_QUERY, startup_role_check

from app.infra.persistence.history_repository import history_repository

from app.process.unified_query.agent.main_graph import unified_query_app

from app.process.unified_query.agent.state import create_unified_default_state

from app.shared.utils.sse_utils import (

    SSEEvent,

    clear_cancel,

    create_sse_queue,

    push_to_session,

    request_cancel,

    sse_generator,

)

from app.shared.utils.task_utils import (

    TASK_STATUS_COMPLETED,

    TASK_STATUS_FAILED,

    TASK_STATUS_PROCESSING,

    clear_task,

    get_done_task_list,

    get_task_result,

    update_task_status,

)

from app.api.http.trip_plan_server import router as trip_plan_router





def validate_intent_route_keys() -> None:

    """

    P1-1（R3 防护）：启动期打印意图路由双键生效状态，误开实验版 B1 路由时主动告警。



    两个键仅差一个字母，极易配错：

    - INTENT_ROUTER_LLM_FIRST（多一个 R）：当前*生效*的 v2 路由（intent_route_service），默认 True。

    - INTENT_ROUTE_LLM_FIRST（少一个 R）：*未启用*的 B1 实验路由（intent_route_llm / node_intent_route），默认 False。

    """

    from app.shared.config.common import env_bool  # 延迟导入，避免模块级循环依赖



    v2 = env_bool("INTENT_ROUTER_LLM_FIRST", True)   # 生效 v2 路由

    b1 = env_bool("INTENT_ROUTE_LLM_FIRST", False)    # 实验 B1 路由



    logger.info(

        f"[启动校验] 意图路由双键状态：生效 v2 INTENT_ROUTER_LLM_FIRST={v2} | "

        f"实验 B1 INTENT_ROUTE_LLM_FIRST={b1}"

    )

    if b1:

        logger.warning(

            "[启动校验][R3] 检测到实验版 B1 路由 INTENT_ROUTE_LLM_FIRST=True，"

            "该键对应未启用的 node_intent_route（与生效 v2 仅差一个字母 R）。"

            "如非意在启用 B1 实验路由，请改回 False，避免路由行为偏离预期。"

        )





@asynccontextmanager

async def lifespan(_: FastAPI):

    """

    服务生命周期钩子：启动阶段同步预热本地模型。



    预热是重 CPU/GPU 的阻塞操作，丢到线程里执行以免占住事件循环；

    预热失败不影响服务启动，只退化为首次查询时懒加载。

    """

    validate_intent_route_keys()

    logger.info("查询服务启动中，开始预热本地模型...")

    await asyncio.to_thread(warmup_all, True)

    logger.info("查询服务预热完成，开始接收请求。")

    yield

    logger.info("查询服务关闭。")





app = FastAPI(

    title=settings.query_app_name,

    description="统一 RAG 查询服务（文旅/闲聊自动路由），负责问答、SSE 输出与历史记录查询。",

    version="0.3.1",

    lifespan=lifespan,

)

app.add_middleware(

    CORSMiddleware,

    allow_origins=list(settings.cors_origins) or ["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)



# 注册行程规划路由（第四路结构化 JSON 行程，与 /query 对话路由解耦）

app.include_router(trip_plan_router)



# A5 部署切片（T8）：启动期按角色自检配置，缺失前置到启动暴露（APP_ROLE 未设=本地开发，跳过）

ROLE_CHECK_RESULT = startup_role_check(ROLE_QUERY)





def new_session_id() -> str:

    """生成新的会话 ID"""

    return str(uuid.uuid4())





def invoke_query(session_id: str, query: str, is_stream: bool, user_id: str = "") -> dict:

    """

    调用查询主图并维护统一的任务状态。



    Args:

        session_id: 当前会话 ID。

        query: 用户原始问题。

        is_stream: 是否为流式模式。



    Returns:

        dict: 查询图执行后的最终状态。

    """

    # 0. 知识库数据版本检查（内部 5s 节流）：导入服务更新知识库后，

    #    在此经 Mongo 版本号比对刷新本进程的 BM25/LLM 两件套缓存

    maybe_refresh_query_caches()

    clear_task(session_id)

    clear_cancel(session_id)  # 新查询开始时清掉上一轮的停止标记

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    initial_state = create_unified_default_state(

        session_id=session_id,

        original_query=query,

        is_stream=is_stream,

        user_id=user_id or "",

    )

    state = unified_query_app.invoke(initial_state)

    update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

    return state





def run_stream_query_background(session_id: str, query: str, user_id: str = "") -> None:

    """

    在后台执行一次流式查询任务。

    节点内部通过 push_to_session 推送 delta 增量，

    全流程结束后由本函数统一推送 FINAL 事件。



    Args:

        session_id: 当前流式查询对应的会话 ID。

        query: 用户原始问题。

    """

    try:

        state = invoke_query(session_id=session_id, query=query, is_stream=True, user_id=user_id)

        # 全流程结束，推送最终完整结果（answer + image_urls + domain + map_data）

        push_to_session(

            session_id,

            SSEEvent.FINAL,

            {

                "answer": get_task_result(session_id, "answer") or state.get("answer", ""),

                "status": "completed",

                "image_urls": state.get("image_urls", []),

                "domain": state.get("domain", ""),

                # 行程类回答附带地图数据（问题5）：前端据此内嵌高德地图；非行程类为空结构

                "map_data": state.get("map_data") or {},

            },

        )

    except Exception as exc:

        logger.exception(f"流式查询执行失败，session_id={session_id}, error={exc}")

        update_task_status(session_id, TASK_STATUS_FAILED, True)

        push_to_session(

            session_id,

            SSEEvent.ERROR,

            {"message": "查询执行失败，请检查日志或稍后重试。"},

        )





def start_stream_query(

    background_tasks: BackgroundTasks,

    query: str,

    session_id: str | None = None,

    user_id: str | None = None,

) -> QueryResponse:

    """

    启动一条流式查询任务。



    Args:

        background_tasks: FastAPI 后台任务对象。

        query: 用户原始问题。

        session_id: 可选会话 ID；为空时自动生成。



    Returns:

        QueryResponse: 返回处理中提示与当前会话 ID。

    """

    final_session_id = session_id or new_session_id()

    # 创建 SSE 队列，供 /stream 端点消费

    create_sse_queue(final_session_id)

    background_tasks.add_task(

        run_stream_query_background,

        final_session_id,

        query,

        user_id or '',

    )

    return QueryResponse(message="结果正在处理中...", session_id=final_session_id)





def execute_query(query: str, session_id: str | None = None, user_id: str | None = None) -> QueryResponse:

    """

    以非流式方式执行查询。



    Args:

        query: 用户原始问题。

        session_id: 可选会话 ID；为空时自动生成。



    Returns:

        QueryResponse: 包含最终答案和已完成节点的响应对象。

    """

    final_session_id = session_id or new_session_id()

    state = invoke_query(session_id=final_session_id, query=query, is_stream=False, user_id=user_id or "")

    answer = get_task_result(final_session_id, "answer") or state.get("answer", "")

    return QueryResponse(

        message="处理完成!",

        session_id=final_session_id,

        answer=answer,

        done_list=get_done_task_list(final_session_id),

        image_urls=state.get("image_urls", []),

        domain=state.get("domain", ""),

        map_data=state.get("map_data") or {},

    )





def build_history_response(session_id: str, limit: int = 10) -> HistoryResponse:

    """

    查询并组装指定会话的历史记录。



    Args:

        session_id: 目标会话 ID。

        limit: 返回记录条数上限。



    Returns:

        HistoryResponse: 组装后的历史消息集合。

    """

    records = history_repository.list_recent(session_id, limit=limit)

    items = [

        HistoryItem(

            _id=str(record.get("_id")) if record.get("_id") is not None else "",

            session_id=record.get("session_id", ""),

            role=record.get("role", ""),

            text=record.get("text", ""),

            rewritten_query=record.get("rewritten_query", ""),

            # Mongo 历史字段仍叫 item_names（兼容存量数据），对外语义统一为关联主体

            item_names=record.get("item_names", []),

            image_urls=record.get("image_urls") or [],

            map_data=record.get("map_data") or {},

            ts=record.get("ts"),

        )

        for record in records

    ]

    return HistoryResponse(session_id=session_id, items=items)





def build_session_list(limit: int = 50) -> SessionListResponse:

    """

    按会话聚合全部历史会话，供会话管理页展示。



    Args:

        limit: 最多返回的会话数。



    Returns:

        SessionListResponse: 会话摘要列表（按最后活跃时间倒序）。

    """

    records = history_repository.list_sessions(limit=limit)

    items = [SessionSummary(**record) for record in records]

    return SessionListResponse(sessions=items)





def clear_query_history(session_id: str) -> dict:

    """

    清空指定会话的历史记录。



    Args:

        session_id: 目标会话 ID。



    Returns:

        dict: 删除结果说明。

    """

    delete_count = history_repository.clear_session(session_id)

    return {

        "message": f"删除:{session_id}会话对应的聊天记录成功!!",

        "deleted_count": delete_count,

    }





# ====================== 接口定义 ======================





@app.get("/health")

def health():

    """返回查询服务健康检查结果"""

    return {

        "ok": True,

        "app": settings.query_app_name,

        "env": settings.app_env,

        "module": "query",

    }

@app.get("/registry")

def registry_overview():

    """G/H/I 只读可观测端点：技能目录 / 数据源时效 / 动作清单（便于可管理可迁移）。"""

    out: dict = {"skills": [], "data_sources": [], "stale_policies": [], "actions": []}

    try:

        from app.shared.runtime.skill_registry import registry as skill_registry

        out["skills"] = [s.model_dump() for s in skill_registry.list_all()]

    except Exception as e:  # noqa: BLE001

        out["skills"] = [{"error": str(e)}]

    try:

        from app.shared.runtime.data_source_registry import registry as data_registry

        out["data_sources"] = [s.model_dump() for s in data_registry.list_all()]

        out["stale_policies"] = data_registry.staleness_report()

    except Exception as e:  # noqa: BLE001

        out["data_sources"] = [{"error": str(e)}]

    try:

        from app.shared.runtime.action_registry import registry as action_registry

        out["actions"] = [s.model_dump() for s in action_registry.list_all()]

    except Exception as e:  # noqa: BLE001

        out["actions"] = [{"error": str(e)}]

    return out

@app.get("/")

def index():

    """返回查询服务首页导航信息"""

    return {

        "message": "Enterprise RAG Query Service",

        "docs": "/docs",

        "openapi": "/openapi.json",

        "routes": {

            "query": "/query",

            "stream": "/stream/{session_id}",

            "history": "/history/{session_id}",

        },

    }





@app.get("/html")

def query_html():

    """返回统一查询演示页面（文旅/闲聊自动路由）"""

    html_path = STATIC_ROOT / "app" / "web" / "chat.html"

    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])





@app.get("/preview.jpg")

def preview_bg():

    """返回页面背景底图（preview.jpg），供 chat.html 做半透明背景使用。"""

    img_path = STATIC_ROOT / "preview.jpg"

    return FileResponse(path=img_path, media_type="image/jpeg")





@app.get("/preview2.jpg")

def preview2_bg():

    """返回备用背景底图（preview2.jpg）。"""

    img_path = STATIC_ROOT / "preview2.jpg"

    return FileResponse(path=img_path, media_type="image/jpeg")





@app.get("/ink-chat.jpg")

def ink_chat_bg():

    """返回水墨主题检索页背景底图（png/ 下的红日红梅水墨 jpg），供 chat.html 水墨主题使用。"""

    img_path = STATIC_ROOT / "png" / "7237.jpg_wh860.jpg"

    return FileResponse(path=img_path, media_type="image/jpeg")





@app.get("/favicon.ico")

def favicon():

    """返回内嵌 SVG favicon，避免浏览器自动请求 /favicon.ico 出现 404。"""

    svg = (

        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'

        '<rect width="64" height="64" rx="6" fill="#6b9080"/>'

        '<text x="32" y="44" font-size="34" text-anchor="middle" fill="#f8fafc">简</text>'

        "</svg>"

    )

    return Response(content=svg, media_type="image/svg+xml")





def _maybe_record_feedback(request: QueryRequest) -> None:

    """Loop D2：隐式反馈捕获（纯日志，异常安全）。基于当前 query 识别改写/追问/放弃信号。"""

    try:

        q = (request.query or "").strip()

        if not q:

            return

        prev = ""

        try:

            recs = history_repository.list_recent(request.session_id, limit=2)

            if recs:

                last = recs[-1]

                if isinstance(last, dict):

                    prev = last.get("query") or last.get("user") or last.get("original_query") or ""

                else:

                    prev = getattr(last, "query", "") or getattr(last, "original_query", "")

        except Exception:

            prev = ""

        sig = detect_implicit_feedback(q, prev_query=prev)

        if sig:

            record_implicit_feedback(request.session_id, sig, query=q)

    except Exception:

        pass





@app.post("/query", response_model=QueryResponse)

async def query(request: QueryRequest, background_tasks: BackgroundTasks):

    """

    统一处理普通问答与流式问答请求。



    - 非流式：同步执行全流程，直接返回完整答案

    - 流式：后台启动查询任务，返回 session_id，前端通过 /stream/{session_id} 建立 SSE 连接消费增量

    """

    # === D2 Loop 闭环：隐式反馈捕获（纯日志，不影响主链路） ===

    _maybe_record_feedback(request)



    if request.is_stream:

        return start_stream_query(

            background_tasks=background_tasks,

            query=request.query,

            session_id=request.session_id,

            user_id=request.user_id,

        )

    return execute_query(query=request.query, session_id=request.session_id, user_id=request.user_id)





@app.get("/stream/{session_id}")

async def stream_query_result(session_id: str, request: Request):

    """

    建立指定会话的 SSE 结果流。

    前端通过 EventSource 连接此端点，消费 ready/progress/delta/final/error 事件。

    """

    return StreamingResponse(

        sse_generator(session_id, request),

        media_type="text/event-stream",

        headers={

            "Cache-Control": "no-cache",

            "Connection": "keep-alive",

            "X-Accel-Buffering": "no",

        },

    )





@app.post("/stop/{session_id}")

def stop_stream_generation(session_id: str):

    """

    用户手动停止生成。

    设置协作式取消标记：流式输出循环在下一个 chunk 前检查到标记即中断 LLM 输出，

    已生成的部分文本会作为回答保留并落库。

    """

    request_cancel(session_id)

    logger.info(f"收到停止生成请求，session_id={session_id}")

    return {"message": "停止信号已发送", "session_id": session_id}





@app.get("/history/{session_id}", response_model=HistoryResponse)

def history(session_id: str, limit: int = 10):

    """查询会话历史记录"""

    return build_history_response(session_id=session_id, limit=limit)





@app.delete("/history/{session_id}")

def clear_history(session_id: str):

    """清空指定会话的历史记录"""

    return clear_query_history(session_id)





@app.get("/sessions", response_model=SessionListResponse)

def sessions(limit: int = 50):

    """列出全部历史会话摘要（按最后活跃时间倒序），供历史会话管理页使用"""

    return build_session_list(limit=limit)





@app.get("/history-page")

def history_page():

    """返回历史会话管理页面（查看/删除/跳转历史对话）"""

    html_path = STATIC_ROOT / "app" / "web" / "sessions.html"

    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])





@app.get("/trip-page")

def trip_page():

    """返回结构化行程规划页面；后端 /trip/plan、/trip/plan/nl 已在 query_server 挂载。"""

    html_path = STATIC_ROOT / "app" / "web" / "trip.html"

    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])





if __name__ == "__main__":

    import uvicorn



    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)

