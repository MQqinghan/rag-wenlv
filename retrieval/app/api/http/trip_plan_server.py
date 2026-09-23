"""
行程规划 HTTP 路由（结构化 JSON 行程 —— 第四路能力）。

- POST /trip/plan       ：结构化 TripRequest → TripPlanResponse
- POST /trip/plan/nl    ：自然语言问句 → 抽取槽位 → TripPlanResponse
- GET  /trip/page       ：行程规划演示页 trip.html
- GET  /trip/amap-js-key：返回前端地图所需高德 Web(JS) Key（供 trip.html 渲染地图）

设计要点：
- 与既有 /query 路由完全解耦，避免污染文旅/闲聊回归（40 题评测）；
- 失败显式返回 success=false + 可读 message，绝不返回占位假行程；
- 长耗时规划放 asyncio.to_thread，避免阻塞事件循环。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse
from mimetypes import guess_type

from app.shared.schemas.trip_plan import (
    TripPlan,
    TripPlanNLRequest,
    TripPlanResponse,
    TripRequest,
)
from app.process.trip_plan.agent.main_graph import invoke_nl, invoke_structured
from app.shared.config.common import env_str
from app.shared.runtime.logger import logger

# R12（2026-09-11）：静态页面根目录必须锚定本文件所在工作空间。
# 不可用共享层 logger.PROJECT_ROOT——其 Path.resolve() 会展开 junction 指向 RAG_shared_infra（那里没有 app/web）。
STATIC_ROOT = Path(__file__).resolve().parents[3]

router = APIRouter(prefix="/trip", tags=["行程规划"])

# 前端地图 JS Key（Web端 JS API），与后端 Web服务 Key（AMAP_API_KEY）不同用途
TRIP_AMAP_JS_KEY: str = env_str("TRIP_AMAP_JS_KEY", default="")
AMAP_JS_SECURITY_CODE: str = env_str("AMAP_JS_SECURITY_CODE", default="")


@router.post("/plan", response_model=TripPlanResponse, summary="生成结构化旅行计划")
async def plan_trip(request: TripRequest) -> TripPlanResponse:
    """根据结构化表单生成结构化 JSON 行程（含 POI 事实核查）。"""
    try:
        final = await asyncio.to_thread(invoke_structured, request)
        data = final.get("trip_plan") or {}
        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功（行程内景点均已通过高德事实核查）",
            data=TripPlan(**data),
        )
    except Exception as e:  # noqa: BLE001 —— 显式失败：返回可读错误，不冒充成功
        logger.exception("生成旅行计划失败")
        return TripPlanResponse(success=False, message=f"生成旅行计划失败：{e}", data=None)


@router.post("/plan/nl", response_model=TripPlanResponse, summary="自然语言生成旅行计划")
async def plan_trip_nl(request: TripPlanNLRequest) -> TripPlanResponse:
    """从一句话中抽取行程槽位并生成结构化行程（第四路对话入口）。"""
    try:
        final = await asyncio.to_thread(invoke_nl, request.query, request.session_id)
        data = final.get("trip_plan") or {}
        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功（行程内景点均已通过高德事实核查）",
            data=TripPlan(**data),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("NL 生成旅行计划失败")
        return TripPlanResponse(success=False, message=f"生成旅行计划失败：{e}", data=None)


@router.get("/page", summary="行程规划演示页")
def trip_page():
    """返回行程规划演示页（结构化表单 + 地图结果）。"""
    html_path = STATIC_ROOT / "app" / "web" / "trip.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@router.get("/amap-js-key", summary="前端高德 JS Key")
def amap_js_key() -> dict:
    """供 trip.html 拉取高德 Web(JS) Key；未配置时返回空（地图禁用，行程列表仍可用）。"""
    return {
        "key": TRIP_AMAP_JS_KEY,
        "securityJsCode": AMAP_JS_SECURITY_CODE,
        "configured": bool(TRIP_AMAP_JS_KEY),
    }
