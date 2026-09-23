"""
行程规划（结构化 JSON 行程）LangGraph 状态定义。

与统一查询图解耦：本图专注「高德采集 → LLM 规划 → POI 事实核查」独立链路，
由 /trip/plan（结构化）与 /trip/plan/nl（自然语言）两个 HTTP 入口驱动。
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Optional, TypedDict

from app.shared.runtime.date_utils import current_date_text


class TripPlanGraphState(TypedDict):
    """行程规划图的状态定义。"""

    session_id: str
    is_stream: bool  # 是否流式（占位，结构化 JSON 结果以同步返回为主）
    original_query: str  # NL 入口的原始问句（结构化入口可空）
    current_date: str

    # --- 输入（结构化 TripRequest dict）---
    trip_request: dict[str, Any]

    # --- 中间产物 ---
    pois: list[dict[str, Any]]
    pois_text: str
    weather_rows: list[dict[str, Any]]
    weather_text: str
    hotels: list[dict[str, Any]]
    hotels_text: str
    routes_text: str
    routes: list  # 往返交通结构化数据（RouteInfo dict 列表，出发地为空时为 []）

    # --- 输出 ---
    plan_raw: dict[str, Any]  # 核查前的 TripPlan dict
    trip_plan: dict[str, Any]  # 核查后的最终行程 dict（HTTP 返回体 data）
    rendered_text: str  # 可读 Markdown 渲染（供降级/日志）
    error: str  # 显式错误信息（成功为空）


_trip_plan_default_state: TripPlanGraphState = {
    "session_id": "",
    "is_stream": False,
    "original_query": "",
    "current_date": "",
    "trip_request": {},
    "pois": [],
    "pois_text": "",
    "weather_rows": [],
    "weather_text": "",
    "hotels": [],
    "hotels_text": "",
    "routes_text": "",
    "plan_raw": {},
    "trip_plan": {},
    "rendered_text": "",
    "error": "",
}


def create_trip_plan_state(
    session_id: Optional[str] = None,
    is_stream: bool = False,
    trip_request: Optional[dict[str, Any]] = None,
    original_query: str = "",
    **overrides,
) -> TripPlanGraphState:
    """创建行程规划图默认状态，支持覆盖。current_date 在创建时注入。"""
    state = copy.deepcopy(_trip_plan_default_state)
    state["session_id"] = session_id or f"trip_{uuid.uuid4().hex[:12]}"
    state["is_stream"] = is_stream
    state["original_query"] = original_query or ""
    if trip_request:
        state["trip_request"] = dict(trip_request)
    state.update(overrides)
    if not state.get("current_date"):
        state["current_date"] = current_date_text()
    return state
