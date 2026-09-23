# -*- coding: utf-8 -*-
"""对话内行程规划 · 槽位桥接（第二步：对话直连 trip_plan 智能体）。

背景（2026-09-14 主人拍板方案 A·第二步）：
    第一步已实现"缺项先反问"（plan_slot_service），但行程仍由 itinerary_service
    的"KB 文本那套"生成（纯 Markdown，无 POI 事实核查、无结构化数据）。
    第二步把对话内规划改由独立的 trip_plan 智能体产出结构化 TripPlan
    （高德 POI + 天气 + 往返交通 + 预算），再渲染成 Markdown 回灌对话。

分层（依赖方向 lint 强制 shared <- infra <- rag <- process <- api）：
    本模块属 **rag 层**，只做**纯逻辑**：
        抽值 → 构造 TripRequest → 渲染（图片 / 地图 / 落历史 / SSE）
    **驱动 trip_plan 子图**那一步属 process 层职责，放在
    `app/process/unified_query/agent/trip_plan_engine.py`，避免 rag 反向依赖 process。

边界（刻意为之）：
    - 开关 `PLAN_ENGINE_TRIP_PLAN` 默认 OFF；OFF 时本模块不被调用，
      对话规划完全走原 `itinerary_service.generate_itinerary`，行为逐字节不变。
    - 任一环节失败（抽不到城市 / 无有效行程）**返回 None**，
      由 process 层协调器回落原引擎——保证"换了新引擎也不会答不出话"。
    - 缺项（天数/日期/交通/住宿）一律用合理默认补全，不再二次反问：
      主人拍板"同一会话最多反问一次"，补信息那轮必须能直接出结果。
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm import llm_provider
from app.shared.config.common import env_bool
from app.shared.runtime.date_utils import current_date_text
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.schemas.trip_plan import TripRequest

# ============================================================
# 开关与默认值
# ============================================================
_DEFAULT_DAYS = 3
_DEFAULT_TRANSPORTATION = "公共交通"
_DEFAULT_ACCOMMODATION = "经济型酒店"
_MAX_DAYS = 30
_MAX_IMAGES = 6

_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def trip_plan_engine_enabled() -> bool:
    """对话内行程规划是否改用 trip_plan 智能体（默认 OFF）。"""
    return env_bool("PLAN_ENGINE_TRIP_PLAN", False)


# ============================================================
# 值抽取
# ============================================================
@step_log("extract_trip_plan_values")
def extract_trip_plan_values(query: str, history_text: str = "") -> dict:
    """从当前问题 + 历史会话抽取行程参数值。

    Returns:
        dict：`{city/origin/days/start_date/preferences/transportation/accommodation/free_text/budget}`。
        **任何异常返回 {}**（桥接随即回落原引擎），绝不把对话链路卡死。
    """
    query = (query or "").strip()
    if not query:
        return {}

    try:
        def _call_model() -> dict:
            prompt = load_prompt(
                "tourism/trip_plan_slots",
                history_text=history_text or "（无）",
                current_date=current_date_text(),
                query=query,
            )
            client = llm_provider.chat(json_mode=True)
            messages = [
                SystemMessage(content="你是旅行需求结构化助手，只抽取用户已给出的信息，不做任何推测。"),
                HumanMessage(content=prompt),
            ]
            return (client | JsonOutputParser()).invoke(messages)

        raw = cached_invoke(
            namespace="trip_plan_slots",
            cache_parts=(query, history_text or ""),
            producer=_call_model,
            cache_label="行程参数抽取",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"行程参数抽取失败，回落原行程引擎：{e}")
        return {}

    return raw if isinstance(raw, dict) else {}


# ============================================================
# 值规整与 TripRequest 构造
# ============================================================
def _today() -> date:
    try:
        return datetime.strptime(current_date_text(), "%Y-%m-%d").date()
    except Exception:  # noqa: BLE001
        return date.today()


def parse_days(value: Any) -> int:
    """把天数解析为 1~30 的整数；无法解析或越界返回 0。"""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        n = value
    else:
        text = str(value or "").strip()
        match = re.search(r"\d+", text)
        if match:
            n = int(match.group())
        else:
            n = 0
            for ch in text:
                if ch in _CN_NUM:
                    n = _CN_NUM[ch]
                    break
    return n if 1 <= n <= _MAX_DAYS else 0


def normalize_date(value: Any) -> Optional[date]:
    """把日期字符串解析为 date；无法解析返回 None（由调用方补默认）。"""
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # 月日形式：按当前年份补齐
    for fmt in ("%m-%d", "%m/%d", "%m月%d日"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.date().replace(year=_today().year)
        except ValueError:
            continue
    return None


def normalize_preferences(value: Any) -> list[str]:
    """把偏好规整成字符串列表（兼容数组 / "自然风光、美食" 这类串）。"""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [p for p in re.split(r"[、,，/;；\s]+", text) if p]


def _parse_budget(value: Any) -> Optional[int]:
    """把预算解析为整数元；无法解析返回 None（不强行给默认预算）。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def build_trip_request(values: dict) -> Optional[TripRequest]:
    """把抽取结果补齐默认值后构造 TripRequest；目的地缺失返回 None（→ 回落原引擎）。

    默认策略（主人拍板"补充信息那一轮必须能出结果"）：
        - days 缺省 3 天
        - start_date 缺省"明天"
        - end_date = start_date + days - 1
        - transportation 缺省"公共交通"，accommodation 缺省"经济型酒店"
        - origin 允许为空（trip_plan 侧 origin 为空则不含往返交通段）
    """
    values = values or {}
    city = str(values.get("city") or "").strip()
    if not city:
        logger.info("对话内 trip_plan 桥接：未解析到目的地城市，回落原行程引擎")
        return None

    days = parse_days(values.get("days")) or _DEFAULT_DAYS
    start_date = normalize_date(values.get("start_date")) or (_today() + timedelta(days=1))
    end_date = start_date + timedelta(days=days - 1)

    return TripRequest(
        city=city,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        travel_days=days,
        transportation=str(values.get("transportation") or "").strip() or _DEFAULT_TRANSPORTATION,
        accommodation=str(values.get("accommodation") or "").strip() or _DEFAULT_ACCOMMODATION,
        preferences=normalize_preferences(values.get("preferences")),
        free_text_input=str(values.get("free_text") or "").strip(),
        origin=str(values.get("origin") or "").strip(),
        budget=_parse_budget(values.get("budget")),
    )


# ============================================================
# 结构化行程 → 对话可用的渲染数据
# ============================================================
def collect_image_urls(trip_plan: dict, limit: int = _MAX_IMAGES) -> list[str]:
    """从结构化行程的景点里收集图片 URL（去重、限流）。"""
    urls: list[str] = []
    for day in (trip_plan or {}).get("days") or []:
        for attr in (day or {}).get("attractions") or []:
            candidates = [attr.get("image_url")] + list(attr.get("photos") or [])
            for url in candidates:
                if isinstance(url, str) and url.strip() and url not in urls:
                    urls.append(url)
                if len(urls) >= limit:
                    return urls
    return urls


def build_render_map(trip_plan: dict) -> dict:
    """把结构化行程转成前端地图数据（自带经纬度，无需地理编码）。

    输出格式与 `itinerary_map_service.build_itinerary_map` 对齐：
        {"ok","name","city","days":[{"index","label","attractions":[
              {"name","lon","lat","address","kind"}]}]}
    ok=False 时前端不渲染地图。
    """
    plan = trip_plan or {}
    city = str(plan.get("city") or "").strip()
    days_out: list[dict] = []
    for day in plan.get("days") or []:
        points: list[dict] = []
        for attr in (day or {}).get("attractions") or []:
            location = attr.get("location") or {}
            lon, lat = location.get("longitude"), location.get("latitude")
            if lon is None or lat is None:
                continue
            points.append({
                "name": str(attr.get("name") or ""),
                "lon": lon,
                "lat": lat,
                "address": str(attr.get("address") or ""),
                "kind": "poi",
            })
        if points:
            index = int((day or {}).get("day_index") or 0)
            days_out.append({
                "index": index,
                "label": f"第{index + 1}天",
                "attractions": points,
            })
    if not city or not days_out:
        return {"ok": False, "name": "", "city": city, "days": []}
    return {"ok": True, "name": f"{city}行程", "city": city, "days": days_out}


# ============================================================
# 对话 state ↔ 结构化请求（纯逻辑，供 process 层协调器调用）
# ============================================================
@step_log("prepare_trip_plan_request")
def prepare_trip_plan_request(state: dict) -> Optional[TripRequest]:
    """读会话历史 + 抽值 + 构造 TripRequest。

    Returns:
        TripRequest；无法确定目的地时 None（调用方据此回落原引擎）。
    """
    session_id = state.get("session_id") or ""
    query = state.get("original_query") or state.get("rewritten_query") or ""

    # 读历史（补信息那一轮的关键：目的地在前一轮、细节在本轮）
    history_text = ""
    try:
        from app.rag.common.history_text_utils import build_history_text
        from app.rag.tourism_query.attraction_confirm_service import load_history

        history = state.get("history") or load_history(session_id)
        history_text = build_history_text(history)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"对话内 trip_plan 桥接：读取会话历史失败（不影响规划）：{e}")

    values = extract_trip_plan_values(query, history_text)
    return build_trip_request(values)


@step_log("finalize_trip_plan_answer")
def finalize_trip_plan_answer(state: dict, final: dict) -> Optional[dict]:
    """把 trip_plan 智能体的返回渲染成对话节点更新（答案 / 图片 / 地图 / 落历史）。

    Returns:
        dict: `{"answer","image_urls","itinerary_map","map_data"}`；
        智能体未产出可读行程时 None（调用方回落原引擎）。
    """
    from app.infra.persistence import history_repository

    answer = str((final or {}).get("rendered_text") or "").strip()
    if not answer:
        logger.warning("对话内 trip_plan 桥接：智能体未产出可读行程，回落原行程引擎")
        return None

    session_id = state.get("session_id") or ""
    is_stream = bool(state.get("is_stream", False))
    trip_plan = (final or {}).get("trip_plan") or {}
    image_urls = collect_image_urls(trip_plan)
    map_data = build_render_map(trip_plan)

    # 按对话输出协议落 result / SSE / state
    from app.shared.utils.sse_utils import SSEEvent, push_to_session
    from app.shared.utils.task_utils import set_task_result

    set_task_result(session_id, "answer", answer)
    if is_stream:
        push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
    state["answer"] = answer
    state["image_urls"] = image_urls
    state["itinerary_map"] = map_data
    state["map_data"] = map_data
    state["trip_plan"] = trip_plan

    # 落历史（多轮续接靠它；失败不影响本次回答）
    try:
        history_repository.save_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            rewritten_query=state.get("rewritten_query") or state.get("original_query"),
            item_names=state.get("item_names", []),
            image_urls=image_urls,
            domain=state.get("domain", "tourism"),
            is_plan=True,
            map_data=map_data,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"对话内 trip_plan 桥接：落历史失败（不影响本次回答）：{e}")

    logger.info(
        f"对话内 trip_plan 智能体完成：city={trip_plan.get('city')} "
        f"天数={len(trip_plan.get('days') or [])} 图片={len(image_urls)} "
        f"地图点位={sum(len(d.get('attractions') or []) for d in map_data.get('days') or [])}"
    )
    return {
        "answer": answer,
        "image_urls": image_urls,
        "itinerary_map": map_data,
        "map_data": map_data,
    }
