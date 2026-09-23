"""

行程规划（结构化 JSON）服务层。



由 helloagents-trip-planner 的 backend/app/agents/trip_planner_agent.py 迁移改编：

- 4 个 SimpleAgent 管道 → LangGraph 节点图（本模块提供各节点所需纯函数）；

- MCP 工具调用 → amap_gateway（REST）直连；

- 保留核心差异化能力「POI 事实核查」（_verify_attractions_with_amap）：

  名称命中但坐标偏差>阈值的用高德坐标覆盖；查无此地的用高德真实 POI 顶替/移除。

"""

from __future__ import annotations



import json

import math

import re

from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import date, datetime, timedelta

from typing import Any, Optional



from app.shared.schemas.trip_plan import (

    Attraction,

    Budget,

    DayPlan,

    Hotel,

    Location,

    Meal,

    TripPlan,

    TripRequest,

    WeatherInfo,

)

from app.infra.amap_gateway import OUTCOME_RATE_LIMITED, amap_gateway

from app.infra.weather_gateway import weather_gateway

from app.infra import rail_gateway

from app.infra.rail_gateway import format_duration, format_fares

from app.infra.llm import llm_provider

from app.shared.runtime.date_utils import current_date_text

from app.shared.config.common import env_bool, env_int, env_str

from app.shared.runtime.load_prompt import load_prompt

from app.shared.runtime.logger import logger, step_log



# POI 事实核查：计划坐标与高德返回坐标的最大偏差（米），与旧工程保持一致

AMAP_MATCH_DIST_M = 3000

# 规划 LLM 单次生成上限 token（结构化行程较长）

PLAN_MAX_TOKENS = 4500

# 行程规划节点专用模型（留空 = 沿用全局 LLM_PROVIDER 默认模型，行为与改动前一致）。
# 本机实测：同一规划提示词下 qwen-flash 生成吞吐约为 glm-4-flash 的 3 倍、该节点墙钟减半。
TRIP_PLANNER_MODEL: str = env_str("TRIP_PLANNER_MODEL", default="")

# 按天并行规划（B 方案，2026-09-11）：整份生成 → 每天一次并发生成，墙钟≈max(单日) 而非 Σ。
# 关闭开关或单天行程时自动回退原单次生成，行为与改动前一致。
TRIP_PARALLEL_PLAN: bool = env_bool("TRIP_PARALLEL_PLAN", default=True)
TRIP_DAY_WORKERS: int = env_int("TRIP_DAY_WORKERS", default=4)

# 单日行程输出上限 token（单日 JSON 短于整份，2000 足够）
DAY_MAX_TOKENS = 2000
# 总览/预算输出上限 token（输出很短）
OVERVIEW_MAX_TOKENS = 800

# 每天景点数量范围

MIN_ATTRACTION_PER_DAY = 1

MAX_ATTRACTION_PER_DAY = 3

# POI 核查并发度（C1 主体）：受 amap_gateway._Throttle 统一限速，不会打爆高德 QPS

TRIP_VERIFY_WORKERS: int = env_int("TRIP_VERIFY_WORKERS", default=4)

# 偏好关键词采集并发度（C2）：多关键词 search_poi 并行 + 去重保序，受同一 _Throttle 限速

TRIP_COLLECT_POIS_WORKERS: int = env_int("TRIP_COLLECT_POIS_WORKERS", default=4)

# 每日景点下限（C1-fix）：核查后当日不足该值时用高德真实 POI 补齐，保证行程可用

TRIP_MIN_ATTRS_PER_DAY: int = env_int("TRIP_MIN_ATTRS_PER_DAY", default=2)

# 补位候选池大小（M6 + 方案A）：只取高德按相关度返回的"热门"前若干名（宁缺毋滥），

# 冷门/低质候选独立核查易失败且会让行程像硬凑；offset 设小以保证补到的都是热门、可核查。

TOPUP_POI_OFFSET: int = env_int("TRIP_TOPUP_POI_OFFSET", default=6)



# 全局单例（类仅用于组织函数，避免直接 import 的循环依赖）

_extract_date_cache: dict[str, str] = {}





# ---------------------------------------------------------------- 工具函数

def _json_strip(text: str) -> str:

    """去掉 Markdown 代码块围栏。"""

    text = re.sub(r"^\s*```(?:json)?\s*", "", (text or ""), flags=re.MULTILINE)

    return re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE).strip()





def parse_llm_json(text: str) -> dict:

    """从 LLM 文本中稳健抽取 JSON 对象；失败返回空 dict。"""

    raw = _json_strip(text or "")

    try:

        data = json.loads(raw)

        return data if isinstance(data, dict) else {}

    except Exception:

        pass

    # 兜底：截取首尾花括号

    start, end = raw.find("{"), raw.rfind("}")

    if start != -1 and end > start:

        try:

            data = json.loads(raw[start : end + 1])

            return data if isinstance(data, dict) else {}

        except Exception:

            return {}

    return {}





def haversine_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:

    """球面距离（米）。"""

    lng1, lat1, lng2, lat2 = map(math.radians, (lng1, lat1, lng2, lat2))

    a = (

        math.sin((lat2 - lat1) / 2) ** 2

        + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2

    )

    return 6371000 * 2 * math.asin(math.sqrt(a))





def _loc_of(location: Optional[Any]) -> Optional[tuple[float, float]]:

    """Location 对象 / dict / 字符串 统一取 (lng, lat)。"""

    if location is None:

        return None

    if isinstance(location, Location):

        return location.longitude, location.latitude

    if isinstance(location, dict):

        try:

            return float(location["longitude"]), float(location["latitude"])

        except (KeyError, TypeError, ValueError):

            return None

    if isinstance(location, str):

        parsed = Location.from_text(location)

        return (parsed.longitude, parsed.latitude) if parsed else None

    return None





def _coerce_request(raw: Any) -> Optional[TripRequest]:

    """把 TripRequest / dict 统一规整为 TripRequest；不合法返回 None。"""

    try:

        if isinstance(raw, TripRequest):

            return raw

        if isinstance(raw, dict):

            return TripRequest(**raw)

    except Exception as e:  # noqa: BLE001

        logger.warning(f"TripRequest 规整失败: {e}")

    return None





@step_log("trip_extract_request")

def extract_trip_request(query: str) -> Optional[dict]:

    """

    自然语言 → TripRequest 槽位（LLM JSON 抽取）。



    Returns:

        dict | None：成功返回 TripRequest 的 dict；无法确定城市/日期返回 None。

    """

    if not query or not query.strip():

        return None

    client = llm_provider.chat(json_mode=True)

    prompt = load_prompt("trip/trip_extract", current_date=current_date_text(), query=query.strip())

    try:

        data = parse_llm_json(client.invoke(prompt).content)

    except Exception as e:  # noqa: BLE001

        logger.warning(f"行程槽位抽取调用失败: {e}")

        return None

    if not data or not data.get("city"):

        logger.info(f"行程槽位抽取：未解析到城市，放弃 NL 结构化入口: {query[:30]}")

        return None

    req = _coerce_request(data)

    if req is None:

        return None

    return req.model_dump()





# ---------------------------------------------------------------- 信息采集

@step_log("trip_collect_pois")

def collect_pois(request: TripRequest, city_limit: bool = True) -> list[dict[str, Any]]:

    """

    按偏好关键词高德搜索景点 POI（去重保序）。



    Returns:

        归一化 POI dict 列表（字段同 amap_gateway.search_poi）。

    """

    keywords: list[str] = []

    prefs = [p.strip() for p in (request.preferences or []) if p and p.strip()]

    if prefs:

        keywords = list(dict.fromkeys(prefs[:4]))  # 多偏好全部拼接检索，避免只用第一个

    if not keywords:

        keywords = ["景点"]



    # C2（2026-09-09）：多关键词 search_poi 并行采集（ThreadPoolExecutor 范式，与核查一致），

    # 受 amap_gateway._Throttle 统一限速；任一路软失败返回 [] 不影响其余路。

    def _fetch_one(kw: str) -> tuple[str, list[dict[str, Any]]]:

        return kw, amap_gateway.search_poi(kw, city=request.city, citylimit=city_limit, offset=8)



    if len(keywords) > 1:

        workers = max(1, min(TRIP_COLLECT_POIS_WORKERS, len(keywords)))

        if workers == 1:

            fetched = [_fetch_one(kw) for kw in keywords]

        else:

            with ThreadPoolExecutor(max_workers=workers) as pool:

                fetched = list(pool.map(_fetch_one, keywords))

    else:

        fetched = [_fetch_one(keywords[0])]



    merged: list[dict[str, Any]] = []

    seen: set[str] = set()

    for kw, pois in fetched:

        for p in pois:

            name = str(p.get("name") or "").strip()

            if not name or name in seen:

                continue

            seen.add(name)

            merged.append(p)

        if len(merged) >= 12:

            break

    # 通用兜底：候选仍不足时补一轮“景点/必游”，保证规划器有充足素材

    if len(merged) < 5:

        for p in amap_gateway.search_poi("景点", city=request.city, citylimit=city_limit, offset=8):

            name = str(p.get("name") or "").strip()

            if name and name not in seen:

                seen.add(name)

                merged.append(p)

            if len(merged) >= 12:

                break

    return merged[:12]





@step_log("trip_collect_weather")

def collect_weather(request: TripRequest) -> list[dict[str, Any]]:

    """

    按行程区间裁剪逐日天气（统一走 weather_gateway：主和风、备高德），失败返回 []。



    和风最多可给到 30 天，故按行程跨度请求；高德备源硬上限 4 天会自动截断。

    """

    try:

        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()

        end = datetime.strptime(request.end_date, "%Y-%m-%d").date()

        span = max(1, (end - start).days + 1)

    except (ValueError, TypeError):

        start = end = None

        span = max(1, int(getattr(request, "travel_days", 1) or 1))

    result = weather_gateway.forecast(destination=request.city, city=request.city, days=span)

    rows = result.get("days") or []

    if not rows:

        logger.warning(f"行程天气采集失败[{request.city}],原因:{result.get('error')}")

        return []

    if result.get("degraded"):

        logger.warning(f"行程天气已降级至备源[{result.get('source')}],天数可能被截断[{request.city}]")

    if start and end:

        window = [r for r in rows if _date_in_window(r.get("date", ""), start, end)]

        # 出发日超出预报窗口（如一个月后出发）时保留全部可用数据，交由规划器参考

        return window or rows

    return rows





def _date_in_window(date_text: str, start: date, end: date) -> bool:

    try:

        d = datetime.strptime(str(date_text)[:10], "%Y-%m-%d").date()

        return start <= d <= end

    except ValueError:

        return False





@step_log("trip_collect_hotels")

def collect_hotels(request: TripRequest) -> list[dict[str, Any]]:

    """按住宿偏好搜索酒店 POI。"""

    kw = "酒店"

    if "豪华" in request.accommodation:

        kw = "豪华酒店"

    elif "民宿" in request.accommodation:

        kw = "民宿"

    pois = amap_gateway.search_poi(kw, city=request.city, citylimit=True, offset=8)

    return pois[:6]





def _poi_line(prefix: str, p: dict[str, Any]) -> str:

    loc = str(p.get("location") or "")

    tel = str(p.get("tel") or "")

    type_text = str(p.get("type") or "").split(";")[0] or str(p.get("typecode") or "")

    addr = str(p.get("address") or "")

    line = f"{prefix} {p.get('name')}"

    if type_text:

        line += f"（{type_text}）"

    if addr:

        line += f"｜{addr}"

    if loc:

        line += f"｜坐标{lnglat_cn(loc)}"

    if tel:

        line += f"｜电话{tel}"

    return line





def lnglat_cn(loc_text: str) -> str:

    """"lng,lat" → "lng,lat"（保持高德字符串原样，供 LLM 阅读即可）。"""

    return loc_text





def collect_pois_text(pois: list[dict[str, Any]]) -> str:

    if not pois:

        return "（无可用景点数据——请勿编造景点，仅在确无候选时输出说明）"

    return "\n".join(_poi_line(f"[{i + 1}]", p) for i, p in enumerate(pois))





def collect_weather_text(weather_rows: list[dict[str, Any]]) -> str:

    if not weather_rows:

        return "（无可用天气数据——天气字段填“-”/0，禁止编造天气）"

    lines = []

    for r in weather_rows:

        d = str(r.get("date") or "")[:10]

        day_w = r.get("desc") or "-"

        night_w = r.get("night_desc") or "-"

        t_max, t_min = r.get("temp_max", "-"), r.get("temp_min", "-")

        precip = r.get("precip")

        # 高德备源无降水量（网关置 None），此时不展示该字段，禁止编造

        precip_text = f" 降水{precip}mm" if precip not in (None, "", "0", "0.0") else ""

        wind_text = f" {r.get('wind')}" if r.get("wind") else ""

        lines.append(f"{d}: 白天{day_w}({t_max}°C) / 夜间{night_w}({t_min}°C){precip_text}{wind_text}")

    return "\n".join(lines)





def collect_hotels_text(hotels: list[dict[str, Any]]) -> str:

    if not hotels:

        return "（无可用酒店数据——hotel 字段填 null）"

    return "\n".join(_poi_line(f"[{i + 1}]", p) for i, p in enumerate(hotels))



# ---------------------------------------------------------------- 规划生成

# ---------------------------------------------------------------- 往返交通采集

def _is_rail_mode(transportation: str) -> bool:

    """往返交通方式判定：公共交通/高铁/火车/动车/铁路等走铁路网关取真实车次与票价。"""

    t = (transportation or "").strip()

    return any(k in t for k in ("高铁", "火车", "铁路", "列车", "动车", "轻轨", "地铁", "公共交通"))





def _rail_to_info(leg: str, origin: str, destination: str, rail: dict) -> dict:

    """把 12306-MCP 归一化结果转 RouteInfo（取最快一班为代表；里程不编造、票价透传）。"""

    trains = rail.get("trains") or []

    if not trains:

        return {

            "leg": leg, "mode": "高铁", "origin": origin, "destination": destination,

            "duration_text": "", "distance_text": "",

            "cost_note": "铁路数据暂不可用，班次与票价请以 12306 官方为准",

        }



    def _dur_min(t: dict) -> int:

        d = str(t.get("duration") or "")

        if ":" in d:

            h, _, m = d.partition(":")

            try:

                return int(h) * 60 + int(m)

            except (TypeError, ValueError):

                return 10 ** 9

        return 10 ** 9



    best = min(trains, key=_dur_min)

    cls = (best.get("train_class") or "")

    if "高速" in cls or "动车" in cls:

        mode_label = "高铁"

    elif cls:

        mode_label = cls

    else:

        mode_label = "火车"

    dur = format_duration(best.get("duration"))

    fares = format_fares(best.get("prices"))

    cost_note = f"参考票价：{fares}（以 12306 官方为准）" if fares else "票价以 12306 官方为准"

    return {

        "leg": leg,

        "mode": mode_label,

        "origin": origin,

        "destination": destination,

        "duration_text": f"约 {dur}" if dur else "",

        "distance_text": "",  # 铁路接口不提供里程，绝不编造

        "cost_note": cost_note,
        "cost": _min_rail_fare(best.get("prices")),

    }





@step_log("trip_collect_routes")

def collect_routes(request: TripRequest) -> Optional[list]:

    """生成往返交通段。



    - 公共交通/高铁/火车类：优先 12306-MCP 铁路网关取真实车次/耗时/票价（MCP 不可用则退化高德 transit）；

    - 自驾类：高德驾车路线（跨城先地理编码取坐标）；

    - origin 为空返回 None（仅规划目的地市内行程）。

    任一数据源失败均软降级，不阻断主链路、不编造。

    """

    origin = (getattr(request, "origin", "") or "").strip()

    if not origin:

        return None

    trans = (request.transportation or "").strip()

    city = request.city

    start = (request.start_date or "").strip()

    ret = (request.end_date or "").strip() or start



    if _is_rail_mode(trans):

        routes: list[dict] = []

        go = rail_gateway.query_trains(origin, city, start)

        if go.get("ok"):

            routes.append(_rail_to_info("去程", origin, city, go))

        else:

            g = amap_gateway.plan_route(origin, city, origin_city=origin, destination_city=city, route_type="transit")

            if g:

                routes.append(_route_to_info("去程", "transit", origin, city, g, cost_note=_TRANSIT_ESTIMATE_NOTE))

        back = rail_gateway.query_trains(city, origin, ret)

        if back.get("ok"):

            routes.append(_rail_to_info("返程", city, origin, back))

        else:

            b = amap_gateway.plan_route(city, origin, origin_city=city, destination_city=origin, route_type="transit")

            if b:

                routes.append(_route_to_info("返程", "transit", city, origin, b, cost_note=_TRANSIT_ESTIMATE_NOTE))

        return routes or None



    # 自驾：高德驾车（跨城需先地理编码取坐标，否则 INVALID_PARAMS）

    routes = []

    go_geo = amap_gateway.geocode(origin, origin)

    dest_geo = amap_gateway.geocode(city, city)

    if go_geo and dest_geo:

        go = amap_gateway.plan_route(

            f"{go_geo['longitude']:.6f},{go_geo['latitude']:.6f}",

            f"{dest_geo['longitude']:.6f},{dest_geo['latitude']:.6f}",

            origin_city=origin, destination_city=city, route_type="driving",

        )

        if go:

            routes.append(_route_to_info("去程", "driving", origin, city, go))

    back_geo = amap_gateway.geocode(city, city)

    orig_geo = amap_gateway.geocode(origin, origin)

    if back_geo and orig_geo:

        back = amap_gateway.plan_route(

            f"{back_geo['longitude']:.6f},{back_geo['latitude']:.6f}",

            f"{orig_geo['longitude']:.6f},{orig_geo['latitude']:.6f}",

            origin_city=city, destination_city=origin, route_type="driving",

        )

        if back:

            routes.append(_route_to_info("返程", "driving", city, origin, back))

    return routes or None





def _fmt_duration(total_min: int) -> str:

    """把分钟数格式化为人读得懂的耗时描述（约 X 小时 Y 分钟 / 约 X 小时 / 约 X 分钟）。"""

    if not total_min:

        return ""

    if total_min < 60:

        return f"约 {total_min} 分钟"

    hours = total_min // 60

    mins = total_min % 60

    if mins:

        return f"约 {hours} 小时 {mins} 分钟"

    return f"约 {hours} 小时"





# 铁路不可用时回退到高德 transit 的诚实声明（transit/integrated 本为市内公交设计，跨城不可信）

_TRANSIT_ESTIMATE_NOTE = (

    "高德公交路线为市内公交换乘估算，跨城长途实际多为高铁/航班，"

    "具体班次、耗时与票价请以 12306 / 航司官方为准"

)





def _route_to_info(leg: str, mode: str, origin: str, dest: str, raw: dict, cost_note: str = "") -> dict:

    dur_min = int((raw.get("duration") or 0) // 60)

    dist_km = (raw.get("distance") or 0) / 1000.0

    return {

        "leg": leg,

        "mode": "公共交通" if mode == "transit" else "自驾",

        "origin": origin,

        "destination": dest,

        "duration_text": _fmt_duration(dur_min),

        "distance_text": f"约 {dist_km:.0f} 公里" if dist_km else "",

        "cost_note": cost_note or "高德不提供实时票价，请以实际购票为准",

    }





def collect_routes_text(routes: Optional[list]) -> str:

    if not routes:

        return ""

    out = []

    for r in routes:

        parts = [f"【{r['leg']}】{r['origin']} → {r['destination']}"]

        if r.get("mode"):

            parts.append(f"方式：{r['mode']}")

        if r.get("duration_text"):

            parts.append(f"耗时：{r['duration_text']}")

        if r.get("distance_text"):

            parts.append(f"距离：{r['distance_text']}")

        if r.get("cost_note"):

            parts.append(r["cost_note"])

        out.append("，".join(parts))

    return "\n".join(out)





@step_log("trip_plan_build_prompt")

def build_planner_prompt(

    request: TripRequest,

    pois_text: str,

    weather_text: str,

    hotels_text: str,

    routes_text: str = "",

) -> str:

    """组装规划器提示词（对应旧工程 _build_planner_query + PLANNER_AGENT_PROMPT）。"""

    prefs = "、".join(request.preferences or []) or "无"

    return load_prompt(

        "trip/planner",

        city=request.city,

        start_date=request.start_date,

        end_date=request.end_date,

        travel_days=request.travel_days,

        transportation=request.transportation,

        accommodation=request.accommodation,

        preferences=prefs,

        free_text_input=request.free_text_input or "（无）",

        attractions_info=pois_text,

        weather_info=weather_text,

        hotel_info=hotels_text,

        route_info=routes_text,

        current_date=current_date_text(),

    )





@step_log("trip_plan_generate")

def run_planner(

    request: TripRequest,

    pois_text: str,

    weather_text: str,

    hotels_text: str,

    routes_text: str = "",

    extra_transport: int = 0,

) -> TripPlan:

    """

    调用 LLM 生成结构化行程 JSON 并解析为 TripPlan。



    Raises:

        ValueError: 两次尝试（json 模式/普通模式）均解析失败时显式抛出，禁止静默兜底假数据。

    """

    prompt = build_planner_prompt(request, pois_text, weather_text, hotels_text, routes_text)

    last_err: Optional[Exception] = None

    for json_mode in (True, False):

        try:

            client = llm_provider.chat(model=TRIP_PLANNER_MODEL or None, json_mode=json_mode).bind(max_tokens=PLAN_MAX_TOKENS)

            response = client.invoke(prompt)

            text = response.content if hasattr(response, "content") else str(response)

            data = parse_llm_json(str(text))

            if not data or not data.get("days"):

                raise ValueError("响应中缺少 days 数组")

            plan = TripPlan(**data)

            plan = _normalize_plan(plan, request)
            if extra_transport:
                if plan.budget is None:
                    plan.budget = Budget()
                plan.budget.total_transportation = (plan.budget.total_transportation or 0) + extra_transport
                plan.budget.total = (plan.budget.total or 0) + extra_transport
            return plan

        except Exception as e:  # noqa: BLE001

            last_err = e

            logger.warning(f"行程规划生成/解析失败(json_mode={json_mode}): {e}")

    raise ValueError(f"行程规划响应解析失败: {last_err}")


# ---------------------------------------------------------------- 按天并行规划（B 方案）
def _min_rail_fare(prices: Any) -> int:
    """从 12306 票价字典取最低席别金额（真实数据透传，失败返回 0，绝不估算）。"""
    if not isinstance(prices, dict):
        return 0
    nums = [int(v) for v in prices.values() if isinstance(v, (int, float)) and v > 0]
    return min(nums) if nums else 0


def _as_int(value: Any) -> int:
    """把 LLM 给的钱数稳健转 int（支持 "¥120"/"120元"/浮点；失败返回 0）。"""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    m = re.search(r"-?\d+", str(value).replace(",", ""))
    return int(m.group(0)) if m else 0


def _day_date_text(request: TripRequest, day_index: int) -> str:
    """按出发日 + day_index 推算该天日期；解析失败返回空串。"""
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        return (start + timedelta(days=day_index)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _split_pois_for_days(pois: list[dict[str, Any]], days: int) -> list[list[dict[str, Any]]]:
    """
    把候选景点切成 days 份（连续分块），保证跨天**不重叠**，从根上避免同一景点多天重复。
    总量不足时允许末尾若干天为空，交由 verify_plan 用高德真实 POI 补位。
    """
    if days <= 0:
        return []
    total = len(pois)
    base, rem = divmod(total, days)
    chunks: list[list[dict[str, Any]]] = []
    idx = 0
    for i in range(days):
        size = base + (1 if i < rem else 0)
        chunks.append(list(pois[idx: idx + size]))
        idx += size
    return chunks


def build_day_prompt(
    request: TripRequest, day_index: int, pois_text: str, weather_text: str, hotels_text: str, routes_text: str = ""
) -> str:
    """组装单日规划提示词（对应 trip/day_planner.prompt）。"""
    prefs = "、".join(request.preferences or []) or "无"
    day_date = _day_date_text(request, day_index) or f"第{day_index + 1}天"
    return load_prompt(
        "trip/day_planner",
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        travel_days=request.travel_days,
        day_number=day_index + 1,
        day_index=day_index,
        day_date=day_date,
        transportation=request.transportation,
        accommodation=request.accommodation,
        preferences=prefs,
        free_text_input=request.free_text_input or "（无）",
        attractions_info=pois_text,
        weather_info=weather_text,
        hotel_info=hotels_text,
        route_info=routes_text or "（无）",
        current_date=current_date_text(),
    )


@step_log("trip_plan_day_generate")
def _run_one_day(
    request: TripRequest, day_index: int, pois_text: str, weather_text: str, hotels_text: str, routes_text: str = ""
) -> tuple[DayPlan, int]:
    """
    生成单日行程（json 模式失败自动回退普通模式）。
    返回 (DayPlan, 当天市内交通预估费)；两次尝试均失败时抛 ValueError。
    """
    prompt = build_day_prompt(request, day_index, pois_text, weather_text, hotels_text, routes_text)
    last_err: Optional[Exception] = None
    for json_mode in (True, False):
        try:
            client = llm_provider.chat(model=TRIP_PLANNER_MODEL or None, json_mode=json_mode).bind(max_tokens=DAY_MAX_TOKENS)
            response = client.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            data = parse_llm_json(str(text))
            if isinstance(data.get("days"), list) and data["days"]:
                data = data["days"][0]
            if not data or not data.get("attractions"):
                raise ValueError("响应缺少 attractions")
            data["day_index"] = day_index
            if not data.get("date"):
                data["date"] = _day_date_text(request, day_index)
            cost = _as_int(data.pop("transportation_cost", 0))
            return DayPlan(**data), cost
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(f"第{day_index + 1}天行程生成/解析失败(json_mode={json_mode}): {e}")
    raise ValueError(f"第{day_index + 1}天行程响应解析失败: {last_err}")


@step_log("trip_plan_overview_generate")
def _run_overview(
    request: TripRequest, pois_text: str, weather_text: str, hotels_text: str, routes_text: str = ""
) -> dict:
    """生成总体建议与预算预估；失败返回空 dict（调用方用代码兜底）。"""
    prefs = "、".join(request.preferences or []) or "无"
    prompt = load_prompt(
        "trip/plan_overview",
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        travel_days=request.travel_days,
        transportation=request.transportation,
        accommodation=request.accommodation,
        preferences=prefs,
        free_text_input=request.free_text_input or "（无）",
        attractions_info=pois_text,
        weather_info=weather_text,
        hotel_info=hotels_text,
        route_info=routes_text or "（无）",
        current_date=current_date_text(),
    )
    for json_mode in (True, False):
        try:
            client = llm_provider.chat(model=TRIP_PLANNER_MODEL or None, json_mode=json_mode).bind(max_tokens=OVERVIEW_MAX_TOKENS)
            response = client.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            data = parse_llm_json(str(text))
            if isinstance(data, dict) and data:
                return data
        except Exception as e:  # noqa: BLE001
            logger.warning(f"行程总览/预算生成失败(json_mode={json_mode}): {e}")
    return {}


def _budget_from_overview(raw: Any) -> Optional[Budget]:
    """把总览返回的 budget 转 Budget；字段全空/非法时返回 None（由代码兜底）。"""
    if not isinstance(raw, dict):
        return None
    return Budget(
        total_attractions=_as_int(raw.get("total_attractions")),
        total_hotels=_as_int(raw.get("total_hotels")),
        total_meals=_as_int(raw.get("total_meals")),
        total_transportation=_as_int(raw.get("total_transportation")),
        total=_as_int(raw.get("total")),
    )


def _compute_budget_from_days(days: list[DayPlan], extra_transport: int = 0) -> Budget:
    """总览调用失败时的代码兜底：按各天实际金额求和。"""
    att = sum(_as_int(a.ticket_price) for d in days for a in d.attractions)
    hotels = sum(_as_int(d.hotel.estimated_cost) for d in days if d.hotel)
    meals = sum(_as_int(m.estimated_cost) for d in days for m in d.meals)
    transport = _as_int(extra_transport)
    return Budget(
        total_attractions=att,
        total_hotels=hotels,
        total_meals=meals,
        total_transportation=transport,
        total=att + hotels + meals + transport,
    )


def _weather_infos_from_rows(rows: list[dict[str, Any]]) -> list[WeatherInfo]:
    """用采集到的真实天气行填充 weather_info（数值直出，避免让 LLM 二次转抄）。"""
    out: list[WeatherInfo] = []
    for r in rows or []:
        try:
            out.append(
                WeatherInfo(
                    date=str(r.get("date") or "")[:10],
                    day_weather=str(r.get("desc") or ""),
                    night_weather=str(r.get("night_desc") or ""),
                    day_temp=r.get("temp_max") if r.get("temp_max") not in (None, "") else 0,
                    night_temp=r.get("temp_min") if r.get("temp_min") not in (None, "") else 0,
                    wind_direction="",
                    wind_power=str(r.get("wind") or ""),
                )
            )
        except Exception:  # noqa: BLE001 —— 单行异常不影响整体
            continue
    return out


@step_log("trip_plan_parallel")
def run_planner_parallel(
    request: TripRequest,
    pois_text: str,
    weather_text: str,
    hotels_text: str,
    routes_text: str = "",
    pois: Optional[list[dict[str, Any]]] = None,
    weather_rows: Optional[list[dict[str, Any]]] = None,
    extra_transport: int = 0,
) -> TripPlan:
    """
    按天并行规划（B 方案）：把一次「整份生成」拆成 N 个「单日生成」并发执行，
    墙钟 ≈ max(单日) 而非 Σ；再并发跑一次「总览/预算」，二者互不阻塞。

    - 候选景点按天连续切块、跨天不重叠 → 从根上消除同景点多天重复；
    - 任一天失败 → 整体回退到原单次生成 `run_planner`（可靠性不降级）；
    - 关闭开关或单天行程 → 直接走 `run_planner`，行为与改动前完全一致。
    """
    req = _coerce_request(request)
    if req is None:
        raise ValueError("行程请求参数不合法")

    days_n = max(1, int(req.travel_days or 1))
    pois = list(pois or [])
    if not TRIP_PARALLEL_PLAN or days_n <= 1 or not pois:
        return run_planner(req, pois_text, weather_text, hotels_text, routes_text)

    chunks = _split_pois_for_days(pois, days_n)
    chunk_texts = [collect_pois_text(c) for c in chunks]

    days: list[Optional[DayPlan]] = [None] * days_n
    day_costs: list[int] = [0] * days_n
    overview: dict = {}

    workers = max(1, min(TRIP_DAY_WORKERS, days_n + 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        day_futs = {
            pool.submit(_run_one_day, req, i, chunk_texts[i], weather_text, hotels_text, routes_text): i
            for i in range(days_n)
        }
        overview_fut = pool.submit(_run_overview, req, pois_text, weather_text, hotels_text, routes_text)
        for fut in as_completed(list(day_futs.keys()) + [overview_fut]):
            if fut is overview_fut:
                try:
                    overview = fut.result() or {}
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"行程总览并发任务异常: {e}")
                continue
            idx = day_futs[fut]
            try:
                day, cost = fut.result()
                days[idx] = day
                day_costs[idx] = cost
            except Exception as e:  # noqa: BLE001
                logger.warning(f"第{idx + 1}天并行规划失败: {e}")

    if any(d is None for d in days):
        missing = [i + 1 for i, d in enumerate(days) if d is None]
        logger.warning(f"按天并行规划有 {len(missing)} 天未成功（第{missing}天），回退整份单次生成")
        return run_planner(req, pois_text, weather_text, hotels_text, routes_text)

    merged_days = [d for d in days if d is not None]
    plan = TripPlan(
        city=req.city,
        start_date=req.start_date,
        end_date=req.end_date,
        days=merged_days,
        weather_info=_weather_infos_from_rows(list(weather_rows or [])),
        overall_suggestions=str(overview.get("overall_suggestions") or ""),
        budget=_budget_from_overview(overview.get("budget")) or _compute_budget_from_days(merged_days, sum(day_costs) + extra_transport),
    )
    logger.info(
        f"按天并行规划完成：{len(merged_days)}天，并发度={workers}，"
        f"总览={'有' if overview else '缺失(已代码兜底)'}"
    )
    return _normalize_plan(plan, req)





def _normalize_plan(plan: TripPlan, request: TripRequest) -> TripPlan:

    """兜底补齐缺失字段：城市/日期/天序/预算缺省。"""

    plan.city = plan.city or request.city

    plan.start_date = plan.start_date or request.start_date

    plan.end_date = plan.end_date or request.end_date

    for i, day in enumerate(plan.days):

        if not day.date and plan.start_date:

            try:

                d = datetime.strptime(plan.start_date, "%Y-%m-%d").date() + timedelta(days=i)

                day.date = d.strftime("%Y-%m-%d")

            except ValueError:

                pass

        day.day_index = i

        # 无坐标/名称残缺的景点由后续 POI 核查节点统一治理，此处不静默删除

    if plan.budget is None:

        plan.budget = Budget()

    return plan





@step_log("trip_plan_validate")

def validate_plan(plan: TripPlan, request: TripRequest) -> None:

    """基础校验：天数对齐、每天至少保留 1 个景点（否则显式失败）。"""

    expected = request.travel_days

    if len(plan.days) < expected:

        logger.warning(f"行程天数不足：期望{expected}天，实际{len(plan.days)}天（交由核查节点尽力补齐）")





# ---------------------------------------------------------------- POI 事实核查

def _amap_text_search(keywords: str, city: str, offset: int = 10) -> list[dict[str, Any]]:

    """

    高德文本搜索 POI（原 MCP 调用等价物）。



    offset 默认 10（核查/匹配场景足够）；M6 修复后补齐场景传更大值，

    供多天依次补齐时有足够候选，避免候选耗尽导致不同日期重复取用同一批景点。

    """

    return amap_gateway.search_poi(keywords, city=city, citylimit=True, offset=offset)





def _find_match(

    name: str,

    location_xy: Optional[tuple[float, float]],

    pois: list[dict[str, Any]],

) -> Optional[dict[str, Any]]:

    """

    在候选 POI 中找与计划景点匹配的一条。

    匹配规则：名称互相包含（命中），或坐标偏差 ≤ AMAP_MATCH_DIST_M。

    """

    name = (name or "").strip()

    for poi in pois:

        pname = str(poi.get("name") or "").strip()

        if name and pname and (name in pname or pname in name):

            return poi

        ploc = Location.from_text(str(poi.get("location") or ""))

        if location_xy is not None and ploc:

            dist = haversine_m(

                location_xy[0], location_xy[1], ploc.longitude, ploc.latitude

            )

            if dist <= AMAP_MATCH_DIST_M:

                return poi

    return None





def _poi_to_attraction(poi: dict[str, Any], city: str) -> Optional[Attraction]:

    """把高德 POI 转成景点对象；坐标缺失时用地址地理编码补齐，仍缺则返回 None。"""

    name = str(poi.get("name") or "").strip()

    if not name:

        return None

    address = str(poi.get("address") or "").strip() or city

    location = Location.from_text(str(poi.get("location") or ""))

    if location is None:

        geo = amap_gateway.geocode(address, city)

        if geo:

            location = Location(longitude=geo["longitude"], latitude=geo["latitude"])

    if location is None:

        return None

    ptype = str(poi.get("typecode") or poi.get("type") or "")

    return Attraction(

        name=name,

        address=address,

        location=location,

        visit_duration=120,

        description=f"经高德地图核实的真实POI（类型代码:{ptype}）" if ptype else "经高德地图核实的真实POI",

        category="景点",

        poi_id=str(poi.get("id") or ""),

    )





def _verify_attraction(attr: Attraction, city: str) -> tuple[str, Optional[Attraction], list[str]]:

    """

    核查单个景点（**供线程池并发调用，任何情况都不抛异常**）。



    Returns:

        (action, attraction, notes)

        - kept      ：命中且无需修正，原样保留

        - corrected ：命中并用高德坐标修正/补全

        - replaced  ：查无此地，已顶替为高德真实 POI

        - removed   ：查无此地且无法顶替，移除

        - unverified：**核查失败（限流 / 异常）→ 保留原景点，绝不误删真实 POI**

    """

    try:

        pois = _amap_text_search(attr.name, city)

    except Exception as e:  # noqa: BLE001 —— 单景点失败隔离，不影响同批其他景点

        return "unverified", attr, [f"{attr.name}(核查异常:{type(e).__name__})"]



    # 【关键区分】限流/异常导致"没查到" ≠ "查无此地"。

    # 旧逻辑把二者混为一谈，限流时真实景点被当成幻觉移除（trip 评测 4 条全 FAIL 的根因之一）。

    if not pois and amap_gateway.last_outcome() == OUTCOME_RATE_LIMITED:

        return "unverified", attr, [f"{attr.name}(限流未核实)"]



    match = _find_match(attr.name, _loc_of(attr.location), pois)

    if match is not None:

        m_loc = Location.from_text(str(match.get("location") or ""))

        if m_loc is None:

            geo = amap_gateway.geocode(str(match.get("address") or attr.address or attr.name), city)

            if geo:

                m_loc = Location(longitude=geo["longitude"], latitude=geo["latitude"])

        old = _loc_of(attr.location)

        notes: list[str] = []

        if m_loc is not None and old is not None:

            dist = haversine_m(old[0], old[1], m_loc.longitude, m_loc.latitude)

            if dist > AMAP_MATCH_DIST_M:

                notes.append(f"{attr.name}(偏差{dist / 1000:.1f}km→已修正)")

                attr.location = m_loc

                addr = str(match.get("address") or "").strip()

                if addr:

                    attr.address = addr

        elif m_loc is not None and old is None:

            notes.append(f"{attr.name}(补全坐标)")

            attr.location = m_loc

        if not attr.poi_id:

            attr.poi_id = str(match.get("id") or "")

        action = "corrected" if notes else "kept"

        return action, attr, notes



    # 幻觉/无法核实 → 用高德搜索结果顶替

    for p in pois:

        new_attr = _poi_to_attraction(p, city)

        if new_attr:

            return "replaced", new_attr, [f"{attr.name}→{new_attr.name}"]

    return "removed", None, [attr.name]





def _topup_attractions(

    kept: list[Attraction], city: str, need: int, used: Optional[set[str]] = None

) -> list[Attraction]:

    """

    C1-fix + M6（2026-09-09）：当日景点不足时，用高德「景点」通用候选补齐。



    - 只补到恰好满足 need，不覆盖、不重复；

    - M6：`used` 为跨天已用景点名集合，补齐时跳过全程已出现过的景点（避免同一景点多天重复）。

    - 方案A：只取高德"热门"前若干候选（TOPUP_POI_OFFSET，默认 6），宁缺毋滥不取冷门——

      冷门/低质候选独立核查易失败、且会让行程像硬凑；返回不足 need 时由调用方降档为美食/休闲文本。



    Returns:

        实际补齐的景点（可能少于 need——候选耗尽时宁缺毋滥，不重复、不硬凑）。

    """

    added: list[Attraction] = []

    if need <= 0:

        return added

    taken: set[str] = {a.name for a in kept if getattr(a, "name", None)}

    if used:

        taken |= set(used)

    for p in _amap_text_search("景点", city, offset=TOPUP_POI_OFFSET):

        if len(added) >= need:

            break

        cand = _poi_to_attraction(p, city)

        if cand and cand.name not in taken:

            added.append(cand)

            taken.add(cand.name)

    return added





def _has_scenic_candidates(city: str, used: Optional[set[str]] = None) -> bool:

    """

    方案A：判断该地是否还有可用的不同景点候选。

    景点候选不足（无新名字）时，说明该地可看景点本就有限——此时不应硬补，

    应降档为"美食/休闲"文本说明，避免把冷门/低质 POI 硬塞进行程。

    """

    try:

        for p in _amap_text_search("景点", city, offset=TOPUP_POI_OFFSET):

            cand = _poi_to_attraction(p, city)

            if cand and cand.name not in (used or set()):

                return True

        return False

    except Exception:  # noqa: BLE001 —— 探测失败不阻断主流程，按"仍有候选"保守处理

        return True





# 方案A：目的地资源较少时，向该日 description 自然带出的"美食/休闲为主"说明模板

_RELAX_NOTE_POOL = [

    "该地以市井烟火与慢节奏生活著称，可看的集中景点有限；建议当日放慢脚步，以品尝本地小吃、逛特色街区、寻一间地道小馆为主",

    "这里更适合体验式慢游而非赶景点；当日不妨把重心放在当地美食与街巷闲逛上，感受地道生活气息",

    "此目的地景点资源较少但美食与休闲氛围很足，当天行程建议以寻味探店、街角漫步为主，不必强求走景点",

]

_RELAX_NOTE_POOL_IDX = 0





def _append_relax_note(day) -> None:

    """把"美食/休闲为主"的说明追加到该日 description 末尾，供规划器参考。"""

    global _RELAX_NOTE_POOL_IDX

    if not getattr(day, "description", ""):

        return

    note = _RELAX_NOTE_POOL[_RELAX_NOTE_POOL_IDX % len(_RELAX_NOTE_POOL)]

    _RELAX_NOTE_POOL_IDX += 1

    # 避免重复追加

    if note.split("；")[0] not in day.description:

        day.description = f"{day.description}（{note}）"





@step_log("trip_verify_poi")

def verify_plan(plan: TripPlan, city: str) -> TripPlan:

    """

    强约束：每个景点必须能在高德检索到（名称命中或坐标 ≤ 阈值）。

    - 名称命中但坐标偏差 > 阈值的：用高德坐标覆盖计划坐标；

    - 未通过的 POI 用高德搜索结果顶替；顶替失败则移除；

    - **核查失败（限流/异常）≠ 查无此地**：保留原景点并标注「未核实」，绝不误删；

    - 整天空/不足下限时，用高德真实景点候选补位（跨天去重）；

      若该地景点候选本身不足（资源少），方案A 降档为在该日 description 里

      自然带出"美食/休闲/慢节奏"说明，不硬抠冷门/低质景点。



    并发（C1 主体，2026-09-09）：逐景点串行 → `ThreadPoolExecutor`

    （`TRIP_VERIFY_WORKERS`，默认 4），受 `amap_gateway._Throttle` 统一限速。



    核查结论以【数据保障】前缀附到 overall_suggestions 末尾。

    """

    tasks: list[tuple[int, Attraction]] = [

        (di, attr) for di, day in enumerate(plan.days) for attr in day.attractions

    ]

    results: list[Optional[tuple[str, Optional[Attraction], list[str]]]] = [None] * len(tasks)



    if tasks:

        workers = max(1, min(TRIP_VERIFY_WORKERS, len(tasks)))

        if workers == 1:

            for i, (_, attr) in enumerate(tasks):

                results[i] = _verify_attraction(attr, city)

        else:

            with ThreadPoolExecutor(max_workers=workers) as pool:

                futures = {pool.submit(_verify_attraction, attr, city): i for i, (_, attr) in enumerate(tasks)}

                for fut, i in futures.items():

                    try:

                        results[i] = fut.result()

                    except Exception as e:  # noqa: BLE001 —— 并发框架异常也按"未核实"保留

                        attr_i = tasks[i][1]

                        results[i] = ("unverified", attr_i, [f"{attr_i.name}(核查异常:{type(e).__name__})"])



    replaced: list[str] = []

    removed: list[str] = []

    corrected: list[str] = []

    unverified: list[str] = []

    per_day: dict[int, list[Attraction]] = {di: [] for di in range(len(plan.days))}



    for i, (di, _attr) in enumerate(tasks):

        action, kept_attr, notes = results[i] or ("removed", None, [])

        if action == "corrected":

            corrected.extend(notes)

        elif action == "replaced":

            replaced.extend(notes)

        elif action == "removed":

            removed.extend(notes)

        elif action == "unverified":

            unverified.extend(notes)

        if kept_attr is not None:

            per_day.setdefault(di, []).append(kept_attr)



    # M6（2026-09-09）：跨天去重集合——先收集全程已保留的景点名，

    # 后续任何一天补齐时都要避开，杜绝同一景点出现在多个日期。

    used_names: set[str] = set()

    for _kept in per_day.values():

        for a in _kept:

            if getattr(a, "name", None):

                used_names.add(a.name)



    backfilled: list[str] = []

    soft_filled: list[str] = []  # 方案A（2026-09-09）：景点不足时改用"美食/休闲"文本补位，不硬抠冷门景点

    for di, day in enumerate(plan.days):

        kept = per_day.get(di, [])

        was_empty = not kept

        label = day.description or day.date or f"第{day.day_index + 1}天"

        # C1-fix + 方案A（2026-09-09）：当日不足下限时，先尝试用高德真实 POI 补（跨天去重）；

        # 若景点候选本身不足（该地景点资源少），不要硬抠冷门/低质景点（它们独立核查易失败，

        # 且会让行程显得像硬凑）——降档为在该日 description 里自然带出"以美食/休闲/慢节奏为主"，

        # 让规划器把当天重心放在吃与逛，而非硬排景点。

        need = TRIP_MIN_ATTRS_PER_DAY - len(kept)

        if need > 0:

            # 仅当整天为空或极度不足（<=1）且确有剩余景点候选时才补真实景点

            added = _topup_attractions(kept, city, need, used_names) if was_empty or len(kept) <= 1 else []

            if added:

                kept = kept + added

                used_names.update(a.name for a in added if getattr(a, "name", None))

                if was_empty:

                    removed.append(f"{label}:原行程全部无法核实，已回填高德真实POI")

                else:

                    backfilled.append(f"{label}(不足{TRIP_MIN_ATTRS_PER_DAY}个→补齐至{len(kept)}个)")

            elif was_empty and not _has_scenic_candidates(city, used_names):

                # 该地景点候选确实少（拿不到更多不同景点）→ 不硬补，改文本补位

                _append_relax_note(day)

                soft_filled.append(f"{label}(该地景点较少，当日以美食/休闲为主，未硬补景点)")

            elif not was_empty:

                # 还有 1 个真实景点但补不满下限：不再强求第 2 个，避免硬凑

                logger.warning(f"⚠️  第{di + 1}天不足{TRIP_MIN_ATTRS_PER_DAY}个且景点候选有限，保留现有{len(kept)}个真实景点（不硬补冷门）")

        day.attractions = kept



    notes_all: list[str] = []

    if corrected:

        notes_all.append("已用高德坐标修正/补齐偏差过大的景点: " + "; ".join(corrected[:6]))

    if replaced:

        notes_all.append("已替换为高德真实POI: " + "; ".join(replaced[:6]))

    if removed:

        notes_all.append("已移除无法核实的景点: " + "; ".join(removed[:6]))

    if unverified:

        notes_all.append("以下景点因高德限流/异常未能核实（保留未删除，建议稍后重试核实）: " + "; ".join(unverified[:6]))

    if backfilled:

        notes_all.append("已用高德真实POI补齐当日景点下限: " + "; ".join(backfilled[:6]))

    if soft_filled:

        notes_all.append("以下日期当地景点资源较少，已调整为美食/休闲节奏（未硬补景点）: " + "; ".join(soft_filled[:6]))

    if notes_all:

        note = "【数据保障】" + "。".join(notes_all) + "。"

        plan.overall_suggestions = (plan.overall_suggestions or "") + "\n" + note

        logger.info(f"🛡️  {note}")

    return plan





def render_plan_markdown(plan: TripPlan) -> str:

    """把结构化行程渲染成可读 Markdown（供聊天/导出/降级展示）。"""

    lines: list[str] = []

    lines.append(f"# {plan.city} {len(plan.days)}日旅行计划")

    lines.append(f"**日期**：{plan.start_date} 至 {plan.end_date}")

    if plan.overall_suggestions:

        lines.append(f"**总体建议**：{plan.overall_suggestions}\n")

    lines.append("## 每日行程")

    for day in plan.days:

        lines.append(f"### 第{day.day_index + 1}天 {day.date}")

        if day.description:

            lines.append(day.description)

        if day.hotel and day.hotel.name:

            lines.append(f"- 🏨 **住宿**：{day.hotel.name}")

        for attr in day.attractions:

            loc = ""

            if attr.location is not None:

                loc = f"（{attr.location.longitude:.4f},{attr.location.latitude:.4f}）"

            desc = attr.description or ""

            lines.append(f"- 🏞️ **{attr.name}**{loc} — {desc}".rstrip())

        meal_map = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐", "snack": "加餐"}

        for meal in day.meals:

            label = meal_map.get(meal.type, meal.type)

            cost = f"约{meal.estimated_cost}元" if meal.estimated_cost else ""

            lines.append(f"- 🍽️ **{label}**：{meal.name} {cost}".rstrip())

        lines.append("")

    if plan.weather_info:

        lines.append("## 天气参考")

        for w in plan.weather_info:

            lines.append(

                f"- {w.date}：白天{w.day_weather or '-'} {w.day_temp}°C / "

                f"夜间{w.night_weather or '-'} {w.night_temp}°C {w.wind_direction or ''}{w.wind_power or ''}".rstrip()

            )

        lines.append("")

    if plan.routes:

        lines.append("## 往返交通")

        for r in plan.routes:

            line = f"- 【{r.leg}】{r.origin} → {r.destination}"

            if r.mode:

                line += f"（{r.mode}）"

            if r.duration_text:

                line += f"，耗时 {r.duration_text}"

            if r.distance_text:

                line += f"，距离 {r.distance_text}"

            if r.cost_note:

                line += f"；{r.cost_note}"

            lines.append(line)

        lines.append("")



    if plan.budget:

        b = plan.budget

        lines.append(

            "## 预算预估\n"

            f"- 景点门票：¥{b.total_attractions}\n"

            f"- 酒店住宿：¥{b.total_hotels}\n"

            f"- 餐饮费用：¥{b.total_meals}\n"

            f"- 交通费用：¥{b.total_transportation}\n"

            f"- **合计**：¥{b.total}"

        )

    return "\n".join(lines)





@step_log("trip_run_structured")

def plan_once(request: TripRequest) -> TripPlan:

    """

    结构化行程规划一次串联（采集 → 规划 → POI 核查），供节点图与评测共用。

    任何失败显式上抛（HTTP 层转 success=false），不返回假数据。

    """

    req = _coerce_request(request)

    if req is None:

        raise ValueError("行程请求参数不合法")

    pois = collect_pois(req)

    if not pois:

        raise ValueError(f"高德地图未能检索到 {req.city} 的景点数据，请检查城市名或稍后重试")

    weather_rows = collect_weather(req)

    hotels = collect_hotels(req)

    routes = collect_routes(req)

    routes_text = collect_routes_text(routes)

    plan = run_planner(

        req,

        collect_pois_text(pois),

        collect_weather_text(weather_rows),

        collect_hotels_text(hotels),

        routes_text,

    )

    validate_plan(plan, req)

    plan = verify_plan(plan, req.city)

    return plan







