"""
路线规划工具服务模块（数据源：高德开放平台 Web服务 API，需 .env 配置 AMAP_API_KEY）。

链路：复用行程信息解析（出发地/目的地，与天气工具共用一次 LLM 调用+缓存）
      → 高德地理编码（两地 → 经纬度）→ 高德驾车路径规划 v3/direction/driving
      → 组装中文路线简报，写入 state["tool_route"] 供答案生成阶段引用。

目的地编码未命中时用 LLM 粗略坐标兜底；出发地无兜底（编码失败即放弃，避免报错路线）。
降级策略：未配置 Key、未提及出发地、编码/请求失败均返回 ok=False 空简报，不阻断主链路。
"""
import httpx

from app.rag.common.place_name_utils import strip_admin_suffix
from app.rag.tourism_query.weather_tool_service import (
    _CHINA_LAT_RANGE,
    _CHINA_LON_RANGE,
    extract_destination_info,
)
from app.shared.config.common import env_bool, env_int, env_str
from app.shared.runtime.logger import logger, step_log

# 路线工具总开关
ROUTE_TOOL_ENABLE: bool = env_bool("ROUTE_TOOL_ENABLE", default=True)
# 高德 Key 与请求超时（秒），与 .env 共用
AMAP_API_KEY: str = env_str("AMAP_API_KEY", default="")
AMAP_TIMEOUT: int = env_int("AMAP_TIMEOUT", default=6)


def _env_float(name: str, default: float) -> float:
    """读取浮点型环境变量，非法值回退默认。"""
    try:
        return float(env_str(name, default=str(default)))
    except (TypeError, ValueError):
        return default


# 油费估算参数（确定性计算在代码层完成，不经 LLM，防编造）
PLAN_FUEL_CONSUMPTION: float = _env_float("PLAN_FUEL_CONSUMPTION", default=8.0)   # 百公里油耗（升）
PLAN_FUEL_PRICE: float = _env_float("PLAN_FUEL_PRICE", default=7.8)               # 油价（元/升）

_AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
_AMAP_DRIVE_URL = "https://restapi.amap.com/v3/direction/driving"
# 跨城公交综合规划（返回火车/大巴/市内公交组合方案，跨城场景由 city+cityd 指定两地）
_AMAP_TRANSIT_URL = "https://restapi.amap.com/v3/direction/transit/integrated"


@step_log("amap_geocode")
def amap_geocode(address: str) -> tuple[float, float] | None:
    """
    高德地理编码：中文地址 → (纬度, 经度)。

    Returns:
        tuple | None: (纬度, 经度)；未命中/未配置/异常返回 None。
    """
    if not AMAP_API_KEY:
        return None
    try:
        params = {"address": address, "key": AMAP_API_KEY}
        with httpx.Client(timeout=AMAP_TIMEOUT) as client:
            resp = client.get(_AMAP_GEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        geocodes = data.get("geocodes") or []
        if data.get("status") != "1" or not geocodes:
            return None
        # 高德 location 格式为 "经度,纬度"
        lon_text, lat_text = str(geocodes[0].get("location", "")).split(",")
        return float(lat_text), float(lon_text)
    except Exception as e:
        logger.warning(f"高德地理编码失败[{address}],错误信息:{str(e)}")
        return None


@step_log("fetch_transit_route")
def fetch_transit_route(
    origin_geo: tuple[float, float],
    dest_geo: tuple[float, float],
    origin_city: str,
    dest_city: str,
) -> dict | None:
    """
    高德跨城公交综合规划：起点/终点 (纬度, 经度) + 两地城市名 → 最优方案的 {duration_seconds, summary}。
    跨城方案一般含火车（railway）/大巴 + 市内公交/步行接驳。

    Returns:
        dict | None: {"duration_seconds","summary"}；无可用方案返回 None。
    """
    params = {
        "origin": f"{origin_geo[1]:.6f},{origin_geo[0]:.6f}",        # 高德格式：经度,纬度
        "destination": f"{dest_geo[1]:.6f},{dest_geo[0]:.6f}",
        "city": origin_city,
        "cityd": dest_city,
        "strategy": "0",  # 推荐模式
        "key": AMAP_API_KEY,
    }
    with httpx.Client(timeout=AMAP_TIMEOUT) as client:
        resp = client.get(_AMAP_TRANSIT_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"高德公交规划返回异常:{data.get('info')}")
    transits = (data.get("route") or {}).get("transits") or []
    if not transits:
        return None
    # 取耗时最短的方案
    best = min(
        (t for t in transits if t.get("duration")),
        key=lambda t: int(t.get("duration", 0)),
        default=None,
    )
    if not best:
        return None
    # 从 segments 提取主要交通方式概述（如"火车+地铁"）
    modes: list[str] = []
    for segment in best.get("segments") or []:
        if segment.get("railway"):
            modes.append("火车")
        elif segment.get("bus"):
            bus_lines = (segment.get("bus") or {}).get("buslines") or []
            name = (bus_lines[0].get("name") or "") if bus_lines else ""
            # 长途大巴/机场巴士等含"班线/机场"字样归为大巴，其余归市内公交
            modes.append("大巴" if any(k in name for k in ("班线", "机场", "客运")) else "公交")
        elif segment.get("taxi"):
            modes.append("出租")
    summary = "+".join(dict.fromkeys(modes)) or "综合方案"
    return {"duration_seconds": best.get("duration"), "summary": summary}


@step_log("format_duration")
def format_duration(seconds) -> str:
    """秒数转中文时长（如"5小时30分钟"/"40分钟"），异常输入返回空串。"""
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return ""
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}小时{minutes}分钟" if minutes else f"{hours}小时"
    return f"{minutes}分钟"


@step_log("fetch_driving_route")
def fetch_driving_route(origin: tuple[float, float], destination: tuple[float, float]) -> dict | None:
    """
    高德驾车路径规划：起点/终点 (纬度, 经度) → 首选路径的 {distance_km, duration_seconds}。

    Returns:
        dict | None: {"distance_km","duration_seconds"}；失败返回 None。
    """
    params = {
        "origin": f"{origin[1]:.6f},{origin[0]:.6f}",        # 高德格式：经度,纬度
        "destination": f"{destination[1]:.6f},{destination[0]:.6f}",
        "extensions": "base",
        "key": AMAP_API_KEY,
    }
    with httpx.Client(timeout=AMAP_TIMEOUT) as client:
        resp = client.get(_AMAP_DRIVE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"高德路径规划返回异常:{data.get('info')}")
    paths = (data.get("route") or {}).get("paths") or []
    if not paths:
        return None
    return {
        "distance_km": int(paths[0].get("distance", 0)) / 1000,
        "duration_seconds": paths[0].get("duration"),
    }


@step_log("format_route_text")
def format_route_text(origin: str, destination: str, route: dict, transit: dict | None = None) -> str:
    """把驾车/公交路径结果组装成给 LLM 看的中文简报（含代码层确定性计算的油费估算）。"""
    duration = format_duration(route.get("duration_seconds")) or "未知"
    text = f"{origin}→{destination}:全程约{route['distance_km']:.0f}公里"
    if route.get("distance_km", 0) > 0:
        fuel_cost = route["distance_km"] / 100 * PLAN_FUEL_CONSUMPTION * PLAN_FUEL_PRICE
        text += (
            f",驾车约{duration},预计油费约{fuel_cost:.0f}元"
            f"（按百公里{PLAN_FUEL_CONSUMPTION:g}L、油价{PLAN_FUEL_PRICE:g}元/L估算）"
        )
    # 公共交通方案（有数据时给出对比项；票价无常口数据，标注以官方为准）
    if transit:
        transit_duration = format_duration(transit.get("duration_seconds")) or "未知"
        text += f"；公共交通（{transit.get('summary', '')}）约{transit_duration}，票价以12306/官方渠道为准"
    return text


@step_log("plan_route")
def plan_route(state: dict) -> dict:
    """
    路线工具主入口：解析出发地/目的地 → 高德编码 → 驾车规划 → 简报。
    任一步失败或被关闭均返回 ok=False，不抛异常（节点侧兜底）。

    Returns:
        dict: {"ok","origin","destination","text"}；ok=True 时 text 为中文路线简报。
    """
    empty = {
        "ok": False, "origin": "", "destination": "", "text": "",
        "distance_km": None, "duration_seconds": None, "fuel_cost": None,
        "transit_duration_seconds": None, "transit_summary": "",
    }
    if not ROUTE_TOOL_ENABLE:
        return empty
    if not AMAP_API_KEY:
        logger.warning("路线工具:未配置 AMAP_API_KEY,路线规划停用(请在 .env 填写)")
        return empty
    info = extract_destination_info(state)
    origin = info.get("origin", "")
    destination = info.get("destination", "")
    if not destination:
        logger.info("路线工具:未解析到目的地,跳过路线规划")
        return empty
    if not origin:
        logger.info("路线工具:问题未提及出发地,跳过路线规划")
        return empty
    try:
        origin_geo = amap_geocode(origin)
        if not origin_geo:
            logger.info(f"路线工具:出发地编码失败[{origin}],跳过路线规划")
            return empty
        dest_geo = amap_geocode(destination)
        if dest_geo is None:
            # 目的地编码未命中时用 LLM 粗略坐标兜底（校验中国范围）
            lat, lon = info.get("latitude"), info.get("longitude")
            if (
                lat is None or lon is None
                or not (_CHINA_LAT_RANGE[0] <= lat <= _CHINA_LAT_RANGE[1])
                or not (_CHINA_LON_RANGE[0] <= lon <= _CHINA_LON_RANGE[1])
            ):
                logger.info(f"路线工具:目的地编码失败且无可用兜底坐标[{destination}],跳过路线规划")
                return empty
            dest_geo = (lat, lon)
        route = fetch_driving_route(origin_geo, dest_geo)
        if not route:
            return empty
        # 公共交通方案（独立尝试，失败不影响驾车数据——部分景区/跨城场景可能无公交方案）
        transit = None
        try:
            dest_city = (info.get("city") or destination).strip()
            transit = fetch_transit_route(origin_geo, dest_geo, origin, dest_city)
        except Exception as e:
            logger.info(f"路线工具:公交规划不可用,仅提供驾车方案,原因:{str(e)}")
        # 代码层确定性油费（与 format_route_text 同口径）：交通方式推荐器据此对比各方案成本
        distance_km = float(route.get("distance_km") or 0)
        fuel_cost = (
            distance_km / 100 * PLAN_FUEL_CONSUMPTION * PLAN_FUEL_PRICE
            if distance_km > 0
            else 0.0
        )
        logger.info(f"路线工具:规划成功 {origin}→{destination}")
        return {
            "ok": True,
            "origin": origin,
            "destination": destination,
            "text": format_route_text(origin, destination, route, transit),
            # 结构化字段（transport_advice_service 消费；不改变原 text 简报）
            "distance_km": round(distance_km, 1),
            "duration_seconds": route.get("duration_seconds"),
            "fuel_cost": round(fuel_cost, 0),
            "transit_duration_seconds": (transit or {}).get("duration_seconds"),
            "transit_summary": (transit or {}).get("summary", ""),
        }
    except Exception as e:
        logger.warning(f"路线工具:规划失败,降级为空简报,错误信息:{str(e)}")
        return empty
