"""
天气工具服务模块（对话流侧）。

链路：LLM 解析目的地（名称+所属城市+粗略经纬度）→ weather_gateway 定位并取数
      → 组装中文天气简报，写入 state["tool_weather"] 供答案生成阶段引用。

数据源：统一由 app.infra.weather_gateway 负责（默认主和风、备高德，输出已归一化）。
本模块不再直接发起任何天气 HTTP 请求，只负责「解析 → 取数 → 简报文本」，
以此消除此前「对话流走和风 / 行程走高德」的双源口径分裂。

降级策略：网关关闭、无目的地、定位失败、取数失败均返回 ok=False 的空简报，
不阻断主检索链路——天气属于增强信息，缺了只影响规划类回答的天气部分。
"""
import json
import re

from app.infra.llm import llm_provider
from app.infra.weather_gateway import (
    _CHINA_LAT_RANGE,
    _CHINA_LON_RANGE,
    weather_gateway,
)
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.load_prompt import load_prompt
from app.infra.amap_gateway import amap_gateway
from app.shared.runtime.place_utils import _is_generic_place, _city_from_address
from app.shared.runtime.logger import logger, step_log

# re-export：route_tool_service 从本模块导入这两个常量，保持其导入路径不变
__all__ = [
    "_CHINA_LAT_RANGE",
    "_CHINA_LON_RANGE",
    "extract_destination_info",
    "get_weather_brief",
]


def _parse_llm_json(raw: str) -> dict:
    """解析模型 JSON 输出，兼容 ```json 围栏包裹的情况；失败返回空 dict。"""
    raw = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# 合法交通方式枚举（与 transport_advice_service 共用口径；"空" 表示用户未表达）
_TRANSPORT_MODES = ("飞机", "火车", "高铁", "自驾", "大巴", "轮渡")


def _norm_transport(value) -> str:
    """把 LLM 抽取的交通方式归一到枚举值；无法识别一律返回空串（宁缺勿错）。"""
    text = str(value or "").strip()
    if not text or text in ("空", "无", "None", "null"):
        return ""
    for mode in _TRANSPORT_MODES:
        if mode in text:
            return mode
    # 常见近义写法归一
    if "航班" in text or "飞" in text:
        return "飞机"
    if "动车" in text or "铁路" in text:
        return "高铁"
    if "开车" in text or "驾车" in text:
        return "自驾"
    return ""


def _normalize_generic_destination(info: dict) -> dict:
    """OBS-4 治本：destination 为泛称（市区/市中心/城区…）时归一为具体城市。

    - city 已抽取 → destination 直接用 city；
    - city 空但有 origin → 地理编码 origin 反解城市，回填 city 并归一 destination；
    - 都无解 → 保持原样（由天气网关守卫 / 行程闸门兜底，绝不盲信泛称）。
    """
    dest = (info.get("destination") or "").strip()
    if not _is_generic_place(dest):
        return info
    city = (info.get("city") or "").strip()
    if city:
        info["destination"] = city
        logger.info(f"抽取归一[OBS-4]:泛称目的地[{dest}]→使用城市[{city}]")
        return info
    origin = (info.get("origin") or "").strip()
    if origin:
        try:
            geo = amap_gateway.geocode(origin)
            if geo and geo.get("formatted_address"):
                resolved = _city_from_address(geo["formatted_address"])
                if resolved:
                    info["city"] = resolved
                    info["destination"] = resolved
                    logger.info(
                        f"抽取归一[OBS-4]:泛称目的地[{dest}]由出发地[{origin}]反解城市[{resolved}]"
                    )
                    return info
        except Exception as e:  # noqa: BLE001
            logger.warning(f"抽取归一[OBS-4]:出发地地理编码失败[{origin}],错误信息:{e}")
    return info


@step_log("extract_destination_info")
def extract_destination_info(state: dict) -> dict:
    """
    LLM 解析出行信息：出发地、目的地、所属城市/州、粗略经纬度（json_mode + 进程内缓存）。
    天气工具与路线工具共用本函数与缓存，同问题只调用一次模型。
    经纬度仅在和风/高德定位都未命中时兜底使用。

    Returns:
        dict: {"origin","destination","city","latitude","longitude"}；解析失败字段为空。
    """
    original = str(state.get("original_query") or "").strip()
    rewritten = str(state.get("rewritten_query") or "").strip()
    empty = {
        "origin": "", "destination": "", "city": "", "latitude": None, "longitude": None,
        "travel_days": 0, "transport_preference": "", "transport_avoid": "",
    }
    if not (original or rewritten):
        return empty

    def _produce_once(question: str, reference: str) -> dict:
        try:
            client = llm_provider.chat(json_mode=True)
            prompt = load_prompt(
                "tourism/plan_extract",
                question=question,
                rewritten=reference or "（无）",
            )
            data = _parse_llm_json(client.invoke(prompt).content)
            if not data:
                return dict(empty)
            try:
                lat = float(data.get("latitude"))
                lon = float(data.get("longitude"))
            except (TypeError, ValueError):
                lat = lon = None
            try:
                days = int(float(data.get("travel_days") or 0))
            except (TypeError, ValueError):
                days = 0
            return {
                "origin": str(data.get("origin") or "").strip(),
                "destination": str(data.get("destination") or "").strip(),
                "city": str(data.get("city") or "").strip(),
                "latitude": lat,
                "longitude": lon,
                # 用户明说的天数（0=没说）：预算估算器用它算住宿/餐饮，缺失时取默认 2 天
                "travel_days": max(0, min(days, 30)),
                # 交通方式偏好/排除项：由 LLM 只做「理解用户说了什么」，推荐决策交给
                # transport_advice_service 的确定性规则（见该模块头注释）
                "transport_preference": _norm_transport(data.get("transport_preference")),
                "transport_avoid": _norm_transport(data.get("transport_avoid")),
            }
        except Exception as e:
            logger.warning(f"行程信息解析失败,降级为空结构,错误信息:{str(e)}")
            return dict(empty)

    def _extract(question: str, reference: str) -> dict:
        # 缓存命中返回的是同一对象引用，这里必须复制后再用：
        # 下方会用改写句结果回填字段，直接改会污染缓存条目。
        result = cached_invoke(
            namespace="plan_weather_extract",
            cache_parts=(question, reference),
            producer=lambda: _produce_once(question, reference),
            cache_label="destination",
            semantic=True,
        )
        return dict(result) if isinstance(result, dict) else dict(empty)

    # 第一轮：以原始问题为准（首轮提问的出发地/目的地最完整）；改写句仅作指代消解参考
    info = _extract(original or rewritten, rewritten if original else "")

    # 第二轮兜底：延续类追问（"我想坐飞机去"）原始问句里根本没有地名，
    # 改写句才是补全后的完整问题。此前缺这一步导致 destination 解析为空、
    # 天气/路线工具被静默跳过（日志"未从问题中解析到目的地"），行程只能靠模型编造目的地。
    if not info.get("destination") and rewritten and rewritten != (original or rewritten):
        fallback = _extract(rewritten, original)
        for key in ("origin", "destination", "city", "transport_preference", "transport_avoid"):
            if not info.get(key) and fallback.get(key):
                info[key] = fallback[key]
        if not info.get("travel_days") and fallback.get("travel_days"):
            info["travel_days"] = fallback["travel_days"]
        if info.get("destination") and info.get("latitude") is None:
            info["latitude"] = fallback.get("latitude")
            info["longitude"] = fallback.get("longitude")
        if info.get("destination"):
            logger.info(f"行程信息解析兜底成功,改用改写句解析结果: {info.get('destination')}")
    info = _normalize_generic_destination(info)
    return info


@step_log("format_weather_text")
def format_weather_text(location: dict, days: list[dict]) -> str:
    """把定位结果与网关归一化后的逐日预报组装成给 LLM 看的中文简报。"""
    admin = f"（{location['admin']}）" if location.get("admin") else ""
    lines = [f"目的地:{location['name']}{admin}"]
    for day in days:
        t_max = day.get("temp_max") or "?"
        t_min = day.get("temp_min") or "?"
        precip = day.get("precip")
        # 高德源无降水量（网关置 None），此时不展示该字段，禁止编造
        precip_text = f",降水量{precip}mm" if precip not in (None, "", "0", "0.0") else ""
        wind = day.get("wind") or ""
        wind_text = f",{wind}" if wind else ""
        lines.append(f"{day['date']}:{day['desc']},{t_min}~{t_max}°C{precip_text}{wind_text}")
    return "\n".join(lines)


@step_log("get_weather_brief")
def get_weather_brief(state: dict) -> dict:
    """
    天气工具主入口：解析目的地 → 网关定位取数（主和风/备高德）→ 简报。
    网关关闭、无目的地、定位或取数失败均返回 ok=False，不抛异常（节点侧兜底）。

    Returns:
        dict: {"ok","destination","text"}；ok=True 时 text 为中文天气简报。
    """
    empty = {"ok": False, "destination": "", "text": ""}
    info = extract_destination_info(state)
    if not info.get("destination"):
        logger.info("天气工具:未从问题中解析到目的地,跳过天气查询")
        return empty
    try:
        result = weather_gateway.forecast(
            destination=info["destination"],
            city=info.get("city") or "",
            latitude=info.get("latitude"),
            longitude=info.get("longitude"),
        )
        if not result.get("ok"):
            logger.info(f"天气工具:查询失败[{info['destination']}],原因:{result.get('error')}")
            return empty
        loc = {"name": result.get("name") or info["destination"], "admin": result.get("admin") or ""}
        logger.info(
            f"天气工具:定位成功[{loc['name']}]({loc['admin']}) "
            f"数据源={result.get('source')}{'(主源失败已降级)' if result.get('degraded') else ''} "
            f"定位源={result.get('loc_source')} 天数={len(result.get('days') or [])}"
        )
        weather_text = format_weather_text(loc, result.get("days") or [])
        # 出发地天气（跨城出行时出发地的台风/暴雨同样影响航班与交通；失败不影响目的地简报）
        origin = (info.get("origin") or "").strip()
        if origin and origin not in (info["destination"], loc["name"]):
            try:
                origin_res = weather_gateway.forecast(destination=origin, city=origin)
                if origin_res.get("ok"):
                    origin_text = format_weather_text(
                        {"name": origin_res.get("name") or origin, "admin": origin_res.get("admin") or ""},
                        origin_res.get("days") or [],
                    )
                    weather_text = f"{weather_text}\n\n{origin_text}"
            except Exception as e:
                logger.info(f"天气工具:出发地天气查询失败,仅提供目的地天气,错误信息:{str(e)}")
        return {
            "ok": True,
            "destination": info["destination"],
            "text": weather_text,
        }
    except Exception as e:
        logger.warning(f"天气工具:查询失败,降级为空简报,错误信息:{str(e)}")
        return empty
