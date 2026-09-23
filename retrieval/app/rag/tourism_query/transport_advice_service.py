"""
交通方式确定性推荐器（规划分支专用）。

背景（2026-09-10 主人测试反馈问题1/3）：
- 旧实现只把「高德路线简报 + 铁路简报」丢给 LLM，让其自由选择交通方式。三条数据源里
  驾车与公交都来自高德、跨城公交只回火车/大巴，铁路又来自 12306 —— 全是"陆路"，
  模型因此**永远锚定在火车**上（"深圳→三亚 20h57m" 仍写公共交通（火车））。
- 用户在追问里明确表达的交通方式（"当然是坐飞机去啊"）也没有被结构化抽取，
  无法贯彻到行程里。

本模块的定位：
- **用户偏好识别**交给 LLM（`plan_extract` 的 transport_preference / transport_avoid
  槽位，只做"理解用户说了什么"）；
- **方式选择与一切数字**全部在代码里确定性完成（阈值规则 + 距离/时长换算），
  不经 LLM，杜绝编造（与油费估算同一思路）。

数据来源（全部来自既有并行工具，不新增外发出口）：
- 高德驾车距离/时长/油费（tool_route，结构化字段）
- 高德跨城公交时长（tool_route.transit_*）
- 12306 车次最短历时与参考票价（tool_rail.trains）

**已知边界（诚实声明，不做假）**：高德与 12306 都**不提供航班数据**，故飞机方案只给
按距离换算的**飞行时间估算**，并明确标注"估算、航班与票价以航司/购票平台为准"；
绝不编造航班号/具体班次（见 itinerary_service._strip_flight_train_numbers）。
"""
from __future__ import annotations

from app.infra.rail_gateway import format_fares
from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.shared.config.common import env_bool, env_float
from app.shared.runtime.logger import logger, step_log

# 推荐器总开关（关掉即不产出【交通方式建议】，行程回到旧行为）
TRANSPORT_ADVICE_ENABLE: bool = env_bool("TRANSPORT_ADVICE_ENABLE", default=True)

# ---- 阈值规则（全部可配，便于按主人实际体验微调）----
# 驾车距离 ≤ 此值 → 视为市内/近郊，优先自驾
TRANSPORT_DRIVE_MAX_KM: float = env_float("TRANSPORT_DRIVE_MAX_KM", default=300.0)
# 铁路最短历时 ≥ 此值 → 长途，推荐飞机（并列铁路）
TRANSPORT_FLIGHT_MIN_RAIL_HOURS: float = env_float("TRANSPORT_FLIGHT_MIN_RAIL_HOURS", default=8.0)
# 驾车距离 ≥ 此值 → 长途，推荐飞机（并列铁路）
TRANSPORT_FLIGHT_MIN_KM: float = env_float("TRANSPORT_FLIGHT_MIN_KM", default=1000.0)
# 飞行估算参数：巡航速度（km/h）与起降/滑行附加时间（小时）
TRANSPORT_FLIGHT_SPEED_KMH: float = env_float("TRANSPORT_FLIGHT_SPEED_KMH", default=750.0)
TRANSPORT_FLIGHT_OVERHEAD_H: float = env_float("TRANSPORT_FLIGHT_OVERHEAD_H", default=0.5)

# 用户偏好抽取的原始问句回退（LLM 槽位缺失时用正则兜底，尽力而为）
_PREF_HINTS: tuple[tuple[str, str], ...] = (
    ("飞机", "飞机"), ("航班", "飞机"), ("飞过去", "飞机"),
    ("高铁", "高铁"), ("动车", "高铁"),
    ("火车", "火车"), ("铁路", "火车"),
    ("自驾", "自驾"), ("开车", "自驾"), ("驾车", "自驾"),
    ("大巴", "大巴"), ("长途车", "大巴"),
    ("轮渡", "轮渡"), ("坐船", "轮渡"),
)
# 强偏好句式（"当然是坐飞机""还是坐高铁""改坐飞机"）：用户对方式的最终诉求
_STRONG_PREF_PATTERN_HINTS = ("当然是", "还是坐", "要坐", "改成", "换成", "不如", "宁可")


def _rail_min_hours_and_fare(rail: dict) -> tuple[float | None, str | None, str | None]:
    """
    取 12306 车次里的「最短历时（小时）」、「参考票价文本」、「对应车次号」。

    注意：12306-MCP 的 `duration` 是 `"HH:MM"` 文本（见 rail_gateway.format_duration），
    不是秒数，必须按冒号解析。

    Returns:
        (hours, fare_text, code)；无可用数据时对应项为 None。
    """
    trains = rail.get("trains") or []
    best: tuple[int, dict] | None = None
    for train in trains:
        text = str(train.get("duration") or "").strip()
        if ":" not in text:
            continue
        try:
            hours_s, _, minutes_s = text.partition(":")
            minutes_total = int(hours_s) * 60 + int(minutes_s)
        except (TypeError, ValueError):
            continue
        if minutes_total <= 0:
            continue
        if best is None or minutes_total < best[0]:
            best = (minutes_total, train)
    if best is None:
        return None, None, None
    minutes_total, train = best
    hours = minutes_total / 60.0
    fare_text = format_fares(train.get("prices")) or None
    code = str(train.get("code") or "").strip() or None
    return hours, fare_text, code


def _format_hours(hours: float) -> str:
    """小时数 → 中文时长（如"19小时8分钟"）。"""
    if hours is None:
        return ""
    total = int(round(hours * 60))
    h, m = divmod(total, 60)
    if h and m:
        return f"{h}小时{m}分钟"
    if h:
        return f"{h}小时"
    return f"{m}分钟"


def _regex_preference(query: str) -> str:
    """
    正则兜底抽取用户交通偏好（LLM 槽位为空时使用）。

    规则：优先取强偏好句式（"当然是坐飞机"）里出现的方式；否则取问句中唯一出现的方式。
    同时出现多种方式（如"坐火车，你怎么想的？当然是坐飞机"）→ 强偏好句式胜出。
    """
    if not query:
        return ""
    strong_hit = ""
    for hint in _STRONG_PREF_PATTERN_HINTS:
        idx = query.find(hint)
        if idx < 0:
            continue
        tail = query[idx : idx + 12]
        for word, mode in _PREF_HINTS:
            if word in tail and not strong_hit:
                strong_hit = mode
                break
    if strong_hit:
        return strong_hit
    found = {mode for word, mode in _PREF_HINTS if word in query}
    return next(iter(found)) if len(found) == 1 else ""


@step_log("decide_transport")
def decide_transport(state: dict) -> dict:
    """
    交通方式建议主入口（纯确定性，无外发请求）。

    Returns:
        dict: {"ok","specified","avoid","recommended","alternatives","options","reason","text"}
        ok=False 表示本次无法给出建议（未解析出跨城两地 / 推荐器被关闭）。
    """
    empty = {
        "ok": False, "specified": "", "avoid": "", "recommended": "",
        "alternatives": [], "options": [], "reason": "", "text": "",
    }
    if not TRANSPORT_ADVICE_ENABLE:
        return empty

    try:
        info = extract_destination_info(state) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"交通方式推荐:行程信息解析失败,跳过,错误信息:{str(e)}")
        return empty

    origin = str(info.get("origin") or "").strip()
    destination = str(info.get("destination") or "").strip()
    # 只对"两点间跨城"有意义：缺出发地（"我在成都怎么玩"式单程抵达/市内类）不推荐跨城方式
    if not origin or not destination or origin == destination:
        logger.info(
            f"交通方式推荐跳过：出发地/目的地不完整（origin={origin!r} destination={destination!r}）"
        )
        return empty

    route = state.get("tool_route") or {}
    rail = state.get("tool_rail") or {}
    distance_km = route.get("distance_km")
    try:
        distance_km = float(distance_km) if distance_km is not None else None
    except (TypeError, ValueError):
        distance_km = None
    transit_seconds = route.get("transit_duration_seconds")
    try:
        transit_hours = float(transit_seconds) / 3600.0 if transit_seconds else None
    except (TypeError, ValueError):
        transit_hours = None
    rail_hours, rail_fare, rail_code = _rail_min_hours_and_fare(rail)

    # 用户指定方式：LLM 槽位优先，缺失时正则兜底
    query = f"{state.get('original_query') or ''} {state.get('rewritten_query') or ''}"
    specified = str(info.get("transport_preference") or "").strip() or _regex_preference(query)
    avoid = str(info.get("transport_avoid") or "").strip()
    if specified and specified == avoid:
        # 自相矛盾（抽错）→ 按未指定处理，避免把用户排除的方式当诉求
        specified, avoid = "", ""

    recommended = specified
    reason = ""
    alternatives: list[str] = []

    if recommended:
        reason = f"用户已明确指定{recommended}，按用户意愿执行"
    elif distance_km is not None and distance_km <= TRANSPORT_DRIVE_MAX_KM:
        recommended = "自驾"
        alternatives = ["高铁", "火车"] if rail_hours else []
        reason = f"两地约 {distance_km:.0f} 公里，属近程，自驾更灵活"
    elif (rail_hours is not None and rail_hours >= TRANSPORT_FLIGHT_MIN_RAIL_HOURS) or (
        distance_km is not None and distance_km >= TRANSPORT_FLIGHT_MIN_KM
    ):
        recommended = "飞机"
        alternatives = ["高铁", "火车"] if rail_hours else ["自驾"]
        if rail_hours is not None:
            reason = f"铁路最快需约 {_format_hours(rail_hours)}，长途飞行更省时"
        else:
            reason = f"两地约 {(distance_km or 0):.0f} 公里，属长途，飞行更省时"
    elif rail_hours is not None:
        recommended = "高铁"
        alternatives = ["火车", "自驾"] if distance_km else ["火车"]
        reason = f"铁路最快约 {_format_hours(rail_hours)}，性价比较优"
    elif distance_km is not None:
        recommended = "自驾"
        reason = f"两地约 {distance_km:.0f} 公里，自驾直达"
    else:
        # 既无铁路也无路线数据 → 不臆断，交给 LLM 按【参考内容】处理
        logger.info("交通方式推荐：无路线/铁路结构化数据，跳过推荐")
        return empty

    # 用户排除的方式不得出现在推荐与备选里
    alternatives = [x for x in alternatives if x != avoid and x != recommended]

    # ---- 各方案明细（只写有数据支撑的数字，飞机时间明确标"估算"）----
    options: list[dict] = []
    if recommended == "飞机" or "飞机" in alternatives:
        air_hours = None
        if distance_km:
            air_hours = distance_km / max(1.0, TRANSPORT_FLIGHT_SPEED_KMH) + TRANSPORT_FLIGHT_OVERHEAD_H
        line = "- 飞机：空中飞行约 " + (_format_hours(air_hours) if air_hours else "以实际航班为准")
        line += "（按距离估算，含起降；不含两端机场往返）；航班与票价请以航司/购票平台为准"
        options.append({"mode": "飞机", "line": line, "estimated": True})
    if recommended in ("高铁", "火车") or any(x in alternatives for x in ("高铁", "火车")):
        line = "- 铁路："
        if rail_hours is not None:
            line += f"最快约 {_format_hours(rail_hours)}"
            if rail_code:
                line += f"（参考车次 {rail_code}）"
            if rail_fare:
                line += f"，参考票价 {rail_fare}"
            line += "；票价与班次以 12306 官方为准"
        else:
            line += "本次未取到车次数据，班次与票价请以 12306 官方为准"
        options.append({"mode": "铁路", "line": line, "estimated": False})
    if recommended == "自驾" or "自驾" in alternatives:
        line = "- 自驾："
        if distance_km:
            line += f"全程约 {distance_km:.0f} 公里"
            if route.get("duration_seconds"):
                line += f"，驾车约 {_format_hours(float(route['duration_seconds']) / 3600.0)}"
            if route.get("fuel_cost"):
                line += f"，油费约 {float(route['fuel_cost']):.0f} 元（按百公里油耗与油价估算）"
            line += "；过路费与停车费以实际为准"
        else:
            line += "距离数据缺失，请以地图导航为准"
        options.append({"mode": "自驾", "line": line, "estimated": False})
    if transit_hours and recommended not in ("飞机",):
        options.append({
            "mode": "公共交通",
            "line": f"- 公共交通（{route.get('transit_summary') or '综合方案'}）："
                    f"全程约 {_format_hours(transit_hours)}；票价以官方渠道为准",
            "estimated": False,
        })

    header = f"推荐交通方式：{recommended}（{reason}）"
    if specified:
        header = f"用户指定交通方式：{recommended}（严格按指定方式规划，其余方式一律不写）"
    text = header
    if alternatives:
        text += f"\n备选：{'、'.join(alternatives)}"
    if options:
        text += "\n可选方案明细：\n" + "\n".join(o["line"] for o in options)

    logger.info(
        f"交通方式建议：{origin}→{destination} 推荐={recommended} "
        f"指定={specified or '无'} 排除={avoid or '无'} "
        f"距离={distance_km}km 铁路={rail_hours}h"
    )
    return {
        "ok": True,
        "specified": specified,
        "avoid": avoid,
        "recommended": recommended,
        "alternatives": alternatives,
        "options": options,
        "reason": reason,
        "text": text,
    }
