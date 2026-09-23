"""
花销估算服务（规划类问题专用，2026-09-10 主人反馈问题2）。

背景：旧实现受 itinerary prompt 规则5「除油费外禁止任何数字」约束，花销预估四段
全部退化成"以 12306/官方/现场为准"，主人读起来像"没调用工具"。实际上项目里
**已有真实数据源**，只是没被组织进预算：
- 交通：12306 真实席别票价（`tool_rail`）、高德驾车距离×油耗油价（`tool_route.fuel_cost`）；
- 住宿/餐饮：高德 POI 的人均消费（`tool_stay_food`，`biz_ext.cost`）。

主人拍板口径（2026-09-10）＝「真实优先 + 缺失估算」：
1. 有真实数据 → 直接引用，并标注来源（12306 / 高德）；
2. 真实数据缺失 → 用**确定性公式**估算（默认单价 × 天数/晚数），并**必须**标注"预估"；
3. 门票高德不提供 → 一律写"以景区实际收费为准"，不估。

本模块所有数字均在代码层确定性计算，不经 LLM（与油费估算同一思路），
杜绝编造；对外文本逐项带来源标注，供 itinerary prompt 原样引用。
"""
from __future__ import annotations

from statistics import median

from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.shared.config.common import env_bool, env_float, env_int
from app.shared.runtime.logger import logger, step_log

BUDGET_TOOL_ENABLE: bool = env_bool("BUDGET_TOOL_ENABLE", default=True)
# 用户未说天数时的默认行程天数（与 itinerary prompt 的"未指定按 2 天 1 晚"口径一致）
BUDGET_DEFAULT_DAYS: int = env_int("BUDGET_DEFAULT_DAYS", default=2)
# 餐饮：每日餐数与缺省单餐人均（真实数据缺失时使用）
BUDGET_MEALS_PER_DAY: int = env_int("BUDGET_MEALS_PER_DAY", default=3)
BUDGET_MEAL_DEFAULT_PER_MEAL: float = env_float("BUDGET_MEAL_DEFAULT_PER_MEAL", default=60.0)
# 住宿：缺省每晚人均（真实数据缺失时使用）
BUDGET_HOTEL_DEFAULT_PER_NIGHT: float = env_float("BUDGET_HOTEL_DEFAULT_PER_NIGHT", default=400.0)


def _empty() -> dict:
    """统一的失败结构：ok=False + 空明细 + 空简报。"""
    return {"ok": False, "days": 0, "items": [], "total": None, "text": ""}


def _to_float(value) -> float | None:
    """把高德/12306 的字符串数字转 float；非法返回 None（绝不把非数字当 0）。"""
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _median_cost(pois: list[dict]) -> float | None:
    """取一组 POI 的人均消费中位数（高德缺失时返回 None，不做 0 兜底）。"""
    values: list[float] = []
    for poi in pois or []:
        cost = _to_float((poi.get("biz_ext") or {}).get("cost"))
        if cost is not None and cost > 0:
            values.append(cost)
    return float(median(values)) if values else None


def _cheapest_rail_fare(rail: dict) -> tuple[str, float] | None:
    """取 12306 车次里最便宜的席别票价 → (席别名, 金额)；无票价返回 None。"""
    best: tuple[str, float] | None = None
    for train in rail.get("trains") or []:
        for seat, price in (train.get("prices") or {}).items():
            amount = _to_float(price)
            if amount is None or amount <= 0:
                continue
            if best is None or amount < best[1]:
                best = (str(seat), amount)
    return best


def _money(value: float) -> str:
    """金额格式化（整数不带小数点）。"""
    return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"


def _transport_item(state: dict, transport: dict) -> tuple[str, str, bool]:
    """
    交通项：返回 (文本, 来源标注, 是否预估)。

    优先级：用户/推荐方式为飞机 → 无真实票价（高德与 12306 均无航班数据）；
    铁路 → 12306 真实最低票价；自驾 → 代码层油费估算。
    """
    route = state.get("tool_route") or {}
    rail = state.get("tool_rail") or {}
    mode = str(transport.get("recommended") or "")

    if mode == "飞机":
        return "机票价格随航司与购票时间浮动，请以航空公司/购票平台实际报价为准", "需自行查询", False

    if mode in ("高铁", "火车") or (not mode and rail.get("ok")):
        fare = _cheapest_rail_fare(rail)
        if fare:
            seat, amount = fare
            return f"{seat}约 {_money(amount)} 元（单程人均）", "12306 真实票价", False
        return "班次与票价请以 12306 官方为准", "无数据", False

    if mode == "自驾":
        fuel = _to_float(route.get("fuel_cost"))
        if fuel:
            return (
                f"油费约 {_money(fuel)} 元（按高德全程距离与百公里油耗/油价估算，"
                "过路费与停车费另计）",
                "高德距离 + 代码估算油费",
                True,
            )
        return "自驾费用请以地图导航与加油站实际为准", "无数据", False

    # 无推荐（缺路线/铁路数据）时的兜底：有什么写什么
    fare = _cheapest_rail_fare(rail)
    if fare:
        return f"{fare[0]}约 {_money(fare[1])} 元（单程人均）", "12306 真实票价", False
    fuel = _to_float(route.get("fuel_cost"))
    if fuel:
        return f"油费约 {_money(fuel)} 元（估算，过路费另计）", "高德距离 + 代码估算油费", True
    return "交通费用请以 12306 / 航司 / 购票平台为准", "无数据", False


@step_log("estimate_budget")
def estimate_budget(state: dict, transport: dict | None = None) -> dict:
    """
    花销估算主入口（纯确定性计算，无外发请求）。

    Args:
        state: 查询图状态（消费 tool_route / tool_rail / tool_stay_food）。
        transport: 交通方式建议（由 transport_advice_service 产出，避免重复计算）。

    Returns:
        dict: {"ok","days","nights","items":[{"name","text","amount","source","estimated"}],
               "total","text"}；ok=False 表示本次无法给出估算（未解析出目的地等）。
    """
    if not BUDGET_TOOL_ENABLE:
        return _empty()
    try:
        info = extract_destination_info(state) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"花销估算:行程信息解析失败,跳过,错误信息:{str(e)}")
        return _empty()
    if not str(info.get("destination") or "").strip():
        logger.info("花销估算:未解析到目的地,跳过")
        return _empty()

    if transport is None:
        try:
            from app.rag.tourism_query.transport_advice_service import decide_transport

            transport = decide_transport(state)
        except Exception:  # noqa: BLE001 — 推荐器异常不影响预算（交通项退化为"以官方为准"）
            transport = {}

    days = int(info.get("travel_days") or 0) or BUDGET_DEFAULT_DAYS
    days = max(1, min(days, 30))
    nights = max(1, days - 1)

    stay_food = state.get("tool_stay_food") or {}
    items: list[dict] = []

    # ---- 交通 ----
    t_text, t_source, t_est = _transport_item(state, transport or {})
    items.append({"name": "交通", "text": t_text, "source": t_source, "estimated": t_est})

    # ---- 住宿（人均/晚 × 晚数）----
    hotel_unit = _median_cost(stay_food.get("hotels") or [])
    if hotel_unit:
        amount = hotel_unit * nights
        items.append({
            "name": "住宿",
            "text": f"人均约 {_money(hotel_unit)} 元/晚 × {nights} 晚 ≈ {_money(amount)} 元",
            "amount": amount,
            "source": "高德酒店人均价",
            "estimated": False,
        })
    else:
        amount = BUDGET_HOTEL_DEFAULT_PER_NIGHT * nights
        items.append({
            "name": "住宿",
            "text": (
                f"人均约 {_money(BUDGET_HOTEL_DEFAULT_PER_NIGHT)} 元/晚 × {nights} 晚 ≈ {_money(amount)} 元"
                f"（预估：按默认单价×住宿晚数，实际以预订为准）"
            ),
            "amount": amount,
            "source": "预估（无高德房价数据）",
            "estimated": True,
        })

    # ---- 餐饮（人均单餐 × 每日餐数 × 天数）----
    meal_unit = _median_cost(stay_food.get("foods") or [])
    unit = meal_unit or BUDGET_MEAL_DEFAULT_PER_MEAL
    amount = unit * BUDGET_MEALS_PER_DAY * days
    if meal_unit:
        text = (
            f"人均约 {_money(unit)} 元/餐 × {BUDGET_MEALS_PER_DAY} 餐 × {days} 天 ≈ {_money(amount)} 元"
            "（预估：当地人均单餐来自高德 POI）"
        )
        source = "高德餐饮人均 × 预估公式"
    else:
        text = (
            f"人均约 {_money(unit)} 元/餐 × {BUDGET_MEALS_PER_DAY} 餐 × {days} 天 ≈ {_money(amount)} 元"
            "（预估：按默认单餐单价，实际以消费为准）"
        )
        source = "预估（无高德餐饮人均）"
    items.append({"name": "餐饮", "text": text, "amount": amount, "source": source, "estimated": True})

    # ---- 门票（高德不提供，禁止估算）----
    items.append({
        "name": "门票",
        "text": "各景区票价差异较大，请以景区实际收费为准",
        "source": "无数据",
        "estimated": False,
    })
    items.append({
        "name": "其他",
        "text": "购物、娱乐等按个人实际消费计算",
        "source": "无数据",
        "estimated": False,
    })

    total = sum(i["amount"] for i in items if i.get("amount") is not None)

    lines = [f"花销估算（人均，按 {days} 天 {nights} 晚计，来源逐项标注）："]
    for item in items:
        lines.append(f"- {item['name']}：{item['text']}")
    if total:
        lines.append(f"合计（人均，不含门票与购物）：约 {_money(total)} 元")
    text = "\n".join(lines)

    logger.info(
        f"花销估算:目的地={info.get('destination')} 天数={days} 住宿数据="
        f"{'高德' if hotel_unit else '默认'} 餐饮数据={'高德' if meal_unit else '默认'} 合计={total}"
    )
    return {"ok": True, "days": days, "nights": nights, "items": items, "total": total, "text": text}
