"""
住宿 / 餐饮 POI 工具服务（规划类问题专用，2026-09-10 主人反馈问题2）。

背景：旧规划分支只调了高德「景点 POI + 天气 + 路线」，**没有酒店与餐饮**，
导致行程输出里"住宿/餐饮"只能写"以现场为准"，看起来像没调用高德（主人的原话）。
高德文本搜索本身就能召回酒店与餐厅，并可通过 `extensions=all` 拿到评分与人均消费。

链路：复用行程信息解析（目的地 / 所属城市，与天气 / 路线 / 铁路 / 景点共用一次调用 + 缓存）
      → `amap_gateway.search_poi(extensions="all")` 分别召回酒店与餐饮
      → 组装【住宿餐饮参考】简报，写入 `state["tool_stay_food"]`，
        供行程拼装（住宿建议 + 花销估算）引用。

反编造：**字段只做透传**（高德返回什么写什么，缺失即不写），绝不估算金额；
金额类估算由 `budget_service` 统一在代码层完成并标注"预估"。

降级：未配置 Key / 未解析目的地 / 查询失败一律 `ok=False` 空简报，不阻断主链路。
"""
from __future__ import annotations

from datetime import datetime

from app.infra.amap_gateway import AMAP_API_KEY, amap_gateway
from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.shared.config.common import env_bool, env_int, env_str
from app.shared.runtime.logger import logger, step_log

# 工具总开关（关掉即不产出【住宿餐饮参考】）
STAY_FOOD_TOOL_ENABLE: bool = env_bool("STAY_FOOD_TOOL_ENABLE", default=True)
# 酒店 / 餐饮各保留条数上限（控制高德配额与简报长度）
STAY_FOOD_MAX_HOTELS: int = env_int("STAY_FOOD_MAX_HOTELS", default=4)
STAY_FOOD_MAX_FOODS: int = env_int("STAY_FOOD_MAX_FOODS", default=4)
# 检索关键词，多词用 "|" 或 "," 分隔（多词结果合并去重）
STAY_HOTEL_KEYWORDS: str = env_str("STAY_HOTEL_KEYWORDS", default="酒店|民宿")
STAY_FOOD_KEYWORDS: str = env_str("STAY_FOOD_KEYWORDS", default="特色美食|地方菜")
# 是否标注候选「距景点约 X 公里」（只标注、不排序；关掉可省一次高德测距）
STAY_FOOD_DISTANCE_ENABLE: bool = env_bool("STAY_FOOD_DISTANCE_ENABLE", default=True)
# 距离参照景点的检索关键词（单个即可，控制高德配额；取召回结果 top1 作参照）
STAY_FOOD_ANCHOR_KEYWORD: str = env_str("STAY_FOOD_ANCHOR_KEYWORD", default="旅游景区")


def _keyword_list(raw: str, fallback: str) -> list[str]:
    """解析关键词配置（"|" 或 "," 分隔），为空回落默认。"""
    parts = [k.strip() for k in str(raw or "").replace(",", "|").split("|") if k.strip()]
    return parts or [fallback]


def _empty(destination: str = "", city: str = "") -> dict:
    """统一的失败结构：ok=False + 空候选 + 空简报。"""
    return {
        "ok": False,
        "destination": destination,
        "city": city,
        "fetched_at": "",
        "anchor": "",
        "hotels": [],
        "foods": [],
        "text": "",
    }


def _collect(city: str, keywords: list[str], limit: int) -> list[dict]:
    """多关键词召回 + 按名称去重，保持高德相关度顺序（不排序，避免主观偏好）。"""
    merged: dict[str, dict] = {}
    for keyword in keywords:
        for poi in amap_gateway.search_poi(
            keyword, city=city, citylimit=True, offset=max(limit, 4), extensions="all"
        ):
            name = str(poi.get("name") or "").strip()
            if name and name not in merged:
                merged[name] = poi
        if len(merged) >= limit * 2:
            break
    return list(merged.values())[:limit]


def _format_km(distance_m: float) -> str:
    """距离的中文表述：>=1 公里保留一位小数，否则用米。"""
    km = float(distance_m or 0.0) / 1000.0
    if km >= 1.0:
        return f"约 {km:.1f} 公里"
    return f"约 {int(distance_m)} 米"


def _resolve_scenic_anchor(state: dict, city: str) -> tuple[str, str]:
    """
    确定「距离参照景点」：优先复用同批景点 POI 简报的参照点（串行场景），
    否则用单关键词召回 top1（并行场景 tool_poi 尚不可见），都失败返回 ("", "")。

    Returns:
        (景点名, "lng,lat")；未取得时均为空串。
    """
    brief = state.get("tool_poi") or {}
    if brief.get("ok"):
        anchor_name = str(brief.get("anchor") or "").strip()
        for poi in brief.get("pois") or []:
            if poi.get("name") == anchor_name and poi.get("location"):
                return anchor_name, str(poi["location"])
    for poi in amap_gateway.search_poi(
        STAY_FOOD_ANCHOR_KEYWORD, city=city, citylimit=True, offset=3, extensions="base"
    ):
        if poi.get("location"):
            return str(poi.get("name") or "").strip(), str(poi["location"])
    return "", ""


def _annotate_distances(state: dict, pois: list[dict], city: str) -> str:
    """
    给候选逐条标注「距<景点>约 X 公里」（只标注、不排序，不改变原候选顺序）。

    Returns:
        参照景点名（成功标注时）；未标注时为 ""。
    """
    if not STAY_FOOD_DISTANCE_ENABLE:
        return ""
    targets = [p for p in pois if p.get("location")]
    logger.info(f"住宿餐饮工具:距离标注开始,候选={len(targets)},城市={city}")
    if not targets:
        return ""
    anchor_name, anchor_loc = _resolve_scenic_anchor(state, city)
    if not anchor_name or not anchor_loc:
        logger.info("住宿餐饮工具:未取得距离参照景点,跳过距离标注")
        return ""
    raw = amap_gateway.batch_distance(
        [str(p["location"]) for p in targets], anchor_loc
    )
    by_origin = {item.get("origin"): item for item in raw if item.get("origin")}
    hit = 0
    for poi in targets:
        item = by_origin.get(str(poi["location"]))
        if not item:
            continue
        poi["_distance_text"] = f"距{anchor_name}{_format_km(item.get('distance_m') or 0.0)}"
        hit += 1
    if hit:
        logger.info(f"住宿餐饮工具:已标注 {hit} 条候选到「{anchor_name}」的距离")
    return anchor_name if hit else ""


def _poi_segment(poi: dict) -> str:
    """单条候选 → 简报行（只透传高德字段，缺失即不写）。"""
    name = str(poi.get("name") or "").strip()
    if not name:
        return ""
    type_text = str(poi.get("type") or "").split(";")[-1]
    adname = str(poi.get("adname") or "").strip()
    tags = "；".join(x for x in (type_text, adname) if x)
    segment = f"- {name}" + (f"（{tags}）" if tags else "")
    address = str(poi.get("address") or "").strip()
    if address:
        segment += f"：{address}"
    biz = poi.get("biz_ext") or {}
    extras: list[str] = []
    if biz.get("rating"):
        extras.append(f"评分 {biz['rating']}")
    if biz.get("cost"):
        extras.append(f"人均 {biz['cost']} 元")
    if biz.get("opentime"):
        extras.append(f"营业时间 {biz['opentime']}")
    dist_text = str(poi.get("_distance_text") or "").strip()
    if dist_text:
        extras.append(dist_text)
    if extras:
        segment += "；" + "；".join(extras)
    return segment


def build_stay_food_text(result: dict) -> str:
    """把酒店 / 餐饮候选组装成中文简报（供《住宿餐饮参考》区块直接引用）。"""
    lines: list[str] = [
        f"目的地 {result.get('destination')}"
        f"（来源：高德 POI，数据时间 {result.get('fetched_at')}，"
        "价格与营业时间以商家实际执行为准）："
    ]
    hotels = result.get("hotels") or []
    if hotels:
        lines.append("住宿候选：")
        lines.extend(s for s in (_poi_segment(p) for p in hotels) if s)
    foods = result.get("foods") or []
    if foods:
        lines.append("餐饮候选：")
        lines.extend(s for s in (_poi_segment(p) for p in foods) if s)
    if not hotels and not foods:
        lines.append("（本次未取到住宿/餐饮候选数据）")
    return "\n".join(lines)


@step_log("get_stay_food_brief")
def get_stay_food_brief(state: dict) -> dict:
    """
    住宿/餐饮工具主入口：产出 `state["tool_stay_food"]` 简报。

    Returns:
        dict: {"ok","destination","city","fetched_at","hotels","foods","text"}
        ok=False 表示本次无可用数据。
    """
    if not STAY_FOOD_TOOL_ENABLE:
        return _empty()
    if not AMAP_API_KEY:
        logger.warning("住宿餐饮工具:未配置 AMAP_API_KEY,跳过(请在 .env 填写)")
        return _empty()
    try:
        info = extract_destination_info(state) or {}
        destination = str(info.get("destination") or "").strip()
        city = str(info.get("city") or "").strip() or destination
        if not destination:
            logger.info("住宿餐饮工具:未解析到目的地,跳过")
            return _empty()

        hotels = _collect(city, _keyword_list(STAY_HOTEL_KEYWORDS, "酒店"), max(1, STAY_FOOD_MAX_HOTELS))
        foods = _collect(city, _keyword_list(STAY_FOOD_KEYWORDS, "特色美食"), max(1, STAY_FOOD_MAX_FOODS))
        if not hotels and not foods:
            logger.info(
                f"住宿餐饮工具无结果（city={city}）,outcome={amap_gateway.last_outcome()},降级为空简报"
            )
            return _empty(destination, city)

        # 距离标注（只标注、不排序；失败不丢候选数据本身）
        anchor_name = ""
        try:
            anchor_name = _annotate_distances(
                state, (hotels or []) + (foods or []), city
            )
        except Exception as e:  # noqa: BLE001 — 标注失败不影响候选
            logger.warning(f"住宿餐饮工具:距离标注异常,忽略,错误信息:{str(e)}")

        result = {
            "ok": True,
            "anchor": anchor_name,
            "destination": destination,
            "city": city,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hotels": [p for p in hotels if p.get("name")],
            "foods": [p for p in foods if p.get("name")],
        }
        result["text"] = build_stay_food_text(result)
        logger.info(f"住宿餐饮工具:酒店 {len(result['hotels'])} 个、餐饮 {len(result['foods'])} 个（{destination}）")
        return result
    except Exception as e:  # noqa: BLE001 — 工具一律软失败，不阻断主链路
        logger.warning(f"住宿餐饮工具异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        return _empty()
