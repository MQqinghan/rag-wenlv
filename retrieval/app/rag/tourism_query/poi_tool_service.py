"""
景点 POI 工具服务（规划类问题专用，T15-D1）。

链路：复用行程信息解析（目的地 / 所属城市，与天气 / 路线 / 铁路共用一次 LLM 调用 + 缓存）
      → `amap_gateway.search_poi(extensions="all")` 多关键词召回目的地景点
        （含评分 / 人均 / 开放时间 / 行政区）
      → 按"市区（行政区名以『区』结尾）优先"排序，取前 N 个
      → `amap_gateway.batch_distance` 以首个景点为参照，算**其余景点到它的驾车距离**
      → 组装【景点POI参考】简报，写入 `state["tool_poi"]` 供行程拼装阶段引用。

设计取舍：
- **只服务行程分支**（`is_plan=True`），不接入普通文旅问答——改主链路范围最小化（主人拍板方案 A）。
- 用结构化接口替代"从网页摘要里猜景点信息"，既省 web search 调用费，也让行程里的
  开放时间 / 人均消费 / 景点间距离有据可查。
- **关键词实测校准（2026-09-10）**：单个 `"景点"` 召回的多是商务楼/写字楼/园区（质量差）；
  `"旅游景区"` / `"热门景点"` / `"名胜古迹"` 才能召回宽窄巷子、武侯祠、大熊猫基地等真实景区，
  故默认三词合并去重。
- `extensions=all` 比 base 更耗高德配额，故加 `POI_TOOL_ENABLE` 总开关 + 条数上限。

降级策略：未配置 Key / 未解析目的地 / 查询失败一律 `ok=False` 空简报，不阻断主链路。
反编造：POI 字段**只做透传**（高德返回什么写什么，缺失即不写），绝不估算。
"""
from __future__ import annotations

from datetime import datetime

from app.infra.amap_gateway import AMAP_API_KEY, amap_gateway
from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.shared.config.common import env_bool, env_int, env_str
from app.shared.runtime.logger import logger, step_log

# 工具总开关（默认开；关掉即不产出【景点POI参考】，行程回到无 POI 的表现）
POI_TOOL_ENABLE: bool = env_bool("POI_TOOL_ENABLE", default=True)
# 保留景点条数上限（避免简报膨胀 + 控制高德配额消耗）
POI_TOOL_MAX_ITEMS: int = env_int("POI_TOOL_MAX_ITEMS", default=6)
# 是否计算景点间距离（需 >=2 个带坐标的景点；关掉可再省一次高德调用）
POI_TOOL_DISTANCE_ENABLE: bool = env_bool("POI_TOOL_DISTANCE_ENABLE", default=True)
# POI 检索关键词，多个用 "|" 或 "," 分隔（多词结果合并去重）
POI_SEARCH_KEYWORDS: str = env_str("POI_SEARCH_KEYWORDS", default="旅游景区|热门景点|名胜古迹")


def _empty(destination: str = "", city: str = "") -> dict:
    """统一的失败结构：ok=False + 空景点 + 空简报。"""
    return {
        "ok": False,
        "destination": destination,
        "city": city,
        "fetched_at": "",
        "names": [],
        "pois": [],
        "distances": [],
        "text": "",
    }


def _keyword_list() -> list[str]:
    """解析关键词配置（"|" 或 "," 分隔），为空回落内置默认。"""
    parts = [k.strip() for k in POI_SEARCH_KEYWORDS.replace(",", "|").split("|") if k.strip()]
    return parts or ["旅游景区"]


def _collect_pois(city: str, offset: int) -> list[dict]:
    """
    多关键词召回 + 按名称去重，再按"市区优先"排序。

    排序规则（确定性）：行政区名以「区」结尾（如 青羊区 / 武侯区）视为市区景点，排在前；
    「市 / 县 / 镇」等卫星区域排在后。目的：三天市区行程优先拿到宽窄巷子这类核心景点，
    而不是都江堰 / 青城山等远郊景区（它们仍保留在列表里，供用户想跑远郊时选用）。
    """
    merged: dict[str, dict] = {}
    for keyword in _keyword_list():
        for poi in amap_gateway.search_poi(
            keyword, city=city, citylimit=True, offset=max(offset, 4), extensions="all"
        ):
            name = poi.get("name")
            if name and name not in merged:
                merged[name] = poi
    pois = list(merged.values())
    pois.sort(key=lambda p: 0 if str(p.get("adname") or "").endswith("区") else 1)
    return pois


def _format_distance_line(distance_m: float, duration_s: int) -> str:
    """单条距离的中文表述（公里 + 驾车时长）。"""
    km = distance_m / 1000.0
    text = f"约 {km:.1f} 公里" if km >= 0.1 else f"约 {int(distance_m)} 米"
    if duration_s > 0:
        minutes = max(1, round(duration_s / 60))
        text += f"（驾车约 {minutes} 分钟）"
    return text


def build_poi_text(result: dict) -> str:
    """
    把 POI 结果组装成中文简报（供《景点POI参考》区块直接引用）。

    只透传高德返回的字段：名称 / 类型 / 行政区 / 地址 / 评分 / 人均 / 开放时间；缺失即不写。
    """
    lines: list[str] = [
        f"目的地 {result.get('destination')}"
        f"（来源：高德 POI，数据时间 {result.get('fetched_at')}，"
        "门票与开放时间以景区实际执行为准）："
    ]
    for poi in result.get("pois") or []:
        name = poi.get("name")
        if not name:
            continue
        type_text = str(poi.get("type") or "").split(";")[-1]
        adname = str(poi.get("adname") or "").strip()
        tags = "；".join(x for x in (type_text, adname) if x)
        segment = f"- {name}" + (f"（{tags}）" if tags else "")
        address = poi.get("address")
        if address:
            segment += f"：{address}"
        biz = poi.get("biz_ext") or {}
        extras: list[str] = []
        if biz.get("rating"):
            extras.append(f"评分 {biz['rating']}")
        if biz.get("cost"):
            extras.append(f"人均 {biz['cost']} 元")
        if biz.get("opentime"):
            extras.append(f"开放时间 {biz['opentime']}")
        if extras:
            segment += "；" + "；".join(extras)
        lines.append(segment)

    distances = result.get("distances") or []
    if distances:
        anchor = result.get("anchor") or (result.get("names") or [""])[0]
        lines.append(f"景点间距离（高德驾车测距，以 {anchor} 为参照）：")
        for item in distances:
            lines.append(
                f"- {item.get('to')} 距 {anchor} "
                f"{_format_distance_line(item.get('distance_m') or 0.0, item.get('duration_s') or 0)}"
            )
    return "\n".join(lines)


@step_log("get_poi_brief")
def get_poi_brief(state: dict) -> dict:
    """
    景点 POI 工具主入口：产出 `state["tool_poi"]` 简报。

    Returns:
        dict: {"ok","destination","city","fetched_at","names","pois","anchor","distances","text"}
        ok=False 表示本次无可用 POI 数据。
    """
    if not POI_TOOL_ENABLE:
        return _empty()
    if not AMAP_API_KEY:
        logger.warning("景点POI工具:未配置 AMAP_API_KEY,跳过(请在 .env 填写)")
        return _empty()
    try:
        info = extract_destination_info(state) or {}
        destination = str(info.get("destination") or "").strip()
        city = str(info.get("city") or "").strip() or destination
        if not destination:
            logger.info("景点POI工具:未解析到目的地,跳过")
            return _empty()
        limit = max(1, POI_TOOL_MAX_ITEMS)
        got = _collect_pois(city, limit)
        # 归一化失败软降级：高德限流/异常时 search_poi 返回 []（last_outcome 记录原因）
        if not got:
            logger.info(
                f"景点POI工具无结果（city={city}）,outcome={amap_gateway.last_outcome()},降级为空简报"
            )
            return _empty(destination, city)

        pois = [p for p in got if p.get("name") and (p.get("location") or p.get("address"))][:limit]
        if not pois:
            return _empty(destination, city)

        # 景点间距离：以首个景点（市区优先序第一）为参照，其余景点为起点（高德该端点支持多起点+单终点）
        distances: list[dict] = []
        anchor = pois[0]
        others = [p for p in pois[1:] if p.get("location")]
        if POI_TOOL_DISTANCE_ENABLE and others and anchor.get("location"):
            raw = amap_gateway.batch_distance(
                [p["location"] for p in others], anchor["location"]
            )
            by_origin = {item.get("origin"): item for item in raw if item.get("origin")}
            for poi in others:
                item = by_origin.get(poi.get("location"))
                if not item:
                    continue
                distances.append({
                    "from": anchor.get("name"),
                    "to": poi.get("name"),
                    "distance_m": item.get("distance_m") or 0.0,
                    "duration_s": item.get("duration_s") or 0,
                })

        result = {
            "ok": True,
            "destination": destination,
            "city": city,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "names": [p["name"] for p in pois],
            "anchor": anchor.get("name") or "",
            "pois": pois,
            "distances": distances,
        }
        result["text"] = build_poi_text(result)
        logger.info(f"景点POI工具:命中 {len(pois)} 个景点（{destination}），距离 {len(distances)} 条")
        return result
    except Exception as e:  # noqa: BLE001 — 工具一律软失败，不阻断主链路
        logger.warning(f"景点POI工具异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        return _empty()
