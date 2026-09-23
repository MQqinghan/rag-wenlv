"""
行程地图数据服务（2026-09-10 主人反馈问题5）。

目标：让对话流的行程回答也能像 `trip.html` 那样显示高德地图（标记点 + 按天连线）。
对话流与结构化 trip 流不同——行程是**纯文本**产出的，没有结构化景点坐标，因此本模块
负责把文本还原成"按天的景点 + 坐标"。

坐标来源（优先级从高到低，全部真实，绝不臆造）：
1. `state["tool_poi"].pois`（高德景点 POI，含 location）—— 同日出现即直接采用；
2. `state["tool_stay_food"].hotels`（高德酒店 POI，含 location）—— 住宿点也上图更直观；
3. `amap_gateway.geocode(名称, city)` —— 文本里出现但候选池没有的名称，逐点补编码
   （受 `ITINERARY_MAP_GEOCODE_MAX` 上限约束，避免打爆配额）。

定位不到坐标的景点**不上图**（宁缺毋滥），文本里照常展示。
"""
from __future__ import annotations

import re

from app.infra.amap_gateway import AMAP_API_KEY, amap_gateway
from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.shared.config.common import env_bool, env_int
from app.shared.runtime.logger import logger, step_log

ITINERARY_MAP_ENABLE: bool = env_bool("ITINERARY_MAP_ENABLE", default=True)
# 地图点上限（前端渲染与地理编码成本双重约束）
ITINERARY_MAP_MAX_POINTS: int = env_int("ITINERARY_MAP_MAX_POINTS", default=20)
# 允许的地理编码补点次数上限（候选池未命中的名称才会走编码）
ITINERARY_MAP_GEOCODE_MAX: int = env_int("ITINERARY_MAP_GEOCODE_MAX", default=8)

# 天分隔（优先级从高到低）：
# 1) 显式天标题：第1天 / 第二天 / Day 2
# 2) 日期：2026-09-11（仅在没有显式天标题时使用——概览行里也会出现日期，直接混用会切错）
_DAY_TITLE_PATTERN = re.compile(r"第\s*([一二三四五六七八九十\d]+)\s*天|Day\s*(\d+)")
_DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 活动动词后紧跟的地点名（"游览西湖""前往灵隐寺"）——用于候选池未命中时的补点
_PLACE_AFTER_VERB_PATTERN = re.compile(
    r"(?:游览|参观|前往|抵达|游玩|漫步|打卡|逛逛|逛一逛|登上|乘船游览|夜游)"
    r"\s*([\u4e00-\u9fa5A-Za-z0-9·\-]{2,12})"
)
# 名称清洗：截掉常见尾缀（"后""附近""一带""等"）与标点
_TAIL_TRIM_PATTERN = re.compile(r"(后|附近|一带|等地|等|区域|景区内|周边)$")

# 地名可信度过滤（问题5 补充，2026-09-10）：活动动词后抽出的"名称"经常是**从句片段**
# （"周边的轮渡码头""鼓浪屿的轮渡信息""环岛路骑行或散步"），送去地理编码会落到错误坐标——
# 实测"周边的轮渡码头"被高德编到十几公里外的翔安区。地图上出现**错误标记比不标更糟**，
# 故凡命中功能词/动宾短语/章节名/标点者一律丢弃（宁缺毋滥）；候选池命中的 POI 不走此过滤。
_NON_PLACE_PATTERN = re.compile(
    r"的|交通|住宿|餐饮|预算|行程|安排|建议|信息|说明|提示|注意事项|自由活动|购物|休息|"
    r"骑行|散步|漫步|游览|品尝|欣赏|观看|体验|打卡|前往|抵达|游玩|返程|出发|需|乘|"
    r"或|与|及|、|，|,|。|（|）|\(|\)|:|：|/|→|~"
)
# 从句边界字符（截断用）：动词后捕获常把整个从句吃进来（"鼓浪屿需乘轮渡""环岛路骑行或散步"
# "周边的轮渡码头"），先在第一个边界字符处截断，**保住真实地名前缀**（"鼓浪屿""环岛路"）——
# 比整段丢弃覆盖率更高；截断后仍为从句者再交给 `_is_place_like` 丢掉。
_CLAUSE_BREAK_CHARS = set(
    "的或与及需乘要可请并也还再将会能把被让使至往从以及、，,。：:；;！!？?"
    "（）()[]【】/→~骑散逛"
)
# 边界只在第 3 个字符及其后生效，避免误伤"可可托海"这类以功能字起头的地名
_CLAUSE_BREAK_MIN_INDEX = 2
# 地名长度上限（中文地名极少超过 8 字，如"鼓浪屿风景名胜区"为 8 字）
_PLACE_NAME_MAX_LEN = 10


def _truncate_at_clause(text: str) -> str:
    """在第一个「从句边界字符」处截断，保留真实地名前缀（边界不在前 2 字内生效）。"""
    for idx, char in enumerate(text):
        if idx >= _CLAUSE_BREAK_MIN_INDEX and char in _CLAUSE_BREAK_CHARS:
            return text[:idx]
    return text


def _is_place_like(name: str) -> bool:
    """粗判抽取结果是否"像地名"（供地理编码前过滤，非地名一律丢弃，避免错误标记）。"""
    text = str(name or "").strip()
    if len(text) < 2 or len(text) > _PLACE_NAME_MAX_LEN:
        return False
    if _NON_PLACE_PATTERN.search(text):
        return False
    # 纯数字/纯字母/单字重复等明显非地名
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


def _empty() -> dict:
    return {"ok": False, "name": "", "city": "", "days": []}


def _norm_loc(location) -> tuple[float, float] | None:
    """"lng,lat" 字符串 → (lon, lat)；非法返回 None。"""
    text = str(location or "").strip()
    if "," not in text:
        return None
    try:
        lon_s, lat_s = text.split(",", 1)
        return float(lon_s), float(lat_s)
    except (TypeError, ValueError):
        return None


def _cn_to_int(text: str) -> int | None:
    """中文/阿拉伯数字 → int（支持"十""十二"等简单形式）。"""
    text = str(text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if text.startswith("十"):
        return 10 + _CN_NUM.get(text[1:], 0)
    if text.endswith("十"):
        return _CN_NUM.get(text[0], 0) * 10
    if "十" in text:
        left, _, right = text.partition("十")
        return _CN_NUM.get(left, 0) * 10 + _CN_NUM.get(right, 0)
    return _CN_NUM.get(text)


def _split_days(answer_text: str) -> list[tuple[str, str]]:
    """
    按天切分行程正文 → [(天标签, 该天正文)]。

    优先用显式天标题（第N天 / Day N）；没有时才退化为按日期切分；
    两者都没有则返回单组（标签"行程"），保证前端至少能展示点位。
    """
    text = answer_text or ""
    title_matches = list(_DAY_TITLE_PATTERN.finditer(text))
    groups: list[tuple[str, str]] = []
    if title_matches:
        for idx, match in enumerate(title_matches):
            start = match.start()
            end = title_matches[idx + 1].start() if idx + 1 < len(title_matches) else len(text)
            cn, day_no = match.group(1), match.group(2)
            num = _cn_to_int(cn) if cn else (int(day_no) if day_no else None)
            label = f"第{num}天" if num else f"第{idx + 1}天"
            groups.append((label, text[start:end]))
        return groups

    date_matches = list(_DATE_PATTERN.finditer(text))
    if date_matches:
        # 概览行里也会出现日期（"2026-09-11→2026-09-15"）：把连续紧邻的日期视为同一行叙述，
        # 只用"其后还有正文且与下一个日期之间距离足够远"的日期作为分组起点。
        starts: list[re.Match[str]] = []
        for idx, match in enumerate(date_matches):
            nxt = date_matches[idx + 1].start() if idx + 1 < len(date_matches) else len(text)
            if nxt - match.end() < 40 and idx + 1 < len(date_matches):
                continue  # 与下一个日期紧邻（概览里的区间）→ 跳过
            starts.append(match)
        if not starts:
            return [("行程", text)]
        for idx, match in enumerate(starts):
            start = match.start()
            end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
            groups.append((match.group(1), text[start:end]))
        return groups

    return [("行程", text)]


def _build_pool(state: dict, city: str) -> dict[str, dict]:
    """
    候选名称 → {"name","lon","lat","address"}（来自高德 POI，无需编码）。
    名称做长度降序匹配用的原始键。
    """
    pool: dict[str, dict] = {}

    def _add(name: str, location, address: str = "") -> None:
        name = str(name or "").strip()
        if not name or name in pool:
            return
        loc = _norm_loc(location)
        if not loc:
            return
        pool[name] = {"name": name, "lon": loc[0], "lat": loc[1], "address": address or ""}

    for poi in (state.get("tool_poi") or {}).get("pois") or []:
        _add(poi.get("name"), poi.get("location"), str(poi.get("address") or ""))
    for poi in (state.get("tool_stay_food") or {}).get("hotels") or []:
        _add(poi.get("name"), poi.get("location"), str(poi.get("address") or ""))
    return pool


def _match_pool(text: str, pool: dict[str, dict]) -> list[dict]:
    """按名称在文本中的出现位置排序，返回该文本内命中的候选（长名优先，避免短名截胡）。"""
    hits: list[tuple[int, dict]] = []
    for name in sorted(pool.keys(), key=len, reverse=True):
        idx = text.find(name)
        if idx >= 0:
            hits.append((idx, pool[name]))
    hits.sort(key=lambda x: x[0])
    return [h[1] for h in hits]


def _extract_place_names(text: str) -> list[str]:
    """
    从活动动词后抽取地点名（候选池未命中时用于地理编码补点）。

    三步净化：① `_truncate_at_clause` 截掉从句尾巴保住地名前缀；② `_TAIL_TRIM_PATTERN`
    去掉"附近/一带/周边"等尾缀；③ `_is_place_like` 只留"像地名"的片段。目的是
    **不让从句片段拿到错误坐标**（宁缺毋滥，见 `_NON_PLACE_PATTERN` 注释）。
    """
    names: list[str] = []
    for match in _PLACE_AFTER_VERB_PATTERN.finditer(text or ""):
        raw = match.group(1).strip()
        raw = _TAIL_TRIM_PATTERN.sub("", _truncate_at_clause(raw)).strip()
        if len(raw) >= 2 and raw not in names and _is_place_like(raw):
            names.append(raw)
    return names


@step_log("build_itinerary_map")
def build_itinerary_map(state: dict, answer_text: str) -> dict:
    """
    把行程正文还原成地图数据。

    Returns:
        dict: {"ok","name","city","days":[{"index","label","attractions":[
              {"name","lon","lat","address","kind"}]}]}
        ok=False 表示无可用坐标（前端不渲染地图）。
    """
    if not ITINERARY_MAP_ENABLE or not answer_text:
        return _empty()
    try:
        info = extract_destination_info(state) or {}
        city = str(info.get("city") or "").strip() or str(info.get("destination") or "").strip()
        if not city:
            return _empty()

        pool = _build_pool(state, city)
        geocode_budget = max(0, ITINERARY_MAP_GEOCODE_MAX)
        groups = _split_days(answer_text)

        days: list[dict] = []
        total_points = 0
        for idx, (label, body) in enumerate(groups):
            points: list[dict] = []
            seen: set[str] = set()
            for hit in _match_pool(body, pool):
                if hit["name"] in seen:
                    continue
                seen.add(hit["name"])
                points.append({**hit, "kind": "poi"})
            # 兜底：候选池没有的名称走地理编码（有上限）
            if AMAP_API_KEY and geocode_budget > 0:
                for name in _extract_place_names(body):
                    if geocode_budget <= 0 or total_points + len(points) >= ITINERARY_MAP_MAX_POINTS:
                        break
                    if any(name in s or s in name for s in seen):
                        continue
                    geo = amap_gateway.geocode(name, city)
                    geocode_budget -= 1
                    if not geo:
                        continue
                    seen.add(name)
                    points.append({
                        "name": name,
                        "lon": geo["longitude"],
                        "lat": geo["latitude"],
                        "address": str(geo.get("formatted_address") or ""),
                        "kind": "geocoded",
                    })
            points = points[:ITINERARY_MAP_MAX_POINTS]
            total_points += len(points)
            if points:
                days.append({"index": idx, "label": label, "attractions": points})

        if not days:
            logger.info(f"行程地图:未解析到任何带坐标的景点（city={city}）,前端不渲染地图")
            return _empty()

        logger.info(f"行程地图:城市={city} 天数={len(days)} 点位={total_points}")
        return {
            "ok": True,
            "name": f"{info.get('destination') or city}行程",
            "city": city,
            "days": days,
        }
    except Exception as e:  # noqa: BLE001 — 地图数据属增强信息，失败不影响答案
        logger.warning(f"行程地图构建异常,跳过地图渲染,错误信息:{str(e)}")
        return _empty()
