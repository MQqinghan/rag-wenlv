"""
铁路工具服务（规划类问题专用）：解析出行信息 → 调 rail_gateway → 组装中文铁路简报，
写入 `state["tool_rail"]` 供行程拼装阶段引用（与 weather/route 工具并列）。

链路：复用行程信息解析（出发地/目的地，与天气/路线工具共用一次 LLM 调用 + 缓存）
      → 推断出行日期（问题里的显式日期 / 相对时间；无法识别则按"明天"参考并显式声明）
      → rail_gateway.query_trains（自部署 12306-MCP）
      → 组装【铁路参考】简报（含查询日期与"以 12306 官方为准"限定语）。

降级策略：未配置/未解析出发地或目的地/查询失败一律 `ok=False` 空简报，不阻断主链路。
反编造：**票价只做透传，绝不估算**；无票价时简报不出现任何数字。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from app.infra.rail_gateway import (
    RAIL_TOOL_ENABLE,
    format_duration,
    format_fares,
    query_trains,
)
from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.shared.runtime.logger import logger, step_log

# 相对日期词 → 距今天数（注意顺序：先长后短，"大后天"必须先于"后天"判断）
_REL_DAY_WORDS: tuple[tuple[str, int], ...] = (
    ("大后天", 3),
    ("后天", 2),
    ("明天", 1),
    ("明日", 1),
    ("今天", 0),
    ("今日", 0),
    ("当天", 0),
)

# 显式日期：2026-09-11 / 2026/9/11 / 2026年9月11日
_EXPLICIT_DATE_PATTERN = re.compile(r"(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*[日号]?")
# 缺年份日期：9月11日 / 9月11号（不识别"9-11"，避免与票价/天数误撞）
_MONTH_DAY_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")

_WEEKDAY_CHARS = "一二三四五六日天"
_WEEKDAY_PATTERN = re.compile(r"(?:周|星期|礼拜)([一二三四五六日天])")
# R15（2026-09-11 主人实测 E3）：「下周X/下周末/裸下周」按下一自然周（周一起算）解析，
# 不得并入"未来7天首个该星期"口径——周五说"下周"被映射到本周六属实测缺陷。
_NEXT_WEEK_PATTERN = re.compile(r"(?:下周|下星期)(?:(末)|([一二三四五六日天]))?")


def _today(state: dict) -> datetime:
    """取"今天"：优先用 state["current_date"] 的日期段（评测会显式注入），否则用系统时钟。"""
    text = str(state.get("current_date") or "")
    match = re.match(r"\s*(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    return datetime.now()


def infer_travel_date(state: dict) -> tuple[str, bool]:
    """
    从用户问题中推断出行日期（行程首日）。

    优先级：显式年月日 → 缺年份月日 → 相对词（今天/明天/后天/大后天）→ 下周/下周X/下周末（下一自然周）→ 周几 → 兜底"明天"。

    Returns:
        (YYYY-MM-DD, 是否为兜底假设)。兜底为真时上层会在简报里显式声明"按明天参考"，
        绝不让用户误以为是从问题里读出的日期。
    """
    question = f"{state.get('original_query') or ''} {state.get('rewritten_query') or ''}"
    today = _today(state)

    match = _EXPLICIT_DATE_PATTERN.search(question)
    if match:
        try:
            return f"{datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))):%Y-%m-%d}", False
        except ValueError:
            pass

    match = _MONTH_DAY_PATTERN.search(question)
    if match:
        try:
            return f"{datetime(today.year, int(match.group(1)), int(match.group(2))):%Y-%m-%d}", False
        except ValueError:
            pass

    for word, offset in _REL_DAY_WORDS:
        if word in question:
            return f"{today + timedelta(days=offset):%Y-%m-%d}", False

    match = _NEXT_WEEK_PATTERN.search(question)
    if match:
        next_monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
        if match.group(2):  # 下周X / 下星期X → 下一自然周对应星期
            target = _WEEKDAY_CHARS.index(match.group(2))
            return f"{next_monday + timedelta(days=target):%Y-%m-%d}", False
        if match.group(1):  # 下周末 → 下一周的周六
            return f"{next_monday + timedelta(days=5):%Y-%m-%d}", False
        return f"{next_monday:%Y-%m-%d}", False  # 裸"下周" → 下周一

    match = _WEEKDAY_PATTERN.search(question)
    if match:
        target = _WEEKDAY_CHARS.index(match.group(1))  # 一→0 … 日/天→6
        days_ahead = (target - today.weekday()) % 7 or 7
        # 口径：「周X/星期X」（不带"下"字）= 未来 7 天内首个该星期；
        # "下周X/下周末/下周"已在 _NEXT_WEEK_PATTERN 按下一自然周解析。
        return f"{today + timedelta(days=days_ahead):%Y-%m-%d}", False

    return f"{today + timedelta(days=1):%Y-%m-%d}", True


def _empty(origin: str = "", destination: str = "", travel_date: str = "") -> dict:
    """统一的失败结构：ok=False + 空车次 + 空简报（上层写"本次无铁路数据"）。"""
    return {
        "ok": False,
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "date_assumed": False,
        "fetched_at": "",
        "trains": [],
        "codes": [],
        "text": "",
    }


def build_rail_text(result: dict, date_assumed: bool) -> str:
    """把归一化车次结果组装成中文简报（供《铁路参考》区块直接引用）。"""
    lines: list[str] = []
    head = f"查询日期 {result.get('date')}（来源：{result.get('source')}，票价以 12306 官方为准）"
    if date_assumed:
        head = (
            f"未从问题中识别到明确出行日期，以下按默认出行日参考（{head}），"
            "实际班次请以用户真实出行日为准"
        )
    lines.append(head + "：")
    for train in result.get("trains") or []:
        segment = f"- {train.get('code')}"
        if train.get("train_class"):
            segment += f"（{train['train_class']}）"
        segment += (
            f" {train.get('from_station')} {train.get('depart')}"
            f" → {train.get('to_station')} {train.get('arrive')}"
        )
        duration = format_duration(train.get("duration"))
        if duration:
            segment += f"，历时 {duration}"
        fares = format_fares(train.get("prices"))
        # 无票价时只声明出处，绝不写数字（反编造硬约束）
        segment += f"，{fares}" if fares else "，票价请以 12306 官方为准"
        lines.append(segment)
    if not result.get("price_available"):
        lines.append("（本次未取到各席别票价，交通费用请以 12306 官方为准）")
    return "\n".join(lines)


@step_log("get_rail_brief")
def get_rail_brief(state: dict) -> dict:
    """
    铁路工具主入口：产出 `state["tool_rail"]` 简报。

    Returns:
        dict: {"ok","origin","destination","date","date_assumed","fetched_at","trains","codes","text"}
        ok=False 表示本次无可用铁路数据。
    """
    if not RAIL_TOOL_ENABLE:
        return _empty()
    try:
        info = extract_destination_info(state) or {}
        origin = str(info.get("origin") or "").strip()
        destination = str(info.get("destination") or "").strip()
        # 铁路只对"两点间"有意义：出发地或目的地缺失（含"我在X怎么玩"式单程抵达类）直接跳过
        if not origin or not destination or origin == destination:
            logger.info(f"铁路工具跳过：出发地/目的地不完整（origin={origin!r} destination={destination!r}）")
            return _empty(origin, destination)

        travel_date, date_assumed = infer_travel_date(state)
        result = query_trains(origin, destination, travel_date)
        if not result.get("ok"):
            logger.info(
                f"铁路工具无数据（{origin}→{destination} {travel_date}），"
                f"错误信息:{result.get('error')}，降级为空简报"
            )
            return _empty(origin, destination, travel_date)

        return {
            "ok": True,
            "origin": origin,
            "destination": destination,
            "date": travel_date,
            "date_assumed": date_assumed,
            "fetched_at": result.get("fetched_at") or "",
            "trains": result.get("trains") or [],
            "codes": result.get("codes") or [],
            "text": build_rail_text(result, date_assumed),
        }
    except Exception as e:  # noqa: BLE001 — 工具一律软失败，不阻断主链路
        logger.warning(f"铁路工具异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        return _empty()
