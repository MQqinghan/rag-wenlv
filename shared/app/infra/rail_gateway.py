"""
铁路车次/票价网关（数据源：自部署 12306-MCP，Streamable HTTP）—— 单例门面。

背景（T14）：
跨城公共交通是行程规划的最后一块空白——路线工具只给**驾车**，系统此前靠
`itinerary_service._strip_flight_train_numbers` 主动剥离车次号（怕编造），结果
"交通方式 + 在途时长 + 花销预估"只能泛泛而谈。接入 12306-MCP 后可得真实车次、
历时与各席别票价，路费从"拍脑袋"变"有据可查"。

范式对齐 `app/infra/weather_gateway.py`：
- 主源失败/超时 → `ok=False` + error 留痕，**不抛异常、不阻断主链路**；
- 绝不臆造：票价值缺失就留空，由上层写明"以 12306 官方为准"。

⚠️ 字段口径（2026-09-10 实测踩坑，务必遵守）：
- `query-tickets`      车次号在 **`train_no`**（如 `G2942`），票务在 `seats`；
- `query-ticket-price` 车次号在 **`train_code`**，其 `train_no` 是**内部哈希**
  （如 `6i000G294203`），票价在 `prices`。
归一化统一取 `train_code`（价格端点优先）；若误把 `train_no` 当车次号，会把内部
哈希输出给用户，且 `_strip_flight_train_numbers` 无法匹配，等于把乱码当车次号。
"""
from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime
from typing import Any, Optional

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from app.shared.config.common import env_bool, env_float, env_int, env_str
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.logger import logger

# 铁路工具总开关（与 TOOL_WEATHER_ENABLE/ROUTE_TOOL_ENABLE 对齐）
RAIL_TOOL_ENABLE: bool = env_bool("RAIL_TOOL_ENABLE", default=True)
# 自部署 12306-MCP 的 Streamable HTTP 端点（默认本机 uvx 常驻；容器化时改为服务名）
RAIL_MCP_URL: str = env_str("RAIL_MCP_URL", default="http://127.0.0.1:8010/mcp")
# 请求超时（秒）：与 AMAP_TIMEOUT 对齐，超时即放弃铁路补充，绝不拖慢主链路
RAIL_TIMEOUT: float = env_float("RAIL_TIMEOUT", default=6.0)
# 单次查询最多保留多少班次（票价简报够用即可，避免把 context 塞爆）
RAIL_MAX_TRAINS: int = env_int("RAIL_MAX_TRAINS", default=5)

SOURCE_TAG = "12306-MCP（公开接口）"


def _empty(origin: str, destination: str, travel_date: str, error: str) -> dict:
    """统一的失败结构：ok=False + 空车次 + error 留痕（不冒充"没有车次"）。"""
    return {
        "ok": False,
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "fetched_at": "",
        "source": SOURCE_TAG,
        "price_available": False,
        "trains": [],
        "codes": [],
        "error": error,
    }


def _normalize_price_row(row: dict) -> dict:
    """归一化 `query-ticket-price` 的一行（车次号取 `train_code`）。"""
    return {
        "code": str(row.get("train_code") or "").strip(),
        "from_station": row.get("from_station") or "",
        "to_station": row.get("to_station") or "",
        "depart": row.get("start_time") or "",
        "arrive": row.get("arrive_time") or "",
        "duration": row.get("duration") or "",
        "train_class": row.get("train_class_name") or "",
        "prices": {str(k): str(v) for k, v in (row.get("prices") or {}).items()},
    }


def _normalize_ticket_row(row: dict) -> dict:
    """归一化 `query-tickets` 的一行（该端点车次号在 `train_no`，无票价）。"""
    return {
        "code": str(row.get("train_no") or "").strip(),
        "from_station": row.get("from_station") or "",
        "to_station": row.get("to_station") or "",
        "depart": row.get("start_time") or "",
        "arrive": row.get("arrive_time") or "",
        "duration": row.get("duration") or "",
        "train_class": "",
        "prices": {},
    }


async def _call_mcp_async(tool: str, arguments: dict) -> dict:
    """以项目既有 `mcp` SDK 范式（Client + streamable_http + legacy 握手）调用一次工具。"""
    http_client = httpx2.AsyncClient(
        follow_redirects=True,
        timeout=httpx2.Timeout(RAIL_TIMEOUT + 2.0, read=RAIL_TIMEOUT + 2.0),
    )
    try:
        transport = streamable_http_client(RAIL_MCP_URL, http_client=http_client)
        client = Client(transport, mode="legacy")
        async with client:
            result = await client.call_tool(tool, arguments)
            text = result.content[0].text if result.content else ""
        if not text:
            return {"success": False, "error": "empty_response"}
        data = json.loads(text)
        return data if isinstance(data, dict) else {"success": False, "error": "bad_shape"}
    finally:
        await http_client.aclose()


def _call_mcp(tool: str, arguments: dict) -> dict:
    """同步包装 MCP 调用：超时/网络/解析异常一律降级为 {"success": False, "error": ...}。"""
    try:
        return asyncio.run(
            asyncio.wait_for(_call_mcp_async(tool, arguments), timeout=RAIL_TIMEOUT)
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(f"铁路网关超时（>{RAIL_TIMEOUT}s），降级为无铁路数据：{tool}")
        return {"success": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001 — 工具类一律软失败，绝不阻断主链路
        logger.warning(f"铁路网关调用失败[{tool}]，降级为无铁路数据，错误信息:{str(e)}")
        return {"success": False, "error": str(e)}


def _fetch(origin: str, destination: str, travel_date: str) -> dict:
    """真实拉取：先取票价（含车次/时刻），失败再退化为余票端点（只有车次/时刻）。"""
    arguments = {
        "from_station": origin,
        "to_station": destination,
        "train_date": travel_date,
    }
    price_data = _call_mcp("query-ticket-price", arguments)
    trains: list[dict] = []
    price_available = False
    if price_data.get("success") and price_data.get("data"):
        trains = [_normalize_price_row(r) for r in price_data["data"]]
        price_available = True
    else:
        # 退化：票价端点不可用时，至少拿到真实车次与时刻（票价留空，由上层写"以官方为准"）
        ticket_data = _call_mcp("query-tickets", arguments)
        if ticket_data.get("success") and ticket_data.get("trains"):
            trains = [_normalize_ticket_row(r) for r in ticket_data["trains"]]
        else:
            err = (
                price_data.get("error")
                or price_data.get("hint")
                or ticket_data.get("error")
                or ticket_data.get("hint")
                or "no_data"
            )
            return _empty(origin, destination, travel_date, str(err))

    trains = [t for t in trains if t.get("code")][:RAIL_MAX_TRAINS]
    if not trains:
        return _empty(origin, destination, travel_date, "no_valid_train")
    return {
        "ok": True,
        "origin": origin,
        "destination": destination,
        "date": travel_date,
        "fetched_at": f"{datetime.now():%Y-%m-%d %H:%M}",
        "source": SOURCE_TAG,
        "price_available": price_available,
        "trains": trains,
        "codes": [t["code"] for t in trains],
        "error": "",
    }


def query_trains(origin: str, destination: str, travel_date: str) -> dict:
    """
    查询两地之间的铁路车次与票价。

    Args:
        origin: 出发地（城市/车站名，如"深圳"）。
        destination: 目的地（如"成都"）。
        travel_date: 出行日期，格式 YYYY-MM-DD。

    Returns:
        dict: 归一化结果
        {"ok","origin","destination","date","fetched_at","source","price_available",
         "trains":[{code,from_station,to_station,depart,arrive,duration,train_class,prices}],
         "codes":[车次号...],"error"}
        ok=False 表示无可用数据（未配置/参数缺失/超时/无班次），调用方按"无铁路参考"处理。
    """
    origin = str(origin or "").strip()
    destination = str(destination or "").strip()
    travel_date = str(travel_date or "").strip()
    if not RAIL_TOOL_ENABLE:
        return _empty(origin, destination, travel_date, "disabled")
    if not (origin and destination and travel_date):
        return _empty(origin, destination, travel_date, "missing_params")
    if not RAIL_MCP_URL:
        return _empty(origin, destination, travel_date, "mcp_url_not_configured")
    try:
        # 进程内缓存（namespace=rail）：同一 OD + 日期在 TTL 内只打一次 MCP；
        # 命中返回同一对象引用，故回填前深拷贝，避免被上层改写污染缓存。
        result = cached_invoke(
            namespace="rail",
            cache_parts=(origin, destination, travel_date),
            producer=lambda: _fetch(origin, destination, travel_date),
            cache_label="rail",
        )
    except Exception as e:  # noqa: BLE001 — 缓存层异常同样软失败
        logger.warning(f"铁路网关缓存异常,降级直连,错误信息:{str(e)}")
        result = _fetch(origin, destination, travel_date)
    if not isinstance(result, dict):
        return _empty(origin, destination, travel_date, "bad_result")
    return copy.deepcopy(result)


def format_duration(raw: Any) -> str:
    """`07:58` → `7小时58分`；`25:54` → `25小时54分`；非法值原样返回空串。"""
    text = str(raw or "").strip()
    if ":" not in text:
        return ""
    hours, _, minutes = text.partition(":")
    try:
        return f"{int(hours)}小时{int(minutes)}分"
    except (TypeError, ValueError):
        return ""


def format_fares(prices: Optional[dict]) -> str:
    """席别票价 → `二等座 813.5 元 / 一等座 1281.5 元`；无票价返回空串（绝不估算）。"""
    if not prices:
        return ""
    return " / ".join(f"{seat} {price} 元" for seat, price in prices.items())
