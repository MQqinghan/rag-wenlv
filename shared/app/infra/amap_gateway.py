"""
高德地图 Web 服务（REST）网关 —— 单例门面。

由 helloagents-trip-planner 的 backend/app/services/amap_service.py 迁移改编：
- 去掉 MCPTool + uvx amap-mcp-server 子进程依赖，改为直连高德开放平台 REST API
  （本项目 route_tool_service.py 已验证该链路，.env 复用 AMAP_API_KEY/AMAP_TIMEOUT）；
- 照 app/infra/vectorstore/milvus_gateway.py 的「类 + 模块级单例」模式封装。

能力：地理编码 / POI 文本搜索 / POI 详情 / 天气 / 驾车·步行·公交路线。
所有方法均为软失败（返回 []/None/{}），不抛异常，方便上层降级。
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any, Optional

import httpx

from app.shared.config.common import env_int, env_str
from app.shared.runtime.logger import logger

# 高德开放平台 REST 端点
_AMAP_HOST = "https://restapi.amap.com"
_GEO_URL = f"{_AMAP_HOST}/v3/geocode/geo"
_TEXT_URL = f"{_AMAP_HOST}/v3/place/text"
_DETAIL_URL = f"{_AMAP_HOST}/v3/place/detail"
_WEATHER_URL = f"{_AMAP_HOST}/v3/weather/weatherInfo"
_DRIVING_URL = f"{_AMAP_HOST}/v3/direction/driving"
_WALKING_URL = f"{_AMAP_HOST}/v3/direction/walking"
_TRANSIT_URL = f"{_AMAP_HOST}/v3/direction/transit/integrated"
# 批量测距（T15-D1）：一个起点 → 多个终点，用于行程中"景点间距离"
_DISTANCE_URL = f"{_AMAP_HOST}/v3/distance"

# 配置（与 route_tool_service 共用 .env）
AMAP_API_KEY: str = env_str("AMAP_API_KEY", default="")
AMAP_TIMEOUT: int = env_int("AMAP_TIMEOUT", default=6)
AMAP_RETRY_TIMES: int = env_int("AMAP_RETRY_TIMES", default=1)
# 限流治理（2026-09-09 新增，服务于行程规划 C1 并行化前置）：
# - AMAP_MIN_INTERVAL_MS：进程级最小请求间隔，换算 QPS 上限（默认 250ms ≈ 4 QPS，
#   高德个人开发者 Key 并发通常 3~5，留余量）。设为 0 可关闭节流。
# - AMAP_RATE_LIMIT_RETRIES：命中限流后的**额外**退避重试次数（与网络异常重试额度独立）。
# - AMAP_BACKOFF_BASE_MS：退避基数，实际等待 = base * 2^attempt + 抖动，上限 3s。
AMAP_MIN_INTERVAL_MS: int = env_int("AMAP_MIN_INTERVAL_MS", default=250)
AMAP_RATE_LIMIT_RETRIES: int = env_int("AMAP_RATE_LIMIT_RETRIES", default=3)
AMAP_BACKOFF_BASE_MS: int = env_int("AMAP_BACKOFF_BASE_MS", default=400)

# 高德限流特征：HTTP 200 + status!="1"，infocode/info 命中下列关键字即视为"可重试的限流"
# （典型：CUQPS_HAS_EXCEEDED_THE_LIMIT 并发超限 / USER_DAILY_QUERY_OVER_LIMIT 日配额）
_RATE_LIMIT_HINTS = ("CUQPS", "QPS", "OVER_LIMIT", "TOO_FREQUENT", "EXCEED", "LIMIT")
_BACKOFF_MAX_S = 3.0


def _is_rate_limited(data: Optional[dict]) -> bool:
    """
    判定高德返回是否为「限流」（可退避重试）。

    与业务错误（如 INVALID_USER_KEY / INVALID_PARAMS）区分：
    限流重试有意义，参数/密钥错误重试无意义且会加剧限流。
    """
    if not isinstance(data, dict):
        return False
    if _is_status_ok(data):
        return False
    text = f"{data.get('infocode') or ''} {data.get('info') or ''}".upper()
    return any(hint in text for hint in _RATE_LIMIT_HINTS)


class _Throttle:
    """进程级最小间隔节流（线程安全），避免并发打爆高德 QPS 配额。"""

    def __init__(self, min_interval_ms: int) -> None:
        self.min_interval = max(0.0, min_interval_ms / 1000.0)
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def acquire(self) -> None:
        """占用一个请求额度；距上次请求不足最小间隔则阻塞等待。"""
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.monotonic()


# 模块级单例节流器（后续 C1 线程池并发核查时，所有线程共享同一 QPS 额度）
_throttle = _Throttle(AMAP_MIN_INTERVAL_MS)

# 请求结局追踪（thread-local）：供上层区分「限流/异常导致的失败」与「查无此地」
# 并发安全——每个工作线程独立记录自己最近一次请求的结局，互不干扰。
_tls = threading.local()

# 结局取值
OUTCOME_OK = "ok"            # 请求成功且 status=="1"
OUTCOME_BUSINESS = "business"  # 业务错误（密钥/参数问题），重试无意义
OUTCOME_RATE_LIMITED = "rate_limited"  # 限流，重试额度已耗尽
OUTCOME_ERROR = "error"      # 网络/HTTP/JSON 异常

# 地理编码结果缓存：city/adcode -> dict（避免每次行程规划重复地理编码）
_city_geo_cache: dict[str, dict[str, Any]] = {}


def _is_status_ok(data: dict) -> bool:
    """高德 REST 返回体 status=='1' 视为成功。"""
    return str(data.get("status") or "") == "1"


class AmapGateway:
    """高德地图 REST 网关（单例，惰性初始化）。"""

    def __init__(self) -> None:
        self.api_key: str = AMAP_API_KEY
        self.timeout: int = AMAP_TIMEOUT
        self.retry_times: int = AMAP_RETRY_TIMES

    # ---------------------------------------------------------------- 基础
    def _set_outcome(self, value: str) -> None:
        """记录本线程最近一次请求的结局（thread-local，并发安全）。"""
        _tls.outcome = value

    def last_outcome(self) -> str:
        """
        本线程最近一次请求的结局：ok / business / rate_limited / error。

        用途：上层拿到空结果（[] / None / {}）时，用它区分「因限流或异常没查到」
        与「确实查无此地」——二者语义完全不同，前者不能当作"不存在"处理。
        """
        return getattr(_tls, "outcome", OUTCOME_OK)

    def _sleep_backoff(self, attempt: int) -> None:
        """指数退避 + 抖动（抖动用于避免多线程同时重试、再次撞上限流）。"""
        base = AMAP_BACKOFF_BASE_MS / 1000.0
        delay = min(_BACKOFF_MAX_S, base * (2 ** attempt)) + random.uniform(0, base * 0.3)
        time.sleep(delay)

    def _http_get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        GET 并解析 JSON；网络/JSON 异常上抛，由上层软失败处理。

        限流治理（2026-09-09 新增，C1 并行化前置）：
        - 请求前经 `_throttle` 限速，避免并发打爆高德 QPS 配额；
        - 命中限流（HTTP 200 + status!="1" + infocode 含限流特征）时**退避重试**，
          重试额度 `AMAP_RATE_LIMIT_RETRIES` 与网络异常额度独立，互不挤占；
        - 额度耗尽后返回最后一次响应体，交由上层按 status!="1" 软失败（不抛异常）。
        """
        if not self.api_key:
            raise ValueError("AMAP_API_KEY 未配置（.env 中设置 AMAP_API_KEY）")
        params = dict(params)
        params.setdefault("key", self.api_key)
        net_attempts = max(1, self.retry_times + 1)  # 网络/HTTP 异常重试额度
        rate_attempts = max(0, AMAP_RATE_LIMIT_RETRIES)  # 限流额外退避重试额度
        total_attempts = net_attempts + rate_attempts
        last_err: Optional[Exception] = None
        for attempt in range(total_attempts):
            try:
                _throttle.acquire()
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                if not _is_rate_limited(data):
                    self._set_outcome(OUTCOME_OK if _is_status_ok(data) else OUTCOME_BUSINESS)
                    return data
                self._set_outcome(OUTCOME_RATE_LIMITED)
                logger.warning(
                    f"高德限流(url={url}, 第{attempt + 1}/{total_attempts}次): "
                    f"{data.get('infocode')} {data.get('info')}"
                )
                if attempt >= total_attempts - 1:
                    return data  # 额度耗尽：交回上层软失败，绝不冒充"查无此地"为成功
                self._sleep_backoff(attempt)
            except Exception as e:  # noqa: BLE001 —— 网关层统一兜底，逐级软失败
                last_err = e
                self._set_outcome(OUTCOME_ERROR)
                logger.warning(f"高德REST请求失败(url={url}, 第{attempt + 1}次): {e}")
                if attempt >= net_attempts - 1:
                    break  # 网络额度用尽：不再借用限流额度重试网络错误
                self._sleep_backoff(attempt)
        raise last_err  # type: ignore[misc]

    @staticmethod
    def parse_location(loc_text: Optional[str]) -> Optional[dict[str, float]]:
        """解析高德 "lng,lat" 字符串 → {"longitude","latitude"}。"""
        if not loc_text or "," not in str(loc_text):
            return None
        try:
            lng_s, lat_s = str(loc_text).split(",", 1)
            return {"longitude": float(lng_s), "latitude": float(lat_s)}
        except (ValueError, TypeError):
            return None

    # ---------------------------------------------------------------- 地理编码
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        中文地址 → 坐标（含 adcode 供天气接口使用）。

        Returns:
            {"longitude","latitude","adcode","formatted_address"} | None
        """
        if not address or not address.strip():
            return None
        cache_key = f"{city or ''}|{address.strip()}"
        if cache_key in _city_geo_cache:
            return dict(_city_geo_cache[cache_key])
        params: dict[str, Any] = {"address": address.strip()}
        if city:
            params["city"] = city
        try:
            data = self._http_get(_GEO_URL, params)
            if not _is_status_ok(data):
                logger.warning(f"高德地理编码异常({address}): {data.get('info')}")
                return None
            geocodes = data.get("geocodes") or []
            if not geocodes:
                return None
            first = geocodes[0]
            loc = self.parse_location(str(first.get("location") or ""))
            if not loc:
                return None
            result = {
                **loc,
                "adcode": str(first.get("adcode") or ""),
                "formatted_address": str(first.get("formatted_address") or ""),
            }
            _city_geo_cache[cache_key] = result
            return result
        except Exception as e:  # noqa: BLE001
            logger.warning(f"高德地理编码失败({address}): {e}")
            return None

    # ---------------------------------------------------------------- POI
    @staticmethod
    def _normalize_biz_ext(raw: Any) -> dict[str, str]:
        """
        归一化 POI 扩展信息 `biz_ext`（`extensions=all` 时高德返回）。

        **实测字段名（2026-09-10 核对，勿按文档臆测）**：
        - `rating` → 评分（如 "4.8"）
        - `cost` → 人均消费（**无数据时高德返回空列表 `[]`**，须丢弃）
        - 开放时间在 `open_time`（简写，如 "08:00-18:00 08:00-17:30"），
          另有 `opentime2`（含季节性长文案）——本项目取短的 `open_time`，缺失才回落 `opentime`。
        高德对"无数据"的字段可能返回 `[]` / null / 空串，此处统一丢弃：**缺失即不写**，绝不臆造。
        """
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key in ("rating", "cost"):
            value = raw.get(key)
            if value is None or isinstance(value, (list, dict)):
                continue
            text = str(value).strip()
            if text:
                out[key] = text
        for key in ("open_time", "opentime", "opentime2"):
            value = raw.get(key)
            if value is None or isinstance(value, (list, dict)):
                continue
            text = str(value).strip()
            if text:
                out["opentime"] = text
                break
        return out

    def search_poi(
        self,
        keywords: str,
        city: Optional[str] = None,
        citylimit: bool = True,
        offset: int = 10,
        extensions: str = "base",
    ) -> list[dict[str, Any]]:
        """
        文本搜索 POI（v3/place/text），返回归一化 POI dict 列表。

        每条 dict 字段：id/name/type/typecode/address/location("lng,lat")/tel/pname/cityname/
        biz_ext（`extensions="all"` 时含 rating/cost/opentime，否则为空 dict）。
        失败返回 []。

        Args:
            extensions: "base"（默认，字段最少、最省额度）或 "all"（含评分/人均/营业时间，
                配额消耗更高，仅供行程侧"景点结构化信息"按需使用）。
        """
        if not keywords or not keywords.strip():
            return []
        params: dict[str, Any] = {
            "keywords": keywords.strip(),
            "offset": str(max(1, min(offset, 25))),
            "page": "1",
            "extensions": extensions if extensions in ("base", "all") else "base",
        }
        if city:
            params["city"] = city.strip()
            params["citylimit"] = "true" if citylimit else "false"
        try:
            data = self._http_get(_TEXT_URL, params)
            if not _is_status_ok(data):
                logger.warning(f"高德POI搜索异常({keywords}): {data.get('info')}")
                return []
            pois: list[dict[str, Any]] = []
            for item in (data.get("pois") or [])[:offset]:
                if not isinstance(item, dict):
                    continue
                pois.append({
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or "").strip(),
                    "type": str(item.get("type") or ""),
                    "typecode": str(item.get("typecode") or ""),
                    "address": str(item.get("address") or "").strip(),
                    "location": str(item.get("location") or ""),
                    "tel": str(item.get("tel") or "") or None,
                    "pname": str(item.get("pname") or ""),
                    "cityname": str(item.get("cityname") or ""),
                    # 行政区（区/县/县级市）：行程侧用来判断"景点是否在市区"，也便于分天动线
                    "adname": str(item.get("adname") or ""),
                    "biz_ext": self._normalize_biz_ext(item.get("biz_ext")),
                })
            return pois
        except Exception as e:  # noqa: BLE001
            logger.warning(f"高德POI搜索失败({keywords}): {e}")
            return []

    def poi_detail(self, poi_id: str) -> Optional[dict[str, Any]]:
        """
        获取 POI 详情（v3/place/detail）。

        Returns:
            dict | None：详情字段（含 photos 数组，可为空）。
        """
        if not poi_id:
            return None
        try:
            data = self._http_get(_DETAIL_URL, {"id": poi_id})
            if not _is_status_ok(data):
                logger.warning(f"高德POI详情异常(id={poi_id}): {data.get('info')}")
                return None
            pois = data.get("pois") or []
            return pois[0] if pois else None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"高德POI详情失败(id={poi_id}): {e}")
            return None

    # ---------------------------------------------------------------- 批量测距
    def batch_distance(
        self,
        origins: list[str],
        destination: str,
        distance_type: int = 1,
    ) -> list[dict[str, Any]]:
        """
        批量测距（v3/distance）：**多个起点 → 一个终点**，用于行程中"景点间距离"。

        ⚠️ 高德该端点的参数方向与直觉相反（2026-09-10 实测核对，勿按直觉写）：
        `origins` 支持多个坐标对（"|" 分隔，上限 100），`destination` **只支持 1 个**。
        实测传 `origins` 单个 + `destination` 多个会返回 `INVALID_PARAMS(20000)`。

        Args:
            origins: 起点坐标列表，元素为 "lng,lat"。
            destination: 终点坐标（单个），格式 "lng,lat"。
            distance_type: 0=直线 / 1=驾车（默认）/ 3=步行。

        Returns:
            list[dict]：与 origins **按序一一对应**的
            {"origin","distance_m","duration_s"}（origin 为对应起点坐标，便于上层对齐名称）；
            请求失败返回 []。
        """
        if not destination or not origins:
            return []
        origin_list = [str(o).strip() for o in origins if str(o or "").strip()][:100]
        if not origin_list:
            return []
        params: dict[str, Any] = {
            "origins": "|".join(origin_list),
            "destination": str(destination).strip(),
            "type": str(distance_type if distance_type in (0, 1, 3) else 1),
        }
        try:
            data = self._http_get(_DISTANCE_URL, params)
            if not _is_status_ok(data):
                logger.warning(f"高德批量测距异常: {data.get('info')}")
                return []
            out: list[dict[str, Any]] = []
            for item in data.get("results") or []:
                if not isinstance(item, dict):
                    continue
                # origin_id 是 1-based 序号，用于把结果对回 origins 顺序（缺省按追加顺序）
                try:
                    idx = int(item.get("origin_id") or 0) - 1
                except (TypeError, ValueError):
                    idx = len(out)
                origin = origin_list[idx] if 0 <= idx < len(origin_list) else ""
                out.append({
                    "origin": origin,
                    "distance_m": float(item.get("distance") or 0),
                    "duration_s": int(float(item.get("duration") or 0)),
                })
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning(f"高德批量测距失败: {e}")
            return []

    # ---------------------------------------------------------------- 天气
    def weather(self, city: str) -> list[dict[str, Any]]:
        """
        查询城市逐日天气（v3/weather/weatherInfo，预报 3~4 天）。

        Returns:
            归一化逐日天气列表，每条：
            {"date","week","day_weather","night_weather","day_temp","night_temp",
             "wind_direction","wind_power"}；失败返回 []。
        """
        if not city or not city.strip():
            return []
        params: dict[str, Any] = {"city": city.strip(), "extensions": "all"}
        try:
            data = self._http_get(_WEATHER_URL, params)
            if not _is_status_ok(data):
                # 城市名直接查询失败 → 地理编码补 adcode 再试
                geo = self.geocode(city.strip())
                if not geo or not geo.get("adcode"):
                    logger.warning(f"高德天气查询异常({city}): {data.get('info')}")
                    return []
                data = self._http_get(_WEATHER_URL, {"city": geo["adcode"], "extensions": "all"})
                if not _is_status_ok(data):
                    logger.warning(f"高德天气查询异常(adcode,{city}): {data.get('info')}")
                    return []
            forecasts = data.get("forecasts") or []
            if not forecasts:
                return []
            casts = forecasts[0].get("casts") or []
            weather_list: list[dict[str, Any]] = []
            for cast in casts:
                if not isinstance(cast, dict):
                    continue
                weather_list.append({
                    "date": str(cast.get("date") or ""),
                    "week": str(cast.get("week") or ""),
                    "day_weather": str(cast.get("dayweather") or ""),
                    "night_weather": str(cast.get("nightweather") or ""),
                    "day_temp": cast.get("daytemp", 0),
                    "night_temp": cast.get("nighttemp", 0),
                    "wind_direction": str(cast.get("daywind") or cast.get("nightwind") or ""),
                    "wind_power": str(cast.get("daypower") or cast.get("nightpower") or ""),
                })
            return weather_list
        except Exception as e:  # noqa: BLE001
            logger.warning(f"高德天气查询失败({city}): {e}")
            return []

    # ---------------------------------------------------------------- 路线
    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
    ) -> dict[str, Any]:
        """
        两点间路线规划（walking/driving/transit）。

        Returns:
            {"distance","duration","route_type","description"}；失败返回 {}。
        """
        if not origin_address or not destination_address:
            return {}
        url_map = {
            "walking": _WALKING_URL,
            "driving": _DRIVING_URL,
            "transit": _TRANSIT_URL,
        }
        url = url_map.get(route_type, _WALKING_URL)
        # driving/walking 直接传中文地址即可；transit 需先地理编码取坐标 + 城市名
        if route_type == "transit":
            origin_geo = self.geocode(origin_address, origin_city)
            dest_geo = self.geocode(destination_address, destination_city)
            if not origin_geo or not dest_geo:
                return {}
            params: dict[str, Any] = {
                "origin": f"{origin_geo['longitude']:.6f},{origin_geo['latitude']:.6f}",
                "destination": f"{dest_geo['longitude']:.6f},{dest_geo['latitude']:.6f}",
                "city": origin_city or origin_address,
                "cityd": destination_city or destination_address,
                "strategy": "0",
            }
        else:
            params = {
                "origin": origin_address,
                "destination": destination_address,
                "origin_city": origin_city or origin_address,
                "destination_city": destination_city or destination_address,
            }
            if origin_city:
                params["origin_city"] = origin_city
            if destination_city:
                params["destination_city"] = destination_city
        try:
            data = self._http_get(url, params)
            if not _is_status_ok(data):
                logger.warning(f"高德路线规划异常({origin_address}→{destination_address}): {data.get('info')}")
                return {}
            route = data.get("route") or {}
            paths = route.get("paths") or []
            transits = route.get("transits") or []
            if route_type in ("walking", "driving") and paths:
                first = paths[0]
                distance = float(first.get("distance") or 0)
                duration = int(float(first.get("duration") or 0))
            elif route_type == "transit" and transits:
                first = transits[0]
                distance = float(first.get("walking_distance") or first.get("distance") or 0)
                duration = int(float(first.get("duration") or 0))
            else:
                logger.warning(f"高德路线规划未返回可用路径: {origin_address}→{destination_address}")
                return {}
            return {
                "distance": distance,
                "duration": duration,
                "route_type": route_type,
                "description": f"全程约 {distance / 1000:.1f} 公里，耗时约 {duration / 60:.0f} 分钟（{route_type}）",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"高德路线规划失败({origin_address}→{destination_address}): {e}")
            return {}


# 模块级单例（照 milvus_gateway 模式）
amap_gateway = AmapGateway()


