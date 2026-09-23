"""
天气统一网关（app.infra.weather_gateway）。

背景：此前天气是"双源并存、口径分裂"——对话流（node_tool_weather）走和风，
行程规划（trip_plan.collect_weather）走高德，两者预报天数、字段、定位策略全不一致，
同一问题在两条链路上会得到不同口径的天气。

本网关把天气能力收敛为单一门面：
  - 定位三级：和风 Geo → 高德 geocode → 外部传入经纬度兜底（含中国范围校验防幻觉坐标）
  - 数据源主备：默认主和风（3~30 天 + 降水量），主源失败自动降级高德（4 天、无降水量）
  - 输出归一化：date / desc / night_desc / temp_max / temp_min / precip / wind / source

为什么不做"二选一"（2026-09-09 实测结论，取证脚本 logs/_tmp_weather_probe.py）：
  - 高德 v3 weather 硬上限 4 天，而行程 travel_days 上限 30 天，覆盖不到；
  - 高德无降水量字段，会让 answer_out 规则 4（降水→建议带伞）与
    itinerary_out 规则 3（雨天优先室内/恶劣天气提示交通影响）失去量化依据。
  故保留和风为主、高德为备，由网关统一口径，而非砍掉其中一个。

设计约定（与 amap_gateway 保持一致）：
  - 一律软失败：任何异常不上抛，返回 ok=False + error，由上层决定降级文案；
  - 绝不把"查询失败/空数据"冒充成"该地无天气"，失败必须在 error 中留痕。
"""
from typing import Any, Optional

import httpx

from app.infra.amap_gateway import amap_gateway
from app.shared.config.common import env_bool, env_int, env_str
from app.shared.runtime.logger import logger
from app.shared.runtime.place_utils import _is_generic_place

# 天气总开关（沿用原 TOOL_WEATHER_ENABLE，语义不变）
WEATHER_ENABLE: bool = env_bool("TOOL_WEATHER_ENABLE", default=True)
# 预报天数：和风按档位自动向上取最近端点，高德最多 4 天
WEATHER_DAYS: int = min(max(env_int("TOOL_WEATHER_DAYS", default=5), 1), 30)
# 主数据源：qweather（默认，天全+有降水量）/ amap（4 天、无降水量）
WEATHER_PRIMARY: str = env_str("WEATHER_PRIMARY", default="qweather").strip().lower()
# 主源失败时是否自动降级到备源
WEATHER_FALLBACK: bool = env_bool("WEATHER_FALLBACK", default=True)
# 请求超时（秒）
WEATHER_TIMEOUT: int = env_int("TOOL_WEATHER_TIMEOUT", default=6)

# 和风 API Key 与专属 Host（未配置时和风源自动停用，交由备源顶上）
QWEATHER_API_KEY: str = env_str("QWEATHER_API_KEY", default="")
QWEATHER_API_HOST: str = env_str("QWEATHER_API_HOST", default="https://api.qweather.com").rstrip("/")

# 兜底经纬度的中国范围合法性校验（粗界，防模型幻觉出海外坐标）。
# 注意：route_tool_service 从 weather_tool_service 导入同名常量，那边做 re-export 保持兼容。
_CHINA_LAT_RANGE = (15.0, 55.0)
_CHINA_LON_RANGE = (70.0, 140.0)

# 和风逐日预报档位（3/7/15/30，实测当前订阅全部 code=200）
_FORECAST_DAY_TIERS = (3, 7, 15, 30)
# 高德 v3 weather 的硬上限（实测成都/九寨沟/杭州均只返回 4 天）
_AMAP_MAX_DAYS = 4

# 归一化逐日字段（两源统一输出，上游只认这一份结构）
_DAY_FIELDS = ("date", "desc", "night_desc", "temp_max", "temp_min", "precip", "wind", "source")


def _forecast_endpoint(days: int) -> str:
    """按需求天数选最近的和风预报端点（如 5 天 → 7d），避免多拉用不到的数据。"""
    for tier in _FORECAST_DAY_TIERS:
        if days <= tier:
            return f"/v7/weather/{tier}d"
    return f"/v7/weather/{_FORECAST_DAY_TIERS[-1]}d"


def _join_wind(direction: Any, power: Any) -> str:
    """
    拼接风力文案，两源统一为「X风Y级」。

    注意：和风 windDirDay 返回「北风」（自带"风"字），高德 daywind 返回「北」（不带），
    此处需去重，否则会拼出「北风风1-3级」。
    """
    d = str(direction or "").strip()
    p = str(power or "").strip()
    if d and not d.endswith("风"):
        d = f"{d}风"
    if d and p:
        return f"{d}{p}级"
    if d:
        return d
    if p:
        return f"{p}级"
    return ""


def _in_china(latitude: Optional[float], longitude: Optional[float]) -> bool:
    return (
        latitude is not None
        and longitude is not None
        and _CHINA_LAT_RANGE[0] <= latitude <= _CHINA_LAT_RANGE[1]
        and _CHINA_LON_RANGE[0] <= longitude <= _CHINA_LON_RANGE[1]
    )


class WeatherGateway:
    """天气统一网关：定位 → 主源预报 →（失败）备源预报 → 归一化输出。"""

    # ---------------------------------------------------------------- 定位
    def _qweather_geo(self, name: str) -> Optional[dict[str, Any]]:
        """和风 GeoAPI 城市搜索：中文地名 → 精确坐标 + 行政区（admin 为「省 市」拼接）。"""
        if not QWEATHER_API_KEY or not name:
            return None
        try:
            with httpx.Client(timeout=WEATHER_TIMEOUT) as client:
                resp = client.get(
                    f"{QWEATHER_API_HOST}/geo/v2/city/lookup",
                    params={"location": name, "number": 1, "lang": "zh"},
                    headers={"X-QW-Api-Key": QWEATHER_API_KEY},
                )
                resp.raise_for_status()
                data = resp.json()
            locations = data.get("location") or []
            if data.get("code") != "200" or not locations:
                return None
            top = locations[0]
            return {
                "admin": " ".join(part for part in (top.get("adm1"), top.get("adm2")) if part),
                "latitude": float(top.get("lat")),
                "longitude": float(top.get("lon")),
            }
        except Exception as e:  # noqa: BLE001 —— 定位属增强信息，失败只降级不上抛
            logger.warning(f"和风城市搜索失败[{name}],错误信息:{str(e)}")
            return None

    def _locate(
        self,
        destination: str,
        city: str,
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> Optional[dict[str, Any]]:
        """
        三级定位：和风 Geo → 高德 geocode → 外部传入经纬度兜底。

        Returns:
            {"name","admin","latitude","longitude","loc_source"}；全部失败返回 None。
        """
        # OBS-4：泛称目的地（市区/市中心/城区…）不能独立地理编码，否则会被误定位到同名异地
        # （如「市区」→ 台湾省台南市）。泛称须依附城市：city 存在则只用 city；
        # city 空则跳过 destination，走经纬度兜底，绝不盲信泛称。
        candidates = []
        if not _is_generic_place(destination):
            candidates.append(destination)
        if city:
            candidates.append(city)
        for candidate in candidates:
            if not candidate:
                continue
            loc = self._qweather_geo(candidate)
            if loc:
                return {**loc, "name": destination or candidate, "loc_source": "qweather"}
            geo = amap_gateway.geocode(candidate)
            if geo:
                return {
                    "admin": str(geo.get("formatted_address") or ""),
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "name": destination or candidate,
                    "loc_source": "amap",
                }
        if _in_china(latitude, longitude):
            return {
                "admin": "",
                "latitude": latitude,
                "longitude": longitude,
                "name": destination or city,
                "loc_source": "fallback",
            }
        return None

    # ---------------------------------------------------------------- 取数
    def _fetch_qweather(self, loc: dict[str, Any], days: int) -> list[dict[str, Any]]:
        """和风逐日预报（经纬度定位），字段含降水量 precip(mm)。"""
        if not QWEATHER_API_KEY:
            raise RuntimeError("未配置 QWEATHER_API_KEY，和风源不可用")
        with httpx.Client(timeout=WEATHER_TIMEOUT) as client:
            resp = client.get(
                f"{QWEATHER_API_HOST}{_forecast_endpoint(days)}",
                params={"location": f"{loc['longitude']:.2f},{loc['latitude']:.2f}"},
                headers={"X-QW-Api-Key": QWEATHER_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()
        if data.get("code") != "200":
            raise RuntimeError(f"和风预报返回异常码:{data.get('code')}")
        rows: list[dict[str, Any]] = []
        for item in (data.get("daily") or [])[:days]:
            rows.append({
                "date": str(item.get("fxDate") or ""),
                "desc": str(item.get("textDay") or item.get("textNight") or "未知天气"),
                "night_desc": str(item.get("textNight") or item.get("textDay") or ""),
                "temp_max": item.get("tempMax"),
                "temp_min": item.get("tempMin"),
                "precip": item.get("precip"),
                "wind": _join_wind(item.get("windDirDay"), item.get("windScaleDay")),
                "source": "qweather",
            })
        return rows

    def _fetch_amap(self, loc: dict[str, Any], days: int) -> list[dict[str, Any]]:
        """高德逐日预报（城市名定位），硬上限 4 天且无降水量（显式置 None，禁止用 0 冒充）。"""
        raw = amap_gateway.weather(loc.get("name") or loc.get("admin") or "")
        if not raw:
            raise RuntimeError("高德天气返回空")
        rows: list[dict[str, Any]] = []
        for cast in raw[: min(days, _AMAP_MAX_DAYS)]:
            rows.append({
                "date": str(cast.get("date") or ""),
                "desc": str(cast.get("day_weather") or cast.get("night_weather") or "未知天气"),
                "night_desc": str(cast.get("night_weather") or cast.get("day_weather") or ""),
                "temp_max": cast.get("day_temp"),
                "temp_min": cast.get("night_temp"),
                "precip": None,
                "wind": _join_wind(cast.get("wind_direction"), cast.get("wind_power")),
                "source": "amap",
            })
        return rows

    def _fetch(self, source: str, loc: dict[str, Any], days: int) -> list[dict[str, Any]]:
        if source == "qweather":
            return self._fetch_qweather(loc, days)
        if source == "amap":
            return self._fetch_amap(loc, days)
        raise RuntimeError(f"未知天气数据源:{source}")

    def _source_order(self) -> list[str]:
        """主源在前、备源在后；未开降级则只有主源。"""
        primary = WEATHER_PRIMARY if WEATHER_PRIMARY in ("qweather", "amap") else "qweather"
        backup = "amap" if primary == "qweather" else "qweather"
        return [primary, backup] if WEATHER_FALLBACK else [primary]

    # ---------------------------------------------------------------- 主入口
    def forecast(
        self,
        destination: str = "",
        city: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        days: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        查询逐日天气（统一入口）。

        Args:
            destination: 目的地原文（优先用于定位与展示）
            city: 所属城市（destination 未命中时的次级定位词）
            latitude/longitude: 兜底经纬度（前两级都未命中时使用，需落在中国范围内）
            days: 需要的天数，默认取 WEATHER_DAYS

        Returns:
            {"ok","name","admin","days","source","degraded","loc_source","error"}
            ok=True 时 days 为归一化逐日列表；degraded=True 表示主源失败、已走备源。
        """
        name = destination or city
        empty: dict[str, Any] = {
            "ok": False, "name": name, "admin": "", "days": [],
            "source": "", "degraded": False, "loc_source": "", "error": "",
        }
        if not WEATHER_ENABLE:
            return {**empty, "error": "天气工具已关闭（TOOL_WEATHER_ENABLE=false）"}
        if not (destination or city):
            return {**empty, "error": "未提供目的地"}

        want = max(1, min(days or WEATHER_DAYS, 30))
        loc = self._locate(destination, city, latitude, longitude)
        if not loc:
            return {**empty, "error": f"目的地定位失败[{destination or city}]"}

        last_err = ""
        for idx, source in enumerate(self._source_order()):
            try:
                rows = self._fetch(source, loc, want)
                if rows:
                    if idx > 0:
                        logger.warning(f"天气主源[{self._source_order()[0]}]不可用，已降级至备源[{source}]({name})")
                    return {
                        "ok": True,
                        "name": loc["name"],
                        "admin": loc.get("admin") or "",
                        "days": rows,
                        "source": source,
                        "degraded": idx > 0,
                        "loc_source": loc.get("loc_source") or "",
                        "error": "",
                    }
                last_err = f"{source} 返回空"
            except Exception as e:  # noqa: BLE001 —— 单源失败不阻断，继续尝试备源
                last_err = f"{source}: {str(e)}"
                logger.warning(f"天气源[{source}]查询失败({name}),错误信息:{str(e)}")

        return {**empty, "admin": loc.get("admin") or "", "loc_source": loc.get("loc_source") or "", "error": last_err or "全部天气源均无数据"}


# 模块级单例
weather_gateway = WeatherGateway()
