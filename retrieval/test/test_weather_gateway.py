"""
天气统一网关离线单测（不发起任何真实网络请求）。

校验的是网关语义，不是供应商返回值：
  1) 主源（和风）成功 → 不降级，输出归一化字段且含降水量；
  2) 主源失败 → 自动降级备源（高德），degraded=True，降水量显式为 None（禁止用 0 冒充）；
  3) 关闭降级时主源失败 → ok=False 且 error 留痕，绝不冒充"该地无天气"；
  4) 定位三级：和风 Geo → 高德 geocode → 外部坐标兜底（含中国范围校验）；
  5) 高德备源受 4 天硬上限约束，和风按档位给足天数；
  6) 两源输出字段集合完全一致（上游只认一份结构）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infra import amap_gateway as amap_mod
from app.infra import weather_gateway as wg

_QW_ROWS = [
    {"fxDate": "2026-09-09", "textDay": "阵雨", "textNight": "多云",
     "tempMax": "25", "tempMin": "18", "precip": "3.5",
     "windDirDay": "北", "windScaleDay": "1-3"},
    {"fxDate": "2026-09-10", "textDay": "多云", "textNight": "晴",
     "tempMax": "27", "tempMin": "19", "precip": "0.0",
     "windDirDay": "南", "windScaleDay": "1-2"},
]
_AMAP_ROWS = [
    {"date": "2026-09-09", "week": "3", "day_weather": "阵雨", "night_weather": "多云",
     "day_temp": "25", "night_temp": "18", "wind_direction": "北", "wind_power": "1-3"},
]


class _Ctx:
    """把网关的可替换点临时换成桩函数，退出时还原（默认不发任何真实请求）。"""

    _UNSET = object()
    _DEFAULT_GEO = {"latitude": 30.57, "longitude": 104.06, "formatted_address": "四川省成都市"}

    def __init__(self, qweather=None, amap=None, qgeo=_UNSET, geo=_UNSET):
        self.qweather = qweather
        self.amap = amap
        # 未显式指定时：和风 Geo 直接失败、高德 geocode 返回固定坐标，保证零网络请求
        self.qgeo = (lambda name: None) if qgeo is self._UNSET else qgeo
        self.geo = (lambda name: dict(self._DEFAULT_GEO)) if geo is self._UNSET else geo
        self.saved = {}

    def __enter__(self):
        g = wg.weather_gateway
        self.saved = {
            "_fetch_qweather": g._fetch_qweather,
            "_fetch_amap": g._fetch_amap,
            "_qweather_geo": g._qweather_geo,
            "geocode": amap_mod.amap_gateway.geocode,
        }
        wg.QWEATHER_API_KEY = "stub-key"
        if self.qweather is not None:
            g._fetch_qweather = lambda loc, days: self.qweather(loc, days)
        if self.amap is not None:
            g._fetch_amap = lambda loc, days: self.amap(loc, days)
        g._qweather_geo = lambda name: self.qgeo(name)
        amap_mod.amap_gateway.geocode = lambda name: self.geo(name)
        return self

    def __exit__(self, *a):
        g = wg.weather_gateway
        g._fetch_qweather = self.saved["_fetch_qweather"]
        g._fetch_amap = self.saved["_fetch_amap"]
        g._qweather_geo = self.saved["_qweather_geo"]
        amap_mod.amap_gateway.geocode = self.saved["geocode"]
        return False


def _norm_qw(loc, days):
    return [
        {"date": r["fxDate"], "desc": r["textDay"], "night_desc": r["textNight"],
         "temp_max": r["tempMax"], "temp_min": r["tempMin"], "precip": r["precip"],
         "wind": f"{r['windDirDay']}风{r['windScaleDay']}级", "source": "qweather"}
        for r in _QW_ROWS[:days]
    ]


def _norm_amap(loc, days):
    return [
        {"date": r["date"], "desc": r["day_weather"], "night_desc": r["night_weather"],
         "temp_max": r["day_temp"], "temp_min": r["night_temp"], "precip": None,
         "wind": f"{r['wind_direction']}风{r['wind_power']}级", "source": "amap"}
        for r in _AMAP_ROWS[: min(days, 4)]
    ]


def test_primary_ok_no_degrade():
    """主源成功：source=qweather、degraded=False、字段含降水量。"""
    with _Ctx(qweather=_norm_qw, amap=_norm_amap,
              qgeo=lambda n: {"admin": "四川省 成都", "latitude": 30.57, "longitude": 104.06}):
        r = wg.weather_gateway.forecast(destination="成都", days=5)
    assert r["ok"] is True, f"主源应成功: {r}"
    assert r["source"] == "qweather", r["source"]
    assert r["degraded"] is False, "主源成功不应降级"
    assert r["days"][0]["precip"] == "3.5", r["days"][0]
    assert r["days"][0]["wind"] == "北风1-3级", r["days"][0]
    assert r["admin"] == "四川省 成都", r


def test_primary_fail_degrades_to_backup():
    """主源异常 → 降级高德，degraded=True，且降水量为 None（不用 0 冒充）。"""
    def boom(loc, days):
        raise RuntimeError("和风 401")

    with _Ctx(qweather=boom, amap=_norm_amap):
        r = wg.weather_gateway.forecast(destination="成都", days=5)
    assert r["ok"] is True, f"应降级成功: {r}"
    assert r["source"] == "amap", r["source"]
    assert r["degraded"] is True, "主源失败必须标记 degraded"
    assert r["days"][0]["precip"] is None, "高德无降水量，必须为 None 而非 0"


def test_no_fallback_returns_error_not_empty():
    """关闭降级且主源失败：ok=False + error 留痕，绝不冒充"该地无天气"。"""
    def boom(loc, days):
        raise RuntimeError("和风 401")

    old = wg.WEATHER_FALLBACK
    wg.WEATHER_FALLBACK = False
    try:
        with _Ctx(qweather=boom, amap=_norm_amap):
            r = wg.weather_gateway.forecast(destination="成都")
    finally:
        wg.WEATHER_FALLBACK = old
    assert r["ok"] is False, f"关闭降级时应失败: {r}"
    assert r["days"] == [], r
    assert "和风" in (r["error"] or ""), f"error 必须留痕: {r['error']}"


def test_locate_three_tiers():
    """定位三级：和风 Geo 失败→高德 geocode；再失败→外部坐标兜底；海外坐标丢弃。"""
    with _Ctx(qgeo=lambda n: None,
              geo=lambda n: {"latitude": 30.57, "longitude": 104.06, "formatted_address": "四川省成都市"}):
        r = wg.weather_gateway.forecast(destination="成都", days=2)
    assert r["ok"] is True and r["loc_source"] == "amap", r

    with _Ctx(qgeo=lambda n: None, geo=lambda n: None):
        r2 = wg.weather_gateway.forecast(destination="某地", latitude=30.5, longitude=104.0, days=2)
    assert r2["loc_source"] == "fallback", r2

    with _Ctx(qgeo=lambda n: None, geo=lambda n: None):
        r3 = wg.weather_gateway.forecast(destination="某地", latitude=48.8, longitude=2.3, days=2)
    assert r3["ok"] is False and "定位失败" in r3["error"], f"海外坐标必须拒绝: {r3}"


def test_day_budget_per_source():
    """和风按请求天数给足；高德备源被 4 天硬上限截断。"""
    many = [
        {"fxDate": f"2026-09-{9 + i:02d}", "textDay": "晴", "textNight": "晴",
         "tempMax": "25", "tempMin": "18", "precip": "0.0", "windDirDay": "北", "windScaleDay": "1"}
        for i in range(10)
    ]

    def qw_many(loc, days):
        return [
            {"date": r["fxDate"], "desc": r["textDay"], "night_desc": r["textNight"],
             "temp_max": r["tempMax"], "temp_min": r["tempMin"], "precip": r["precip"],
             "wind": "北风1级", "source": "qweather"}
            for r in many[:days]
        ]

    with _Ctx(qweather=qw_many, amap=_norm_amap):
        r = wg.weather_gateway.forecast(destination="成都", days=7)
    assert len(r["days"]) == 7, f"和风应给足 7 天，实际 {len(r['days'])}"

    def amap_many(loc, days):
        rows = [
            {"date": f"2026-09-{9 + i:02d}", "desc": "晴", "night_desc": "晴",
             "temp_max": "25", "temp_min": "18", "precip": None, "wind": "北风1级", "source": "amap"}
            for i in range(10)
        ]
        return rows[: min(days, 4)]

    with _Ctx(qweather=lambda loc, days: [], amap=amap_many):
        r2 = wg.weather_gateway.forecast(destination="成都", days=7)
    assert len(r2["days"]) <= 4, f"高德硬上限 4 天，实际 {len(r2['days'])}"


def test_same_field_set_across_sources():
    """两源输出字段集合一致——上游只认一份归一化结构。"""
    with _Ctx(qweather=_norm_qw, amap=_norm_amap):
        a = wg.weather_gateway.forecast(destination="成都", days=2)["days"][0]
    with _Ctx(qweather=lambda loc, days: [], amap=_norm_amap):
        b = wg.weather_gateway.forecast(destination="成都", days=2)["days"][0]
    assert set(a.keys()) == set(b.keys()) == set(wg._DAY_FIELDS), (a.keys(), b.keys())


def main() -> int:
    tests = [
        test_primary_ok_no_degrade,
        test_primary_fail_degrades_to_backup,
        test_no_fallback_returns_error_not_empty,
        test_locate_three_tiers,
        test_day_budget_per_source,
        test_same_field_set_across_sources,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n汇总：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
