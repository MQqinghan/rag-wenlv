# -*- coding: utf-8 -*-
"""行程规划·往返交通段离线单测。

只验证服务层 collect_routes / _rail_to_info / _route_to_info / collect_routes_text /
_is_rail_mode，不发任何真实 HTTP/MCP：mock 模块级 rail_gateway 与 amap_gateway。
（对应任务清单 #7 的「验证」闭环——原 trip 单测不覆盖路线功能。）

运行：cd 项目根 && .venv\\Scripts\\python.exe test\\test_trip_route.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.schemas.trip_plan import TripRequest
from app.rag.trip_plan import trip_planner_service as svc

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name} {extra}")


class FakeRail:
    """离线假铁路网关：可控 ok / 车次列表，记录调用。"""

    def __init__(self, ok: bool = True, trains=None):
        self.ok = ok
        self.trains = trains if trains is not None else [
            {"code": "G2942", "from_station": "深圳北", "to_station": "成都东",
             "depart": "08:00", "arrive": "15:32", "duration": "7:32",
             "train_class": "高速动车", "prices": {"二等座": 813.5, "一等座": 1281.5}},
            {"code": "K586", "duration": "28:10", "train_class": "普快", "prices": {}},
        ]
        self.calls = []

    def query_trains(self, origin, destination, travel_date):
        self.calls.append((origin, destination, travel_date))
        if not self.ok:
            return {"ok": False, "trains": [], "error": "mcp down"}
        return {"ok": True, "trains": self.trains, "codes": [t["code"] for t in self.trains]}


class FakeAmap:
    """离线假高德网关：可控路线/坐标，记录调用。"""

    def __init__(self, route=None, geo=None):
        self.route = route if route is not None else {"duration": 7200, "distance": 150000}
        self.geo = geo if geo is not None else {"longitude": 104.06, "latitude": 30.67}
        self.route_calls = []
        self.geo_calls = []

    def plan_route(self, origin, destination, origin_city=None, destination_city=None, route_type=None):
        self.route_calls.append((origin, destination, route_type))
        return self.route

    def geocode(self, city, *args, **kwargs):
        self.geo_calls.append(city)
        return self.geo


def _req(**kw) -> TripRequest:
    base = dict(city="成都", start_date="2026-09-14", end_date="2026-09-16",
                travel_days=3, transportation="高铁", accommodation="经济型酒店")
    base.update(kw)
    return TripRequest(**base)


def test_is_rail_mode():
    print("[1] _is_rail_mode 交通方式判定")
    check("高铁→rail", svc._is_rail_mode("高铁") is True)
    check("公共交通→rail", svc._is_rail_mode("公共交通") is True)
    check("动车→rail", svc._is_rail_mode("动车") is True)
    check("自驾→非rail", svc._is_rail_mode("自驾") is False)
    check("空→非rail", svc._is_rail_mode("") is False)


def test_origin_empty():
    print("[2] origin 为空 → None（向后兼容，仅市内行程）")
    svc.rail_gateway = FakeRail()
    svc.amap_gateway = FakeAmap()
    check("origin='' → None", svc.collect_routes(_req(origin="")) is None)
    check("origin 缺省 → None", svc.collect_routes(_req()) is None)
    check("未触发铁路调用", svc.rail_gateway.calls == [])


def test_rail_success():
    print("[3] 铁路成功 → 去程/返程两条 RouteInfo（真实车次+票价）")
    fr = FakeRail(ok=True)
    svc.rail_gateway = fr
    svc.amap_gateway = FakeAmap()
    routes = svc.collect_routes(_req(origin="深圳", transportation="高铁"))
    check("返回 2 段", routes is not None and len(routes) == 2, str(routes))
    check("leg 顺序 去程/返程", [r["leg"] for r in routes] == ["去程", "返程"])
    check("去程 深圳→成都", routes[0]["origin"] == "深圳" and routes[0]["destination"] == "成都")
    check("返程 成都→深圳", routes[1]["origin"] == "成都" and routes[1]["destination"] == "深圳")
    check("mode=高铁", routes[0]["mode"] == "高铁")
    check("取最快班次 7:32", "7小时32分" in routes[0]["duration_text"], routes[0]["duration_text"])
    check("票价透传+12306声明", "813.5" in routes[0]["cost_note"] and "12306" in routes[0]["cost_note"], routes[0]["cost_note"])
    check("铁路不编造里程", routes[0]["distance_text"] == "")
    check("去返各查一次", len(fr.calls) == 2 and fr.calls[0][:2] == ("深圳", "成都") and fr.calls[1][:2] == ("成都", "深圳"))


def test_rail_fallback_transit():
    print("[4] 铁路失败 → 高德 transit 软降级")
    svc.rail_gateway = FakeRail(ok=False)
    fa = FakeAmap()
    svc.amap_gateway = fa
    routes = svc.collect_routes(_req(origin="深圳", transportation="高铁"))
    check("返回 2 段", routes is not None and len(routes) == 2, str(routes))
    check("mode=公共交通", routes[0]["mode"] == "公共交通")
    check("含高德估算声明", "高德" in routes[0]["cost_note"], routes[0]["cost_note"])
    check("走 transit 路线", bool(fa.route_calls) and fa.route_calls[0][2] == "transit", str(fa.route_calls))
    check("耗时格式化", "约 2 小时" in routes[0]["duration_text"], routes[0]["duration_text"])
    check("距离格式化", "150" in routes[0]["distance_text"], routes[0]["distance_text"])


def test_driving():
    print("[5] 自驾 → 高德驾车（先地理编码取坐标）")
    fa = FakeAmap()
    svc.amap_gateway = fa
    svc.rail_gateway = FakeRail()
    routes = svc.collect_routes(_req(origin="重庆", transportation="自驾"))
    check("返回 2 段", routes is not None and len(routes) == 2, str(routes))
    check("mode=自驾", routes[0]["mode"] == "自驾")
    check("驾车 route_type", all(c[2] == "driving" for c in fa.route_calls), str(fa.route_calls))
    check("已地理编码", len(fa.geo_calls) >= 2, str(fa.geo_calls))
    check("坐标传参 lng,lat", "104.060000,30.670000" in fa.route_calls[0][0], fa.route_calls[0][0])


def test_driving_geo_fail():
    print("[6] 自驾·地理编码失败 → 不抛异常、返回 None（软失败）")

    class _NoGeo(FakeAmap):
        def geocode(self, city, *a, **k):
            self.geo_calls.append(city)
            return None

    svc.amap_gateway = _NoGeo()
    routes = svc.collect_routes(_req(origin="重庆", transportation="自驾"))
    check("无路线→None", routes is None)


def test_routes_text():
    print("[7] collect_routes_text 渲染")
    check("None→''", svc.collect_routes_text(None) == "")
    check("[]→''", svc.collect_routes_text([]) == "")
    txt = svc.collect_routes_text([
        {"leg": "去程", "mode": "高铁", "origin": "深圳", "destination": "成都",
         "duration_text": "约 7小时32分", "distance_text": "", "cost_note": "参考票价：二等座 813.5 元（以 12306 官方为准）"},
        {"leg": "返程", "mode": "高铁", "origin": "成都", "destination": "深圳",
         "duration_text": "约 7小时40分", "distance_text": "", "cost_note": "票价以 12306 官方为准"},
    ])
    check("含【去程】", "【去程】" in txt)
    check("含【返程】", "【返程】" in txt)
    check("含箭头", "深圳 → 成都" in txt)
    check("含耗时", "耗时：约 7小时32分" in txt)
    check("含票价", "813.5" in txt)


if __name__ == "__main__":
    for fn in (test_is_rail_mode, test_origin_empty, test_rail_success,
               test_rail_fallback_transit, test_driving, test_driving_geo_fail,
               test_routes_text):
        fn()
    print(f"\n==== 结果：{_PASS} 通过 / {_FAIL} 失败 ====")
    sys.exit(1 if _FAIL else 0)
