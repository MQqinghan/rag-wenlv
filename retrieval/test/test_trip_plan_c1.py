# -*- coding: utf-8 -*-
"""
行程规划 C1 主体离线单测：POI 核查并发化 + 「核查失败 ≠ 查无此地」。

不发任何真实 HTTP 请求：mock `_amap_text_search` 与 `amap_gateway.last_outcome`。

覆盖：
1. 并发核查结果与串行一致（真实景点保留/坐标修正，编造景点顶替或移除）；
2. **限流注入**：检索返回空但 last_outcome=rate_limited → 真实景点必须保留并标注未核实（旧逻辑会误删）；
3. `TRIP_VERIFY_WORKERS=1`（串行）与 =4（并发）结果一致，并发开关不改变语义。

运行：cd 项目根 && .venv\\Scripts\\python.exe test\\test_trip_plan_c1.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.schemas.trip_plan import Attraction, Location, TripPlan
from app.rag.trip_plan import trip_planner_service as svc

_REAL_POI = {
    "id": "poi-1",
    "name": "故宫博物院",
    "address": "北京市东城区景山前街4号",
    "location": "116.397128,39.916527",
    "type": "风景名胜",
    "typecode": "110200",
}
_FILL_POI = {
    "id": "poi-9",
    "name": "城市地标广场",
    "address": "市中心",
    "location": "116.400000,39.920000",
    "type": "风景名胜",
    "typecode": "110200",
}


def _fake_search(keywords: str, city: str, **_) -> list[dict]:
    """离线假检索：命中"故宫"返回真实 POI；"景点"返回回填候选；其余（含编造名）返回空。"""
    if "故宫" in keywords:
        return [_REAL_POI]
    if keywords == "景点":
        return [_FILL_POI]
    return []


def _plan(names: list[str]) -> TripPlan:
    return TripPlan(
        city="北京",
        start_date=date.today().strftime("%Y-%m-%d"),
        end_date=(date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        days=[
            {
                "date": date.today().strftime("%Y-%m-%d"),
                "day_index": 0,
                "description": "第一日",
                "attractions": [
                    {"name": n, "address": "x", "location": Location(longitude=121.0, latitude=31.0)}
                    for n in names
                ],
            }
        ],
        weather_info=[],
        overall_suggestions="默认建议",
        budget=None,
    )


def test_concurrent_verify_matches_serial():
    """场景1+3：并发核查结果与串行一致（真实保留并修正坐标，编造景点顶替/移除）。"""
    names = ["故宫博物院"] * 4 + ["编造景点A", "编造景点B"]
    svc._amap_text_search = _fake_search  # type: ignore[assignment]
    svc.amap_gateway.geocode = lambda *a, **k: None  # type: ignore[assignment]

    results = {}
    for workers in (1, 4):
        svc.TRIP_VERIFY_WORKERS = workers
        plan = svc.verify_plan(_plan(names), "北京")
        results[workers] = [a.name for a in plan.days[0].attractions]

    assert results[1] == results[4], f"串行与并发结果应一致：{results}"
    serial = results[1]
    assert serial.count("故宫博物院") == 4, f"4 个真实景点应全部保留，实际 {serial}"
    assert "编造景点A" not in serial and "编造景点B" not in serial, f"编造景点不应原样保留：{serial}"


def test_rate_limited_keeps_real_pois():
    """场景2：限流导致检索为空 → 真实景点必须保留并标注"未核实"，绝不误删。"""
    names = ["故宫博物院", "天坛公园", "颐和园"]
    # 模拟限流：检索始终返回空，且网关结局标记为 rate_limited
    svc._amap_text_search = lambda *a, **k: []  # type: ignore[assignment]
    svc.amap_gateway.geocode = lambda *a, **k: None  # type: ignore[assignment]
    svc.amap_gateway.last_outcome = lambda: "rate_limited"  # type: ignore[assignment]
    svc.TRIP_VERIFY_WORKERS = 4

    plan = svc.verify_plan(_plan(names), "北京")
    kept = [a.name for a in plan.days[0].attractions]

    assert kept == names, f"限流时真实景点必须原样保留（旧逻辑会误删），实际：{kept}"
    assert "未能核实" in (plan.overall_suggestions or ""), (
        f"应标注未核实，实际建议：{plan.overall_suggestions}"
    )


def test_no_match_removes_fake():
    """对照：非限流的"查无此地"仍应移除/顶替（确认没有矫枉过正）。"""
    svc._amap_text_search = _fake_search  # type: ignore[assignment]
    svc.amap_gateway.geocode = lambda *a, **k: None  # type: ignore[assignment]
    svc.amap_gateway.last_outcome = lambda: "ok"  # type: ignore[assignment]
    svc.TRIP_VERIFY_WORKERS = 4

    plan = svc.verify_plan(_plan(["编造景点A"]), "北京")
    kept = [a.name for a in plan.days[0].attractions]
    assert "编造景点A" not in kept, f"查无此地的编造景点仍应移除/顶替：{kept}"


def main() -> int:
    tests = [
        test_concurrent_verify_matches_serial,
        test_rate_limited_keeps_real_pois,
        test_no_match_removes_fake,
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
