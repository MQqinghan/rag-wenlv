# -*- coding: utf-8 -*-
"""
行程规划服务离线单测：验证 POI 事实核查（坐标修正/幻觉替换/天数校验）。
不发任何真实 HTTP 请求：mock 高德检索层。

运行：cd 项目根 && .venv\\Scripts\\python.exe test\\test_trip_plan.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.schemas.trip_plan import Attraction, Location, TripPlan, TripRequest
from app.rag.trip_plan import trip_planner_service as svc


def _fake_search(keywords: str, city: str, **_) -> list[dict]:
    """离线假高德检索：按关键词返回固定候选。"""
    if "故宫" in keywords:
        return [
            {"id": "poi-1", "name": "故宫博物院", "address": "北京市东城区景山前街4号",
             "location": "116.397128,39.916527", "type": "风景名胜", "typecode": "110200", "tel": ""}
        ]
    if "西湖" in keywords:
        return [
            {"id": "poi-2", "name": "杭州西湖风景名胜区", "address": "杭州市西湖区龙井路1号",
             "location": "120.153576,30.232125", "type": "风景名胜", "typecode": "110200", "tel": ""}
        ]
    if keywords == "景点":
        return [
            {"id": "poi-9", "name": "城市地标广场", "address": "市中心", "location": "116.400000,39.920000",
             "type": "风景名胜", "typecode": "110200", "tel": ""}
        ]
    # 未命中（幻觉景点）：返回空 → 顶替失败将移除
    return []


def _sample_plan() -> TripPlan:
    far_loc = Location(longitude=130.0, latitude=45.0)  # 坐标严重偏差，应被修正
    fake_loc = Location(longitude=121.0, latitude=31.0)  # 名字是编造的，应被顶替
    return TripPlan(
        city="北京",
        start_date=(date.today()).strftime("%Y-%m-%d"),
        end_date=(date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        days=[
            {
                "date": (date.today()).strftime("%Y-%m-%d"),
                "day_index": 0,
                "description": "第一日",
                "transportation": "公共交通",
                "accommodation": "舒适型酒店",
                "attractions": [
                    {"name": "故宫博物院", "address": "旧地址", "location": far_loc, "description": "x"},
                    {"name": "西湖幻境（编造）", "address": "某处", "location": fake_loc, "description": "x"},
                ],
            }
        ],
        weather_info=[],
        overall_suggestions="默认建议",
        budget=None,
    )


def main() -> int:
    # 用假检索替换高德层
    svc._amap_text_search = _fake_search  # type: ignore[assignment]
    svc.amap_gateway.geocode = lambda *a, **k: None  # 坐标全部来自 location 字段

    plan = _sample_plan()
    plan = svc.verify_plan(plan, "北京")
    day0 = plan.days[0]
    kept_names = [a.name for a in day0.attractions]

    # 1) 真实景点（故宫博物院）坐标被高德坐标覆盖（116.397, 39.916）
    palace = next(a for a in day0.attractions if a.name == "故宫博物院")
    assert abs(palace.location.longitude - 116.397128) < 0.01, palace.location
    # 2) 编造景点“西湖幻境（编造）”不存在于北京 → 顶替/移除，绝不能原样保留
    assert "西湖幻境（编造）" not in kept_names, kept_names
    # 3) 若顶替成功则保留高德真实候选（杭州西湖在北京检索不到 → 本用例应被移除）
    assert len(day0.attractions) >= 1, "至少保留一个可核查景点"
    # 4) 全程有【数据保障】备注
    assert "【数据保障】" in plan.overall_suggestions, plan.overall_suggestions

    # 5) 全天被清空场景：整日景点全幻觉 → 用通用“景点”回填
    plan2 = _sample_plan()
    plan2.days[0].attractions = [
        Attraction(name="不存在的地方ZZZ", location=Location(longitude=121.0, latitude=31.0),
                   address="x", description="y")
        for _ in range(3)
    ]
    plan2 = svc.verify_plan(plan2, "北京")
    assert len(plan2.days[0].attractions) >= 1 and "城市地标广场" in [a.name for a in plan2.days[0].attractions], plan2.days[0].attractions

    # 6) Location.from_text 解析
    loc = Location.from_text("116.39,39.90")
    assert loc and abs(loc.longitude - 116.39) < 1e-6 and abs(loc.latitude - 39.90) < 1e-6
    assert Location.from_text("bad") is None

    # 7) haversine 单位：纬度 39° 处 1° 经度 ≈ 86 km（111*cos39°）
    d = svc.haversine_m(116.0, 39.0, 117.0, 39.0)
    assert 80000 < d < 90000, d

    print("✅ test_trip_plan.py 全部通过：坐标修正 / 幻觉替换移除 / 整天空回填 / 解析工具")
    return 0


if __name__ == "__main__":
    sys.exit(main())
