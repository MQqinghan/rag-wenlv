# -*- coding: utf-8 -*-
"""
行程规划 2.1 事项 C 收尾离线单测：C2（偏好关键词采集并发）+ C1-fix（每日景点下限补齐）。

不发任何真实 HTTP 请求：
- C2：mock `amap_gateway.search_poi`，记录「并发活跃峰值」证明多关键词并行；校验去重保序。
- C1-fix：mock `_amap_text_search`，验证「当日不足下限 → 用高德真实 POI 补齐」与
  「已达下限不重排、不覆盖」，且整天空场景仍按原语义回填。

运行：cd 项目根 && .venv\\Scripts\\python.exe test\\test_trip_plan_c2.py
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.schemas.trip_plan import Location, TripPlan, TripRequest
from app.rag.trip_plan import trip_planner_service as svc

# ---------------------------------------------------------------- 夹具
_POOL = {
    "故宫博物院": {"id": "p-1", "name": "故宫博物院", "address": "北京市东城区", "location": "116.397128,39.916527", "type": "景区"},
    "天坛公园": {"id": "p-2", "name": "天坛公园", "address": "北京市东城区天坛路", "location": "116.410555,39.882160", "type": "景区"},
    "颐和园": {"id": "p-3", "name": "颐和园", "address": "北京市海淀区", "location": "116.275392,39.999025", "type": "景区"},
}
_FILLS = [
    {"id": "f-1", "name": "城市地标广场", "address": "市中心", "location": "116.400000,39.920000", "type": "景区"},
    {"id": "f-2", "name": "朝阳公园", "address": "朝阳区", "location": "116.478088,39.933757", "type": "公园"},
    {"id": "f-3", "name": "北海公园", "address": "西城区", "location": "116.389746,39.925437", "type": "公园"},
]


def _fake_text_search(keywords: str, city: str, **_) -> list[dict]:
    """核查层假检索：名称命中返回对应真实 POI；'景点'返回回填候选。"""
    if keywords == "景点":
        return list(_FILLS)
    return [p for n, p in _POOL.items() if n in keywords]


def _day(names: list[str]) -> dict:
    return {
        "date": date.today().strftime("%Y-%m-%d"),
        "day_index": 0,
        "description": "第一日",
        "attractions": [
            {"name": n, "address": "x", "location": Location(longitude=121.0, latitude=31.0)}
            for n in names
        ],
    }


def _plan(names: list[str]) -> TripPlan:
    return TripPlan(
        city="北京",
        start_date=date.today().strftime("%Y-%m-%d"),
        end_date=(date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        days=[_day(names)],
        weather_info=[],
        overall_suggestions="默认建议",
        budget=None,
    )


# ================================================================ C2
def _run_collect_pois_parallel_probe() -> tuple[int, list[str]]:
    """返回 (并发活跃峰值, 去重保序后的名称列表)。"""
    state = {"active": 0, "max": 0, "lock": threading.Lock()}
    _by_kw = {
        "历史文化": [_POOL["故宫博物院"], _POOL["天坛公园"]],
        "自然风光": [_POOL["颐和园"], _POOL["故宫博物院"]],  # 与上一条重叠 → 验证去重
        "美食": [{"id": "f-0", "name": "南锣鼓巷", "address": "东城区", "location": "116.403850,39.938129", "type": "街区"}],
    }

    def fake_search(keywords: str, city=None, citylimit=True, offset=8, **_) -> list[dict]:
        with state["lock"]:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        try:
            time.sleep(0.2)  # 拉长单次调用，便于观测并行度
            return list(_by_kw.get(keywords, []))
        finally:
            with state["lock"]:
                state["active"] -= 1

    orig = svc.amap_gateway.search_poi
    svc.amap_gateway.search_poi = fake_search  # type: ignore[assignment]
    try:
        svc.TRIP_COLLECT_POIS_WORKERS = 3
        req = TripRequest(
            city="北京",
            start_date=date.today().strftime("%Y-%m-%d"),
            end_date=date.today().strftime("%Y-%m-%d"),
            travel_days=1,
            transportation="公共交通",
            accommodation="舒适型酒店",
            preferences=["历史文化", "自然风光", "美食"],
        )
        pois = svc.collect_pois(req)
        return state["max"], [p["name"] for p in pois]
    finally:
        svc.amap_gateway.search_poi = orig  # type: ignore[assignment]


def test_c2_keyword_search_concurrent():
    peak, names = _run_collect_pois_parallel_probe()
    assert peak >= 2, f"多关键词应并发执行（活跃峰值应≥2），实际峰值={peak}"
    # 去重保序：每个关键词内部保持返回顺序；跨关键词重复项只保留首次出现
    assert names == ["故宫博物院", "天坛公园", "颐和园", "南锣鼓巷"], names
    assert len(names) == len(set(names)), f"不应有重复景点：{names}"


def test_c2_single_keyword_still_works():
    """单关键词场景（无偏好）不降级、不走并发路径也能正常返回。"""
    orig = svc.amap_gateway.search_poi
    svc.amap_gateway.search_poi = (
        lambda keywords, city=None, citylimit=True, offset=8, **_: list(_POOL.values())
    )  # type: ignore[assignment]
    try:
        req = TripRequest(
            city="北京",
            start_date=date.today().strftime("%Y-%m-%d"),
            end_date=date.today().strftime("%Y-%m-%d"),
            travel_days=1,
            transportation="公共交通",
            accommodation="舒适型酒店",
            preferences=[],
        )
        pois = svc.collect_pois(req)
        assert len(pois) == 3 and pois[0]["name"] == "故宫博物院"
    finally:
        svc.amap_gateway.search_poi = orig  # type: ignore[assignment]


# ================================================================ C1-fix
def _patch_verify_layer():
    svc._amap_text_search = _fake_text_search  # type: ignore[assignment]
    svc.amap_gateway.geocode = lambda *a, **k: None  # type: ignore[assignment]
    svc.amap_gateway.last_outcome = lambda: "ok"  # type: ignore[assignment]
    svc.TRIP_VERIFY_WORKERS = 4


def test_c1fix_topup_when_day_below_floor():
    """当日仅 1 个真实景点 → 核查后补齐到 2，且保留原景点在前。"""
    _patch_verify_layer()
    plan = svc.verify_plan(_plan(["故宫博物院"]), "北京")
    names = [a.name for a in plan.days[0].attractions]
    assert names[:1] == ["故宫博物院"], names
    assert len(names) >= 2, f"不足下限应被补齐，实际 {names}"
    assert "补齐当日景点下限" in (plan.overall_suggestions or ""), plan.overall_suggestions


def test_c1fix_empty_day_backfills_to_floor():
    """整天被清空（0 个）→ 回填到每日下限 2，语义消息仍保留。"""
    _patch_verify_layer()
    plan = svc.verify_plan(_plan([]), "北京")
    names = [a.name for a in plan.days[0].attractions]
    assert len(names) >= 2, f"整天空应回填到至少 2 个，实际 {names}"
    assert "原行程全部无法核实" in (plan.overall_suggestions or ""), plan.overall_suggestions


def test_c1fix_day_meeting_floor_untouched():
    """当日已满足下限（2 个真实景点）→ 不追加、不覆盖、顺序不变。"""
    _patch_verify_layer()
    plan = svc.verify_plan(_plan(["故宫博物院", "天坛公园"]), "北京")
    names = [a.name for a in plan.days[0].attractions]
    assert names == ["故宫博物院", "天坛公园"], names
    # 无任何数据保障动作 → 不追加每日补齐/移除文案（坐标修正是允许的正常动作）
    assert "补齐当日景点下限" not in (plan.overall_suggestions or "")
    assert "无法核实" not in (plan.overall_suggestions or "")


def test_c1fix_topup_never_overfills():
    """当日已 3 个真实景点 → 保持 3 个不动（不得因为候选多而塞更多）。"""
    _patch_verify_layer()
    plan = svc.verify_plan(_plan(["故宫博物院", "天坛公园", "颐和园"]), "北京")
    names = [a.name for a in plan.days[0].attractions]
    assert names == ["故宫博物院", "天坛公园", "颐和园"], names
    assert len(names) == 3


def test_c1fix_consistent_across_serial_and_parallel():
    """C1-fix 补足逻辑在 TRIP_VERIFY_WORKERS=1（串行）与 =4（并发）下结果一致。"""
    _patch_verify_layer()
    results = {}
    for workers in (1, 4):
        svc.TRIP_VERIFY_WORKERS = workers
        plan = svc.verify_plan(_plan(["故宫博物院"]), "北京")
        results[workers] = [a.name for a in plan.days[0].attractions]
    assert results[1] == results[4], f"串行与并发结果应一致：{results}"


def _multi_day_plan(per_day_names: list[list[str]]) -> TripPlan:
    """构造多日行程（每天给定景点名），用于跨天去重验证。"""
    days = []
    for i, names in enumerate(per_day_names):
        days.append({
            "date": (date.today() + timedelta(days=i)).strftime("%Y-%m-%d"),
            "day_index": i,
            "description": f"第{i + 1}日",
            "attractions": [
                {"name": n, "address": "x", "location": Location(longitude=121.0, latitude=31.0)}
                for n in names
            ],
        })
    return TripPlan(
        city="北京",
        start_date=date.today().strftime("%Y-%m-%d"),
        end_date=(date.today() + timedelta(days=len(per_day_names) - 1)).strftime("%Y-%m-%d"),
        days=days,
        weather_info=[],
        overall_suggestions="默认建议",
        budget=None,
    )


def test_m6_topup_no_cross_day_duplicate():
    """
    M6 回归：多天都需补齐时，补齐的景点不得跨天重复。

    修复前 _topup_attractions 只与当天 kept 比对，三天会补到同一批候选的前 2 个，
    出现「同一景点在多个日期重复」；修复后应依次取用不同候选。
    """
    _patch_verify_layer()
    # 候选池给足（3 天 × 每天 2 个 = 6），确保验证的是"跨天去重"而非"候选耗尽"
    pool = [
        {"id": f"g-{i}", "name": n, "address": "x", "location": "116.4,39.9", "type": "景区"}
        for i, n in enumerate(["候选A", "候选B", "候选C", "候选D", "候选E", "候选F"])
    ]
    svc._amap_text_search = lambda kw, city, offset=10: list(pool) if kw == "景点" else []
    plan = svc.verify_plan(_multi_day_plan([["故宫A"], ["天坛B"], ["颐和C"]]), "北京")
    names = [a.name for day in plan.days for a in day.attractions]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, f"补齐景点出现跨天重复：{dup}；每日={[[a.name for a in d.attractions] for d in plan.days]}"
    # 且每天都应达到下限（候选充足时）
    assert all(len(d.attractions) >= svc.TRIP_MIN_ATTRS_PER_DAY for d in plan.days), \
        f"候选充足时每天都应补齐到下限：{[len(d.attractions) for d in plan.days]}"


def test_m6_topup_prefers_shortage_over_duplicate():
    """
    M6 回归：候选耗尽时宁可当日不足下限，也不重复其他天已排的景点。
    """
    _patch_verify_layer()
    only_two = [
        {"id": "z-1", "name": "唯一候选一", "address": "x", "location": "116.4,39.9", "type": "景区"},
        {"id": "z-2", "name": "唯一候选二", "address": "x", "location": "116.5,39.9", "type": "景区"},
    ]
    svc._amap_text_search = lambda kw, city, offset=10: list(only_two) if kw == "景点" else []
    plan = svc.verify_plan(_multi_day_plan([["故宫A"], ["天坛B"], ["颐和C"]]), "北京")
    names = [a.name for day in plan.days for a in day.attractions]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, f"候选耗尽时也必须零重复：{dup}；每日={[[a.name for a in d.attractions] for d in plan.days]}"


def _relax_plan(names: list[str]) -> TripPlan:
    """构造一个"整天仅剩 0 个真实景点"的单日计划，用于验证不硬补、改为美食休闲文本。"""
    return _plan(names)


def test_a_relax_when_scenic_exhausted():
    """
    方案A 回归：该地景点候选已耗尽（无更多真实景点可取）时，
    整天空的日期不得硬抠冷门/低质 POI，应改为在该日 description 里自然带出
    "美食/休闲为主"说明，并保留空景点列表（不伪造"可看景点"）。
    """
    _patch_verify_layer()
    svc._amap_text_search = lambda kw, city, offset=10: []  # 景点候选为空 → 该地无可补景点
    plan = svc.verify_plan(_relax_plan([]), "北京")
    day = plan.days[0]
    assert day.attractions == [], f"无候选时不应硬补景点：{day.attractions}"
    # description 应带出美食/休闲为主的说明
    assert any(k in (day.description or "") for k in ("美食", "休闲", "吃")), day.description
    # 数据保障备注应提示"未硬补/资源较少"
    assert "未硬补景点" in (plan.overall_suggestions or "") or "资源较少" in (plan.overall_suggestions or ""), \
        plan.overall_suggestions


def test_a_keeps_real_when_one_left_no_scenic_extra():
    """
    方案A 回归：某天还有 1 个真实核查通过的景点、但补不出第 2 个真实候选时，
    保留这 1 个真实景点即可，不硬凑第 2 个冷门景点，也不追加美食说明（当天有真实景点可走）。
    """
    _patch_verify_layer()
    # "景点"关键词返回空 → 补不出第二个；但故宫博物院本身能被名称检索到（核查通过）
    def fake(kw, city, offset=10):
        if kw == "景点":
            return []
        return [p for n, p in _POOL.items() if n in kw]
    svc._amap_text_search = fake
    plan = svc.verify_plan(_plan(["故宫博物院"]), "北京")
    names = [a.name for a in plan.days[0].attractions]
    assert names == ["故宫博物院"], names  # 保留真实景点，不补冷门
    assert "未硬补" not in (plan.overall_suggestions or "")  # 有真实景点时不走降档文案


def main() -> int:
    tests = [
        test_c2_keyword_search_concurrent,
        test_c2_single_keyword_still_works,
        test_c1fix_topup_when_day_below_floor,
        test_c1fix_empty_day_backfills_to_floor,
        test_c1fix_day_meeting_floor_untouched,
        test_c1fix_topup_never_overfills,
        test_c1fix_consistent_across_serial_and_parallel,
        test_m6_topup_no_cross_day_duplicate,
        test_m6_topup_prefers_shortage_over_duplicate,
        test_a_relax_when_scenic_exhausted,
        test_a_keeps_real_when_one_left_no_scenic_extra,
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


