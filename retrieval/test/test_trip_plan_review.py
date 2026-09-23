# -*- coding: utf-8 -*-
"""行程规划子图 · 离线单测（零外部服务、零 LLM 调用、零 Redis）。

覆盖 app/process/trip_plan/agent/nodes/node_budget_review.py 与 node_review.py：
  1. node_budget_review：预算核查（确定性算术）
       - 无预算约束 → 放行（enabled=False）
       - 在预算内   → over=0
       - 超预算     → over / over_percent 正确
       - 异常       → fail-open 返回空 dict（绝不拦链路）
  2. node_review：冲突检测 + 有界自动修
       - 天数多于请求天数 → 裁剪
       - 日期与 day_index 重排对齐出发日序列
       - 跨天重复景点去重
       - 单日游含超长城际交通 → 提示
       - 合并 node_budget_review 的超支提示
       - 异常 / 空行程 → fail-open（返回 {}，不抛错）
  3. 并行安全：budget_review 只读 plan_raw，review 只读 trip_plan + budget_review，
       二者无共享写键（验证并行节点式拓扑下互不污染）。

设计约束（主人确认）：生成速度 + 准确性优先 → 两节点均为确定性规则、零额外 LLM，
与 node_verify_poi 并行挂在 node_plan_itinerary 之后，零墙钟开销。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 控制台为 GBK 时，✅/❌ 等符号会让 print 抛 UnicodeEncodeError（门禁外直接跑时的高频坑）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 强制 memory 后端：单测不依赖 Redis 可达性（llm_cache 在 import 期读取该变量）
os.environ["CACHE_BACKEND"] = "memory"

from app.process.trip_plan.agent.nodes.node_budget_review import node_budget_review  # noqa: E402
from app.process.trip_plan.agent.nodes.node_review import node_review  # noqa: E402
from app.shared.schemas.trip_plan import (  # noqa: E402
    TripPlan,
    DayPlan,
    Attraction,
    Budget,
    RouteInfo,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def _plan_dict(n_days: int, start: str = "2026-09-20", per_day: int = 2,
               dup_cross_day: bool = False, bad_dates: bool = False) -> dict:
    """构造合法 TripPlan dict（可含跨天重复 / 错位日期，用于冲突测试）。"""
    days = []
    seen = []
    for i in range(n_days):
        names = [f"A{i * 2 + 1}", f"A{i * 2 + 2}"]
        if dup_cross_day and i > 0:
            names = seen[:1] + names  # 第 2 天起复刻第 1 天首景点 → 跨天重复
        seen.extend(names)
        date = "2099-01-01" if bad_dates else start
        days.append(DayPlan(
            date=date, day_index=i,
            attractions=[Attraction(name=n, address="x", visit_duration=120) for n in names],
        ))
    plan = TripPlan(
        city="北京", start_date=start, end_date=start, days=days,
        budget=Budget(total=1000), routes=[],
    )
    return plan.model_dump(mode="json")


# ============================================================
# 1. node_budget_review
# ============================================================
print("== 1. node_budget_review（预算核查）==")
# 无预算约束
s = {"session_id": "b0", "is_stream": False,
     "trip_request": {"budget": 0}, "plan_raw": {"budget": {"total": 9000}}}
r = node_budget_review(s)
check("无预算约束 → enabled=False 放行", r.get("budget_review", {}).get("enabled") is False)

# 预算内
s = {"session_id": "b1", "is_stream": False,
     "trip_request": {"budget": 5000}, "plan_raw": {"budget": {"total": 4200}}}
r = node_budget_review(s)
br = r.get("budget_review", {})
check("预算内 → enabled=True 且 over=0", br.get("enabled") is True and br.get("over") == 0)

# 超预算
s = {"session_id": "b2", "is_stream": False,
     "trip_request": {"budget": 5000}, "plan_raw": {"budget": {"total": 6800}}}
r = node_budget_review(s)
br = r.get("budget_review", {})
check("超预算 → enabled=True", br.get("enabled") is True)
check("超预算 over=1800", br.get("over") == 1800)
check("超预算 over_percent=36", br.get("over_percent") == 36)
check("超预算含中文提示", "超预算" in (br.get("note") or ""))

# 异常 fail-open（plan_raw 非 dict，触发 KeyError/TypeError 被吞）
s = {"session_id": "b3", "is_stream": False,
     "trip_request": {"budget": 5000}, "plan_raw": None}
r = node_budget_review(s)
check("异常 → fail-open 返回含 budget_review 键（不抛错）", isinstance(r, dict) and "budget_review" in r)

# ============================================================
# 2. node_review（冲突检测 + 有界自动修）
# ============================================================
print("== 2. node_review（冲突检测 + 有界自动修）==")
_base = {"session_id": "r0", "is_stream": False, "pois": [],
         "weather_text": "", "hotels_text": "", "routes_text": "",
         "budget_review": {}}

# a. 天数裁剪：请求 3 天，规划给 5 天
s = dict(_base, trip_request={"travel_days": 3, "start_date": "2026-09-20"},
         trip_plan=_plan_dict(5))
out = node_review(s)
check("天数多于请求 → 裁剪到 3 天", len(out["trip_plan"]["days"]) == 3)
check("裁剪后写回 trip_plan 与 rendered_text", bool(out.get("rendered_text")))

# b. 日期重排 + 跨天去重
s = dict(_base, trip_request={"travel_days": 3, "start_date": "2026-09-20"},
         trip_plan=_plan_dict(3, dup_cross_day=True, bad_dates=True))
out = node_review(s)
dates = [d["date"] for d in out["trip_plan"]["days"]]
day_idx = [d["day_index"] for d in out["trip_plan"]["days"]]
check("日期重排为出发日序列", dates == ["2026-09-20", "2026-09-21", "2026-09-22"])
check("day_index 重排为 0/1/2", day_idx == [0, 1, 2])
# 跨天去重：首景点只应出现一次（A1）
all_names = [a["name"] for d in out["trip_plan"]["days"] for a in d["attractions"]]
check("跨天重复景点已去重", all_names.count("A1") == 1)

# c. 单日游含超长城际交通 → 提示
routes = [RouteInfo(leg="去程", mode="高铁", origin="成都", destination="北京",
                    duration_text="约 13 小时", distance_text="约 1800 公里").model_dump(mode="json")]
s = dict(_base, trip_request={"travel_days": 1, "start_date": "2026-09-20"},
         trip_plan=_plan_dict(1, per_day=1), )
s["trip_plan"]["routes"] = routes
out = node_review(s)
note = out["trip_plan"].get("overall_suggestions") or ""
check("单日游超长交通 → 提示确认到达后游玩时间", "单日游含超长城际交通" in note)

# d. 合并预算核查提示
s = dict(_base, trip_request={"travel_days": 2, "start_date": "2026-09-20"},
         trip_plan=_plan_dict(2),
         budget_review={"enabled": True, "note": "已超预算 1800 元（目标 5000 元）"})
out = node_review(s)
note = out["trip_plan"].get("overall_suggestions") or ""
check("合并 node_budget_review 超支提示", "冲突核查" in note and "已超预算 1800 元" in note)

# e. fail-open：空/非法 trip_plan
s = dict(_base, trip_request={}, trip_plan={})
check("空 trip_plan → 返回 {} 不抛错", node_review(s) == {})
s = dict(_base, trip_request={}, trip_plan="不是dict")
check("非法 trip_plan → 返回 {} 不抛错", node_review(s) == {})

# ============================================================
# 3. 并行安全：budget_review 与 review 读写键互不污染
# ============================================================
print("== 3. 并行节点式拓扑键隔离 ==")
# budget_review 只读 plan_raw（不读 trip_plan）；review 只读 trip_plan + budget_review。
# 并行下两节点同时运行，各自输出独立键，不应相互覆盖。
shared = {
    "session_id": "p0", "is_stream": False,
    "trip_request": {"budget": 5000, "travel_days": 3, "start_date": "2026-09-20"},
    "plan_raw": {"budget": {"total": 6800}},
    "trip_plan": _plan_dict(5, bad_dates=True),
    "pois": [], "weather_text": "", "hotels_text": "", "routes_text": "",
}
br_out = node_budget_review(shared)
rv_out = node_review(shared)
check("budget_review 只产出 budget_review 键", set(br_out.keys()) == {"budget_review"})
check("review 产出 trip_plan + rendered_text（+error）",
      "trip_plan" in rv_out and "rendered_text" in rv_out)
check("budget_review 超支结论独立成立", br_out["budget_review"]["over"] == 1800)
check("review 仍完成天数裁剪（读自身 trip_plan）",
      len(rv_out["trip_plan"]["days"]) == 3)

print()
print(f"== trip_plan_review 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
