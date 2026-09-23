# -*- coding: utf-8 -*-
"""
结构化行程规划评测执行器（第四路 · trip_plan 图）。

与 data/eval/run_eval.py（知识库检索评测）分离：
本脚本直接驱动 app/process/trip_plan/agent/main_graph.invoke_structured，
用真实高德 REST 采集+LLM 规划+POI 事实核查，站客户视角打分：
- 结构完整性：天数对齐 / 每日景点数 / 坐标齐全率；
- 数据真实性：每个景点反查高德名称命中率；
- 可靠性：是否显式失败而非假数据兜底；
- 性能：端到端耗时。

用法（需在项目 .venv、.env 已配 AMAP_API_KEY 与 LLM）：
  python data/eval/run_trip_eval.py                     # 全量
  python data/eval/run_trip_eval.py --limit 1           # 冒烟
  python data/eval/run_trip_eval.py --ids trip-001      # 定向
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def build_request(case: dict) -> dict:
    """用例 → TripRequest dict（日期以今天+offset 起算）。"""
    start = date.today() + timedelta(days=int(case.get("start_offset_days", 1)))
    days = int(case.get("travel_days", 3))
    end = start + timedelta(days=days - 1)
    return {
        "city": case["city"],
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "travel_days": days,
        "transportation": case.get("transportation", "公共交通"),
        "accommodation": case.get("accommodation", "舒适型酒店"),
        "preferences": case.get("preferences", []),
        "free_text_input": case.get("free_text_input", ""),
    }


def reverse_verify_ok(plan: dict) -> dict:
    """把最终行程每个景点反查高德，统计名称命中率（数据真实性抽查）。"""
    from app.infra.amap_gateway import amap_gateway

    total = 0
    hit = 0
    missing_coord = 0
    by_day = []
    for day in plan.get("days", []):
        attrs = day.get("attractions", [])
        day_total = len(attrs)
        day_hit = 0
        for a in attrs:
            total += 1
            loc = a.get("location")
            if not loc or "longitude" not in loc:
                missing_coord += 1
            pois = amap_gateway.search_poi(a.get("name", ""), plan.get("city"), citylimit=True, offset=5)
            name = (a.get("name") or "").strip()
            if any(
                (str(p.get("name") or "").strip())
                and (name in str(p.get("name")) or str(p.get("name")) in name)
                for p in pois
            ):
                hit += 1
                day_hit += 1
        by_day.append({"day_index": day.get("day_index"), "total": day_total, "verified": day_hit})
    return {
        "attraction_total": total,
        "attraction_verified": hit,
        "verify_rate": round(hit / total, 4) if total else 1.0,
        "missing_coord": missing_coord,
        "by_day": by_day,
    }


# 方案A（2026-09-09）：_RELAX_NOTE_POOL 三种补位措辞的可辨识特征词。
# 命中任一组即判定该日为「美食/休闲补位日」——即当地景点资源确实少、
# 已由 trip_planner_service.verify_plan 将当日降档为以吃/逛为主的行程（非硬抠景点）。
# 评测对这类日子的「景点数低于下限」予以结构豁免（配合主人「2.放宽」的取舍）。
_RELAX_KEYWORDS = (
    "慢节奏",
    "赶景点",
    "景点资源较少",
    "以品尝本地小吃",
    "以当地美食与街巷",
    "以寻味探店",
    "美食与休闲",
    "街角漫步",
)


def _day_is_relaxed(desc: str) -> bool:
    """某日 description 是否已被标记为「美食/休闲补位日」。"""
    return any(k in (desc or "") for k in _RELAX_KEYWORDS)


def _judge_structure(plan: dict, case: dict) -> dict:
    """
    方案A放宽（2026-09-09）：由「全局 min_attrs>=期望」改为「逐天判定」。
    某天景点数<期望时，若该日已降档为美食/休闲补位日（description 带补位说明）则豁免；
    否则（既不足又非补位日）才记为该条的结构缺口。避免资源少的正确降档日误伤整条。
    纯函数（不触发真实 API），供 run_case 与离线单测共用。
    """
    travel_days = int(case["travel_days"])
    expect_min = int(case.get("expect_min_attractions_per_day", 2))
    days = plan.get("days", [])
    days_ok = len(days) == travel_days
    shortage_days: list[int] = []
    relaxed_days: list[int] = []
    for _d in days:
        _n = len(_d.get("attractions", []))
        if _n < expect_min:
            if _day_is_relaxed(str(_d.get("description") or "")):
                relaxed_days.append(_d.get("day_index"))
            else:
                shortage_days.append(_n)
    ok = days_ok and not shortage_days
    reasons = []
    if not days_ok:
        reasons.append(f"天数不一致(期望{travel_days}/实际{len(days)})")
    if shortage_days:
        reasons.append(f"存在景点数<{expect_min}且非美食/休闲补位日(最小{min(shortage_days)})")
    return {"ok": ok, "reasons": reasons, "relaxed_days": relaxed_days, "min_attrs": min((len(d.get('attractions', [])) for d in days), default=0)}


def run_case(case: dict) -> dict:
    from app.shared.schemas.trip_plan import TripRequest
    from app.process.trip_plan.agent.main_graph import invoke_structured

    t0 = time.time()
    rec: dict = {"id": case["id"], "note": case.get("note", "")}
    try:
        req = TripRequest(**build_request(case))
        final = invoke_structured(req)
        plan = final["trip_plan"]
        rec["success"] = True
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["city"] = plan.get("city")
        rec["days"] = len(plan.get("days", []))
        rec["expected_days"] = case["travel_days"]
        day_counts = [len(d.get("attractions", [])) for d in plan.get("days", [])]
        rec["day_counts"] = day_counts
        rec["min_attrs_per_day"] = min(day_counts, default=0)
        rec["suggestion_len"] = len(plan.get("overall_suggestions", ""))
        rec["verify"] = reverse_verify_ok(plan)
        rec["data_guarantee_note"] = "【数据保障】" in (plan.get("overall_suggestions") or "")
        # 方案A放宽（2026-09-09）：逐天判定 + 美食/休闲补位日豁免（见 _judge_structure）。
        struct = _judge_structure(plan, case)
        rec["relaxed_days"] = struct["relaxed_days"]
        real_ok = rec["verify"]["verify_rate"] >= 0.9 and rec["verify"]["missing_coord"] == 0
        rec["accept"] = struct["ok"] and real_ok
        rec["reasons"] = list(struct["reasons"])
        if not real_ok:
            rec["reasons"].append(f"POI 核查率{rec['verify']['verify_rate']}或存在缺坐标景点")
    except Exception as e:  # noqa: BLE001
        rec["success"] = False
        rec["accept"] = False
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["error"] = str(e)[:200]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "data" / "eval_cases_trip.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.ids:
        wanted = [s.strip() for s in args.ids.split(",") if s.strip()]
        by_id = {c["id"]: c for c in cases}
        if any(w not in by_id for w in wanted):
            raise SystemExit("--ids 中存在未知用例 id")
        cases = [by_id[w] for w in wanted]
    if args.limit:
        cases = cases[: args.limit]

    print(f"== 结构化行程评测开始：{len(cases)} 条 ==")
    recs = []
    for c in cases:
        print(f"[*] {c['id']} {c['city']} {c['travel_days']}天 ...")
        r = run_case(c)
        recs.append(r)
        if r.get("success"):
            print(
                f"    -> {'PASS' if r.get('accept') else 'FAIL'} 天数={r.get('days')} "
                f"每日景点={r.get('day_counts')} 核查率={r.get('verify', {}).get('verify_rate')} "
                f"耗时={r.get('elapsed_s')}s"
            )
        else:
            print(f"    -> FAIL error={r.get('error')}")

    summary = {
        "total": len(recs),
        "passed": sum(1 for r in recs if r.get("accept")),
        "failed": sum(1 for r in recs if not r.get("accept")),
        "avg_elapsed_s": round(sum(r.get("elapsed_s", 0) for r in recs) / len(recs), 1) if recs else 0,
        "avg_verify_rate": round(
            sum(r.get("verify", {}).get("verify_rate", 0) for r in recs) / len(recs), 4
        ) if recs else 0,
    }
    print("\n== 汇总 ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = args.out or str(ROOT / "output" / "eval_trip" / time.strftime("%Y%m%d_%H%M%S"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "metrics.json").write_text(
        json.dumps({"summary": summary, "cases": recs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 结构化行程评测报告",
        "",
        "| id | 结果 | 天数 | 每日景点 | POI核查率 | 耗时s | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        note = "; ".join(r.get("reasons", [])) or r.get("error") or (r.get("note") or "")
        lines.append(
            f"| {r['id']} | {'✅' if r.get('accept') else '❌'} | {r.get('days', '-')} | "
            f"{r.get('day_counts') or '-'} | {r.get('verify', {}).get('verify_rate', '-')} | "
            f"{r.get('elapsed_s', '-')} | {note[:60]} |"
        )
    lines.append(f"\n> 汇总：{summary}")
    Path(out_dir, "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n== 结果已写入 {out_dir} ==")


if __name__ == "__main__":
    main()

