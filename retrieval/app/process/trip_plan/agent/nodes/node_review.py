# -*- coding: utf-8 -*-
"""行程规划图：node_review。

审核 Agent（冲突检测 + 有界自动修）：
    对 verify_poi 产出的结构化行程做确定性逻辑冲突检测，并尽可能自动修：
      - 天数多于请求天数 → 裁剪多余天；
      - 每日日期与出发日序列不符 → 重排日期与 day_index；
      - 同一景点跨天重复 → 去除后出现的重复项；
      - 单日游含超长城际交通 → 提示确认到达后游玩时间；
      - 合并 node_budget_review 的超支提示。
    空缺日（确定性修复后仍 0 个景点）做至多 1 次局部重排（复用候选 POI），有界防循环。

设计：
    - 主体为规则判定，无额外 LLM 调用，几乎零速度开销；
    - 任何异常一律 fail-open（返回空更新），保留 verify_poi 的行程，绝不把链路拦死。
"""
import sys
import re
import logging

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.shared.schemas.trip_plan import TripPlan, TripRequest
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task

_LOG = logging.getLogger("trip_plan")

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DUR_H_RE = re.compile(r"约\s*(\d+)\s*小时")


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _expected_dates(start_date: str, n: int) -> list:
    try:
        from datetime import date, timedelta
        y, m, d = (int(x) for x in start_date.split("-"))
        s = date(y, m, d)
        return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]
    except Exception:
        return ["" for _ in range(n)]


def _hours(duration_text) -> int:
    m = _DUR_H_RE.search(duration_text or "")
    return int(m.group(1)) if m else 0


def _attr_name(a) -> str:
    if isinstance(a, dict):
        return a.get("name", "") or ""
    return getattr(a, "name", "") or ""


def _dedup_cross_day(plan: TripPlan) -> list:
    seen = set()
    removed = []
    for d in plan.days:
        keep = []
        for a in d.attractions:
            name = _attr_name(a)
            if name and name in seen:
                removed.append(name)
                continue
            if name:
                seen.add(name)
            keep.append(a)
        d.attractions = keep
    return removed


def _maybe_replan_empty_day(plan: TripPlan, state: TripPlanGraphState) -> bool:
    """空缺日（0 个景点）做至多 1 次局部重排：用未使用的候选 POI 重新生成当天。"""
    req = state.get("trip_request") or {}
    try:
        request = TripRequest(**req) if isinstance(req, dict) else req
    except Exception:
        return False
    for i, d in enumerate(plan.days):
        if d.attractions:
            continue
        try:
            used = {
                _attr_name(a)
                for j, d2 in enumerate(plan.days)
                if j != i
                for a in d2.attractions
            }
            unused = [p for p in (state.get("pois") or []) if p.get("name") not in used][:8]
            if not unused:
                return False
            pois_text = svc.collect_pois_text(unused)
            new_day, _cost = svc._run_one_day(
                request, i, pois_text,
                state.get("weather_text", ""),
                state.get("hotels_text", ""),
                state.get("routes_text", ""),
            )
            plan.days[i] = new_day
            return True
        except Exception as e:  # noqa: BLE001 — 局部重排失败不阻断，保留空缺日
            _LOG.warning(f"行程空缺日局部重排失败，保留: {e}")
            return False
    return False


@node_log("node_review")
def node_review(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        raw = state.get("trip_plan") or {}
        if not raw:
            return {}
        plan = TripPlan(**raw)
        req = state.get("trip_request") or {}
        travel_days = _as_int(req.get("travel_days"))
        start_date = req.get("start_date") or ""
        notes: list = []

        # a. 天数多于请求天数 → 裁剪多余天
        if travel_days and len(plan.days) > travel_days:
            dropped = len(plan.days) - travel_days
            plan.days = plan.days[:travel_days]
            notes.append(f"行程天数多于请求天数，已裁剪多余 {dropped} 天")

        # b. 日期与 day_index 重排对齐出发日序列
        exp = _expected_dates(start_date, len(plan.days))
        realigned = False
        for i, d in enumerate(plan.days):
            d.day_index = i
            if exp[i] and d.date != exp[i]:
                d.date = exp[i]
                realigned = True
        if realigned:
            notes.append("已按出发日重排每日日期")

        # c. 跨天重复景点去重
        removed = _dedup_cross_day(plan)
        if removed:
            notes.append("已去除跨天重复景点: " + "、".join(sorted(set(removed))[:6]))

        # d. 单日游含超长城际交通 → 提示
        if travel_days == 1:
            for r in (plan.routes or []):
                if _hours(getattr(r, "duration_text", "") if not isinstance(r, dict) else r.get("duration_text", "")) >= 12:
                    notes.append("单日游含超长城际交通，建议确认到达后游玩时间是否充足")
                    break

        # e. 合并预算核查提示
        br = state.get("budget_review") or {}
        if br.get("note"):
            notes.append(br["note"])

        # f. 空缺日有界局部重排（至多 1 次）
        if _maybe_replan_empty_day(plan, state):
            notes.append("已对空缺日做 1 次局部重排")

        if notes:
            note = "【冲突核查】" + "；".join(notes) + "。"
            plan.overall_suggestions = (plan.overall_suggestions or "") + "\n" + note

        rendered = svc.render_plan_markdown(plan)
        return {"trip_plan": plan.model_dump(mode="json"), "rendered_text": rendered, "error": ""}
    except Exception as e:  # noqa: BLE001 — 审核失败绝不影响主链路
        _LOG.warning(f"行程冲突核查异常，放行: {e}")
        return {}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
