# -*- coding: utf-8 -*-
"""行程规划图：node_budget_review。

预算核查（确定性、无 LLM、毫秒级）：把规划器算出的总预算与用户约束预算比对，
产出超支标记与降级提示，写入 state["budget_review"]，供 node_review 汇总进建议。

设计：
    - 仅做算术，不调用模型，零额外墙钟开销；
    - 用户未给预算约束（limit 为 None/0）时直接放行，不影响链路；
    - 任何异常一律 fail-open（返回空 dict），绝不把规划链路拦死。
"""
import sys

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@node_log("node_budget_review")
def node_budget_review(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        req = state.get("trip_request") or {}
        limit = _as_int(req.get("budget"))
        if limit <= 0:
            return {"budget_review": {"enabled": False}}

        plan_raw = state.get("plan_raw") or {}
        budget = plan_raw.get("budget") or {}
        total = _as_int(budget.get("total"))

        over = total - limit
        if over <= 0:
            return {
                "budget_review": {
                    "enabled": True,
                    "limit": limit,
                    "total": total,
                    "over": 0,
                    "over_percent": 0,
                    "note": "",
                }
            }

        over_percent = round(over / limit * 100)
        note = (
            f"当前预估约 {total} 元，已超预算（目标 {limit} 元）约 {over} 元（{over_percent}%）。"
            f"建议：减少高价门票景点、选更经济住宿，或交通以 12306 二等座/公共接驳为准。"
        )
        return {
            "budget_review": {
                "enabled": True,
                "limit": limit,
                "total": total,
                "over": over,
                "over_percent": over_percent,
                "note": note,
            }
        }
    except Exception as e:  # noqa: BLE001 — 预算核查失败绝不影响主链路
        import logging
        logging.getLogger("trip_plan").warning(f"预算核查异常，放行: {e}")
        return {"budget_review": {}}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
