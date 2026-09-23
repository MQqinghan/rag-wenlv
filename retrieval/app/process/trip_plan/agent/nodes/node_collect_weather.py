"""
行程规划图：node_collect_weather。

查询行程区间天气；软失败：无天气数据时置空简报（不阻断，由规划器填“-”）。
"""
import sys

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_collect_weather")
def node_collect_weather(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        req = svc._coerce_request(state.get("trip_request") or {})
        rows = svc.collect_weather(req) if req else []
        return {"weather_rows": rows, "weather_text": svc.collect_weather_text(rows)}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
