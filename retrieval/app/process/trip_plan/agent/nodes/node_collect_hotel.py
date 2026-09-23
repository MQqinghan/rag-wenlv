"""
行程规划图：node_collect_hotel。

按住宿偏好高德搜索酒店；软失败：无酒店数据时置空（规划器 hotel 填 null）。
"""
import sys

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_collect_hotel")
def node_collect_hotel(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        req = svc._coerce_request(state.get("trip_request") or {})
        hotels = svc.collect_hotels(req) if req else []
        return {"hotels": hotels, "hotels_text": svc.collect_hotels_text(hotels)}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
