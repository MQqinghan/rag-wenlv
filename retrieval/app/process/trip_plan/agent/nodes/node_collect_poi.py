"""
行程规划图：node_collect_poi。

按偏好关键词高德搜索景点 POI；检索不到则显式失败（不生成无依据行程）。
"""
import sys

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_collect_poi")
def node_collect_poi(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        req = svc._coerce_request(state.get("trip_request") or {})
        pois = svc.collect_pois(req) if req else []
        if not pois:
            msg = "高德地图未能检索到该城市的景点数据，请确认城市名是否正确后重试。"
            state["error"] = msg
            raise ValueError(msg)
        return {"pois": pois, "pois_text": svc.collect_pois_text(pois), "error": ""}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
