"""
行程规划图：node_collect_route。

若用户提供出发地（origin），调用高德 plan_route 生成去程（origin→city）与返程（city→origin）
路线参考文本；origin 为空时置空（向后兼容，仅规划目的地市内行程）。
软失败：路线获取失败不影响主链路（由规划器按「无往返参考」处理）。
"""
import sys

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_collect_route")
def node_collect_route(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        req = svc._coerce_request(state.get("trip_request") or {})
        routes = svc.collect_routes(req) if req else None
        routes_text = svc.collect_routes_text(routes)
        return {"routes_text": routes_text, "routes": routes or []}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
