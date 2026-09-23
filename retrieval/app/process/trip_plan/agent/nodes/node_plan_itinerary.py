"""
行程规划图：node_plan_itinerary / node_verify_poi。

node_plan_itinerary：LLM 生成结构化行程 JSON（对应旧工程「行程规划 Agent」），
  解析失败显式抛错（不静默生成假行程），由 HTTP 层转 success=false。
node_verify_poi：POI 事实核查（强约束），把核查结果写回 state["trip_plan"]（最终 HTTP data）。
"""
import sys

from app.shared.schemas.trip_plan import TripPlan, RouteInfo
from app.process.trip_plan.agent.state import TripPlanGraphState
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_plan_itinerary")
def node_plan_itinerary(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        req = svc._coerce_request(state.get("trip_request") or {})
        routes = state.get("routes") or []
        transport_cost = sum(int(r.get("cost", 0) or 0) for r in routes)
        plan = svc.run_planner_parallel(
            req,
            state.get("pois_text", ""),
            state.get("weather_text", ""),
            state.get("hotels_text", ""),
            state.get("routes_text", ""),
            pois=state.get("pois") or [],
            weather_rows=state.get("weather_rows") or [],
            extra_transport=transport_cost,
        )
        svc.validate_plan(plan, req)
        return {"plan_raw": plan.model_dump(mode="json")}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))


@node_log("node_verify_poi")
def node_verify_poi(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        raw = state.get("plan_raw") or {}
        plan = TripPlan(**raw)
        city = state.get("trip_request", {}).get("city") or plan.city
        plan = svc.verify_plan(plan, city)
        routes = state.get("routes") or []
        if routes:
            try:
                plan.routes = [RouteInfo(**r) if isinstance(r, dict) else r for r in routes]
            except Exception:
                pass
        plan_dict = plan.model_dump(mode="json")
        return {
            "trip_plan": plan_dict,
            "rendered_text": svc.render_plan_markdown(plan),
            "error": "",
        }
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
