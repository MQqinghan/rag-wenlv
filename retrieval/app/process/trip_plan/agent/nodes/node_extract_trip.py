"""
行程规划图：node_extract_trip。

结构化 TripRequest 就绪则原样透传；否则从 NL 问句抽取槽位。
抽取失败：写入 error 并抛 ValueError（HTTP 层转 success=false，不返回假数据）。
"""
import sys

from app.process.trip_plan.agent.state import TripPlanGraphState
from app.rag.trip_plan import trip_planner_service as svc
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_extract_trip")
def node_extract_trip(state: TripPlanGraphState) -> dict:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        trip_request = state.get("trip_request") or {}
        if not trip_request:
            req_dict = svc.extract_trip_request(state.get("original_query", ""))
            if not req_dict:
                msg = "未能从您的描述中解析出目的地城市与日期，请提供更完整的信息（如：帮我规划北京3天行程，9月20日出发）。"
                state["error"] = msg
                raise ValueError(msg)
            trip_request = req_dict
            state["trip_request"] = trip_request
        return {"trip_request": dict(trip_request), "error": ""}
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
