"""
文旅行程拼装节点（规划类问题专用答案生成）：综合 KB 检索 + 天气/路线简报生成结构化行程。
降级策略：生成失败时写入兜底答复，保证 SSE 会话正常收尾。

第二步（2026-09-14 主人拍板方案 A）：开关 `PLAN_ENGINE_TRIP_PLAN` ON 时，
优先委托独立的 trip_plan 智能体（结构化 TripPlan + 高德 POI 事实核查 + 预算）
产出并渲染回对话；桥接失败一律回落原 KB 行程引擎，保证"换了引擎也答得出"。
"""
import sys

from app.process.unified_query.agent.trip_plan_engine import run_trip_plan_engine
from app.rag.tourism_query.itinerary_service import generate_itinerary
from app.rag.tourism_query.trip_plan_bridge import trip_plan_engine_enabled
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task, set_task_result


@node_log("node_itinerary_generate")
def node_itinerary_generate(state):
    """
    节点功能：rerank 后按 is_plan 分流进入，产出结构化行程作为最终答案。

    引擎选择（开关 `PLAN_ENGINE_TRIP_PLAN`）：
        ON  → trip_plan 智能体（结构化行程；失败回落原引擎）
        OFF（默认）→ 原 KB 行程引擎，行为与接线前完全一致
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        update = None
        # 第二步：trip_plan 智能体优先（协调器内部已 try/except，失败返回 None）
        if trip_plan_engine_enabled():
            update = run_trip_plan_engine(state)
            if update is None:
                logger.info("对话内 trip_plan 智能体未产出结果，回落原 KB 行程引擎")
        if update is None:
            update = generate_itinerary(state)
    except Exception as e:
        logger.exception(f"行程拼装失败,降级为兜底答复,session_id={state.get('session_id')},错误信息:{str(e)}")
        fallback = "行程规划生成失败，请稍后重试。您也可以先单独询问目的地的天气、路线或游玩攻略。"
        update = {"answer": fallback, "image_urls": []}
        set_task_result(state["session_id"], "answer", fallback)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return update
