# -*- coding: utf-8 -*-
"""对话内行程引擎协调器（process 层）。

为何单独一层：
    依赖方向 lint 强制 `shared <- infra <- rag <- process <- api`。
    rag 层的 `trip_plan_bridge` 只做纯逻辑（抽值 / 构造请求 / 渲染），
    不得 import process；而"驱动 trip_plan 子图"属 process 层职责，
    故把两者串起来这一步收在本模块，保持分层合规、节点保持薄壳。

链路：
    node_itinerary_generate（开关 ON）
      → run_trip_plan_engine(state)
        → bridge.prepare_trip_plan_request(state)   历史 + 抽值 → TripRequest
        → trip_plan.agent.main_graph.invoke_structured(...)
        → bridge.finalize_trip_plan_answer(state, final)   渲染回对话

失败语义：
    任一步失败返回 None，由调用方回落原 KB 行程引擎（fail-open）。
"""
from __future__ import annotations

from typing import Optional

from app.process.trip_plan.agent.main_graph import invoke_structured
from app.rag.tourism_query import trip_plan_bridge as bridge
from app.shared.runtime.logger import logger, step_log


@step_log("run_trip_plan_engine")
def run_trip_plan_engine(state: dict) -> Optional[dict]:
    """对话内行程规划：桥接槽位 → 驱动 trip_plan 智能体 → 渲染回对话。

    Returns:
        dict: `{"answer","image_urls","itinerary_map","map_data"}`；失败返回 None。
    """
    try:
        request = bridge.prepare_trip_plan_request(state)
        if request is None:
            logger.info("对话内 trip_plan 引擎：未凑齐目的地，回落原 KB 行程引擎")
            return None

        final = invoke_structured(request, session_id=state.get("session_id") or "")
        return bridge.finalize_trip_plan_answer(state, final)
    except Exception as e:  # noqa: BLE001 — 失败必须回落原引擎，绝不把对话链路卡死
        logger.exception(f"对话内 trip_plan 智能体执行失败，回落原行程引擎：{e}")
        return None
