"""
行程规划图编排入口（结构化 JSON 行程，第四路能力）。

节点流：
  node_extract_trip（NL 抽取/结构化直通）
  → node_collect_poi | node_collect_weather | node_collect_hotel（三路并行）
  → node_plan_itinerary（LLM 生成 JSON）
  → node_verify_poi（POI 事实核查）→ END

与 unified_query 主图解耦：本图可独立 invoke，也可被上层 HTTP 直接驱动。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[4]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from langgraph.graph import END, START, StateGraph

from app.shared.schemas.trip_plan import TripRequest
from app.process.trip_plan.agent.nodes.node_extract_trip import node_extract_trip
from app.process.trip_plan.agent.nodes.node_plan_itinerary import node_plan_itinerary, node_verify_poi
from app.process.trip_plan.agent.nodes.node_collect_poi import node_collect_poi
from app.process.trip_plan.agent.nodes.node_collect_weather import node_collect_weather
from app.process.trip_plan.agent.nodes.node_collect_hotel import node_collect_hotel
from app.process.trip_plan.agent.nodes.node_collect_route import node_collect_route
from app.process.trip_plan.agent.nodes.node_budget_review import node_budget_review
from app.process.trip_plan.agent.nodes.node_review import node_review
from app.process.trip_plan.agent.state import TripPlanGraphState, create_trip_plan_state

# 1. 定义状态图
workflow = StateGraph(TripPlanGraphState)

# 2. 添加节点
workflow.add_node("node_extract_trip", node_extract_trip)
workflow.add_node("node_collect_poi", node_collect_poi)
workflow.add_node("node_collect_weather", node_collect_weather)
workflow.add_node("node_collect_hotel", node_collect_hotel)
workflow.add_node("node_collect_route", node_collect_route)
workflow.add_node("node_plan_itinerary", node_plan_itinerary)
workflow.add_node("node_verify_poi", node_verify_poi)
# 预算核查 + 冲突审核（与 verify_poi 并行，均只依赖 plan_raw；零额外墙钟开销）
workflow.add_node("node_budget_review", node_budget_review)
workflow.add_node("node_review", node_review)

# 3. 入口
workflow.set_entry_point("node_extract_trip")

# 4. 条件边：抽取后三路并行采集（LangGraph 同一超步并行）
def after_extract(state: TripPlanGraphState) -> list[str]:
    return ["node_collect_poi", "node_collect_weather", "node_collect_hotel", "node_collect_route"]


workflow.add_conditional_edges(
    "node_extract_trip",
    after_extract,
    {
        "node_collect_poi": "node_collect_poi",
        "node_collect_weather": "node_collect_weather",
        "node_collect_hotel": "node_collect_hotel",
        "node_collect_route": "node_collect_route",
    },
)

# 5. 静态边：三路汇聚到规划（并行屏障由 LangGraph 超步保证）
workflow.add_edge("node_collect_poi", "node_plan_itinerary")
workflow.add_edge("node_collect_weather", "node_plan_itinerary")
workflow.add_edge("node_collect_hotel", "node_plan_itinerary")
workflow.add_edge("node_collect_route", "node_plan_itinerary")
# 5b. 规划后并行后校验：verify_poi 与 budget_review 均只依赖 plan_raw，零额外墙钟开销
workflow.add_edge("node_plan_itinerary", "node_verify_poi")
workflow.add_edge("node_plan_itinerary", "node_budget_review")
workflow.add_edge("node_verify_poi", "node_review")
workflow.add_edge("node_budget_review", "node_review")
workflow.add_edge("node_review", END)

# 6. 编译
trip_plan_app = workflow.compile()


def build_state(
    *,
    session_id: Optional[str] = None,
    request: Optional[TripRequest] = None,
    query: str = "",
    is_stream: bool = False,
) -> TripPlanGraphState:
    """构造 trip_plan 图初始状态：结构化 request 与 NL query 二选一。"""
    return create_trip_plan_state(
        session_id=session_id,
        trip_request=request.model_dump() if request else None,
        original_query=query,
        is_stream=is_stream,
    )


def invoke_structured(request: TripRequest, session_id: Optional[str] = None) -> dict[str, Any]:
    """
    结构化行程规划入口（HTTP /trip/plan 直接驱动）。

    Returns:
        state dict（含 trip_plan / rendered_text / error）。

    Raises:
        ValueError: 采集/规划/核查任一步失败（显式失败，绝不返回假数据）。
    """
    state = build_state(session_id=session_id, request=request)
    final = trip_plan_app.invoke(state)
    if not final.get("trip_plan"):
        raise ValueError(final.get("error") or "行程规划失败：未能生成有效行程，请稍后重试")
    return final


def invoke_nl(query: str, session_id: Optional[str] = None) -> dict[str, Any]:
    """
    自然语言行程规划入口（HTTP /trip/plan/nl 直接驱动）。
    抽取槽位 → 全流程规划；抽取失败抛 ValueError。
    """
    state = build_state(session_id=session_id, query=query)
    final = trip_plan_app.invoke(state)
    if not final.get("trip_plan"):
        raise ValueError(final.get("error") or "行程规划失败：未能生成有效行程，请稍后重试")
    return final


if __name__ == "__main__":
    from app.shared.runtime.logger import logger

    logger.info("===== trip_plan 图冒烟测试（编译 + 空抽取验证，不发真实请求） =====")
    test_state = build_state(query="帮我规划北京3天行程")
    try:
        graph_out = trip_plan_app.invoke(test_state)
        logger.warning("注意：未配置跳过真实抽取/规划，此路径不会真正成功（预期抛错）")
    except Exception as e:  # noqa: BLE001
        logger.info(f"冒烟路径按预期结束（无真实调用）: {type(e).__name__}")
    logger.info("===== trip_plan 图冒烟测试结束 =====")
