"""
统一查询图（文旅/闲聊/行程规划）的状态定义。

字段为文旅/闲聊/行程规划共享，额外携带 domain 路由字段。
"""
import copy

from typing import TypedDict

from app.shared.runtime.date_utils import current_date_text


class UnifiedQueryGraphState(TypedDict):
    """统一查询状态 = 文旅/闲聊/行程规划共享字段 + domain（tourism/chitchat）+ B1 路由信息。

    route_info：B1 v2 意图路由单次输出（rewritten_query/domains/source_policy/tools/is_plan/
    need_clarify），供 B2 域判定与 B3 答案衔接消费；v1 路由不写入。
    """
    session_id: str
    is_stream: bool
    original_query: str
    rewritten_query: str
    current_date: str
    is_plan: bool
    is_datetime_query: bool
    is_confusion: bool
    is_unresolved_deixis: bool
    is_off_domain_product: bool
    history: list
    embedding_chunks: list
    hyde_embedding_chunks: list
    keyword_chunks: list
    web_search_docs: list
    rrf_chunks: list
    reranked_docs: list
    tool_weather: dict
    tool_route: dict
    tool_rail: dict
    tool_poi: dict
    tool_stay_food: dict
    tool_transport: dict
    tool_budget: dict
    itinerary_map: dict
    answer: str
    image_urls: list
    map_data: dict
    domain: str
    route_info: dict
    user_id: str
    plan_slot_ready: bool
    plan_slots: dict


query_default_state: UnifiedQueryGraphState = {
    "session_id": "",
    "is_stream": False,
    "original_query": "",
    "rewritten_query": "",
    "current_date": "",
    "is_plan": False,
    "is_datetime_query": False,
    "is_confusion": False,
    "is_unresolved_deixis": False,
    "is_off_domain_product": False,
    "history": [],
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "keyword_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "tool_weather": {},
    "tool_route": {},
    "tool_rail": {},
    "tool_poi": {},
    "tool_stay_food": {},
    "tool_transport": {},
    "tool_budget": {},
    "itinerary_map": {},
    "answer": "",
    "image_urls": [],
    "map_data": {},
    "domain": "tourism",
    "route_info": {},
    "user_id": "",
    "plan_slot_ready": False,
    "plan_slots": {},
}


def create_unified_default_state(**overrides) -> UnifiedQueryGraphState:
    """创建统一查询图默认状态，支持覆盖。domain 默认 tourism（HTTP 入口会按意图覆盖），
    current_date 同样在创建时注入。"""
    state = copy.deepcopy(query_default_state)
    state["domain"] = "tourism"
    state["route_info"] = {}
    state["user_id"] = ""
    state.update(overrides)
    if not state.get("current_date"):
        state["current_date"] = current_date_text()
    return state
