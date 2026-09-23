"""
意图路由节点：判断用户问题归属（文旅/闲聊），写回 state["domain"]。
作为统一查询图的入口节点，供 unified_query 图引用。

两种路由实现由特性开关 INTENT_ROUTE_LLM_FIRST 二选一（默认 false）：
- false：现有 v2 链路 classify_intent_dispatch（正则护栏 + LLM v2 主导 + 降级），行为不变；
- true ：B1 新版 route_intent（LLM 单次多路决策，附 model_tier 供 Harness 模型路由消费）。
"""
import sys

from app.rag.common.intent_route_service import DOMAIN_CHITCHAT, classify_intent_dispatch
from app.shared.config.common import env_bool
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


def _route_by_llm_first(state) -> str:
    """B1 新版（LLM 主导多路决策）路由，结果落 state 与 route_info。

    仅在特性开关 INTENT_ROUTE_LLM_FIRST=true 时被调用。route_intent 内部对 LLM
    失败/非法域已有 legacy 兜底，不抛异常。写入 route_info 以对齐 v2 口径，供 B2/B3
    与 Harness（model_tier）消费。
    """
    from app.rag.common.intent_route_llm import route_intent

    query = state.get("original_query") or state.get("rewritten_query") or ""
    led = route_intent(
        query,
        history_text=state.get("history_text", ""),
        session_id=state.get("session_id", ""),
    )
    domain = led.get("domain") or DOMAIN_CHITCHAT
    state["domain"] = domain
    prev = state.get("route_info") or {}
    state["route_info"] = {
        **prev,
        "rewritten_query": led.get("rewritten") or query,
        "domains": [domain],
        "source_policy": led.get("source_policy", "kb"),
        "tools": led.get("tools") or ["none"],
        "is_plan": bool(led.get("is_plan")),
        "model_tier": led.get("model_tier", "standard"),
        "route_mode": led.get("route_mode", "llm_led"),
    }
    if led.get("rewritten"):
        state["rewritten_query"] = led["rewritten"]
    if led.get("is_plan"):
        state["is_plan"] = True
    logger.info(f"B1 LLM 主导路由: [{query[:30]}] -> domain={domain} tier={led.get('model_tier')}")
    return domain


@node_log("node_intent_route")
def node_intent_route(state):
    """
    节点功能：意图自动路由（文旅 / 闲聊 二分类）。
    输入：state['original_query']
    输出：state['domain']，供统一图条件路由分发。
    闲聊分支额外预填 rewritten_query=original_query，供联网搜索节点消费。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    if env_bool("INTENT_ROUTE_LLM_FIRST", False):
        domain = _route_by_llm_first(state)
    else:
        domain = classify_intent_dispatch(state)
    state["domain"] = domain
    # 闲聊不经过主体改写节点，联网搜索需要 rewritten_query 非空，此处直接预填
    if domain == DOMAIN_CHITCHAT:
        state["rewritten_query"] = state.get("original_query") or state.get("rewritten_query") or ""
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
