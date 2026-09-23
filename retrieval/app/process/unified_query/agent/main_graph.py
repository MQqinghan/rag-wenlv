"""
统一查询图编排入口：意图自动路由（文旅/闲聊）→ 检索链路 → RRF → 重排 → 答案输出。
架构说明：
- 入口 node_intent_route 按 domain 二分类路由：
    tourism   -> node_attraction_confirm（文旅主体确认）
    chitchat  -> node_web_search_mcp（仅联网搜索 + 闲聊回答）
- 文旅域走「普通向量 + HyDE + 联网搜索」三路并行检索，汇聚到共享的
  node_rrf -> node_rerank，再由 node_rerank 分发到答案输出节点。
- 闲聊链路绕过本地知识库：web_search -> chitchat_answer。
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[4]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from langgraph.graph import END, START, StateGraph

from app.process.unified_query.agent.nodes.node_intent_route import node_intent_route
from app.process.unified_query.agent.nodes.node_attraction_confirm import node_attraction_confirm
from app.process.unified_query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.unified_query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.unified_query.agent.nodes.node_keyword_search import node_keyword_search
from app.process.unified_query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.unified_query.agent.nodes.node_rrf import node_rrf
from app.process.unified_query.agent.nodes.node_rerank import node_rerank
from app.process.unified_query.agent.nodes.node_answer_output import node_answer_output
from app.process.unified_query.agent.nodes.node_chitchat_answer import node_chitchat_answer
from app.process.unified_query.agent.nodes.node_datetime_answer import node_datetime_answer
from app.process.unified_query.agent.nodes.node_clarify import node_clarify
from app.process.unified_query.agent.nodes.node_tool_weather import node_tool_weather
from app.process.unified_query.agent.nodes.node_tool_route import node_tool_route
from app.process.unified_query.agent.nodes.node_tool_rail import node_tool_rail
from app.process.unified_query.agent.nodes.node_tool_poi import node_tool_poi
from app.process.unified_query.agent.nodes.node_tool_stay_food import node_tool_stay_food
from app.shared.runtime.logger import logger
from app.process.unified_query.agent.nodes.node_itinerary_generate import node_itinerary_generate
from app.process.unified_query.agent.nodes.node_plan_slot_gate import node_plan_slot_gate
from app.rag.tourism_query.plan_slot_service import (
    is_plan_pending,
    plan_slot_clarify_enabled,
)

from app.process.unified_query.agent.state import UnifiedQueryGraphState


# 1. 定义状态图对象，使用统一查询状态（含 domain 字段）
workflow = StateGraph(UnifiedQueryGraphState)

# 2. 添加节点
workflow.add_node("node_intent_route", node_intent_route)
# 文旅域
workflow.add_node("node_attraction_confirm", node_attraction_confirm)
workflow.add_node("node_search_embedding_tourism", node_search_embedding)
workflow.add_node("node_search_embedding_hyde_tourism", node_search_embedding_hyde)
workflow.add_node("node_keyword_search_tourism", node_keyword_search)
workflow.add_node("node_answer_output_tourism", node_answer_output)
# 共享节点
workflow.add_node("node_web_search_mcp", node_web_search_mcp)
workflow.add_node("node_tool_weather", node_tool_weather)
workflow.add_node("node_tool_route", node_tool_route)
workflow.add_node("node_tool_rail", node_tool_rail)
# T15-D1：景点 POI 结构化信息 + 景点间距离（仅规划分支挂载）
workflow.add_node("node_tool_poi", node_tool_poi)
# 问题2：住宿 / 餐饮 POI 结构化信息（仅规划分支挂载）
workflow.add_node("node_tool_stay_food", node_tool_stay_food)
workflow.add_node("node_itinerary_generate", node_itinerary_generate)
# G 规划槽位闸门（第一步：缺 目的地/出发地/天数/偏好 时先反问一次；开关默认 OFF）
workflow.add_node("node_plan_slot_gate", node_plan_slot_gate)
workflow.add_node("node_rrf", node_rrf)
workflow.add_node("node_rerank", node_rerank)
workflow.add_node("node_chitchat_answer", node_chitchat_answer)
# 时间直答/澄清节点：纯时间询问与困惑输入不联网、不走检索，命中后直接出答案
workflow.add_node("node_datetime_answer", node_datetime_answer)
workflow.add_node("node_clarify", node_clarify)

# 3. 指定入口节点
workflow.set_entry_point("node_intent_route")


def _plan_slot_pending(state: UnifiedQueryGraphState) -> bool:
    """本会话是否正处于"规划信息收集中"（已反问过、等用户补充）。"""
    return plan_slot_clarify_enabled() and is_plan_pending(state.get("session_id") or "")


# 4. 意图路由：按 domain 三分类分发；chitchat 内再按标记细分（时间直答/澄清 不联网不走检索）
def route_by_domain(state: UnifiedQueryGraphState) -> str:
    domain = state.get("domain")
    # G 规划槽位收集进行中：用户本轮多半是在补信息（如"从成都出发，3 天"），
    # 这类句子不含"规划"字样、容易被判成闲聊或普通文旅，一旦走偏就丢掉收集上下文。
    # 只要会话还挂着 pending 标记（未超时），一律先送回文旅主体确认 → 槽位闸门。
    # 标记在闸门内"一用即清"，不会长期劫持路由。
    if _plan_slot_pending(state):
        logger.info("[plan_slot_gate] 会话处于规划信息收集中，强制走文旅分支")
        return "node_attraction_confirm"
    if domain == "chitchat":
        if state.get("is_datetime_query"):
            return "node_datetime_answer"
        if state.get("is_confusion"):
            return "node_clarify"
        if state.get("is_unresolved_deixis"):
            # 指代不明咨询句：直接确定性澄清反问，跳过联网搜索（联网结果对无指代问句无意义）
            return "node_chitchat_answer"
        return "node_web_search_mcp"
    return {"tourism": "node_attraction_confirm"}.get(domain, "node_attraction_confirm")


workflow.add_conditional_edges(
    "node_intent_route",
    route_by_domain,
    {
        "node_attraction_confirm": "node_attraction_confirm",
        "node_web_search_mcp": "node_web_search_mcp",
        "node_chitchat_answer": "node_chitchat_answer",
        "node_datetime_answer": "node_datetime_answer",
        "node_clarify": "node_clarify",
    },
)


# 6. 文旅主体确认后的分支：同样支持兜底直出 / 四路并行（文旅检索无主体时降级全库搜索）
# 规划类问题（is_plan=True）追加天气+路线工具并行分支，结果经 RRF 屏障后供答案生成引用
def _tourism_parallel_nodes(state: UnifiedQueryGraphState):
    """文旅并行检索节点列表（槽位闸门放行后与老路径共用同一份逻辑，避免两处漂移）。"""
    nodes = [
        "node_search_embedding_tourism",
        "node_search_embedding_hyde_tourism",
        "node_keyword_search_tourism",
    ]
    # D0（T15）：按 source_policy 门禁联网——仅"明确 kb"（本地库可答）时跳过 web；
    # web / kb_then_web 仍挂（时效补充/库外事实必须有网）；route_info 缺失时保守挂（不漏联网）。
    # 注：域外产品/时效类（如 neg-002 华为P60）走 chitchat 分支（route_by_domain），不经此处，不受影响。
    policy = (state.get("route_info") or {}).get("source_policy") or "kb_then_web"
    if policy != "kb":
        nodes.append("node_web_search_mcp")
    else:
        logger.info("[埋点][web_gate] policy=kb 跳过联网（文旅侧）")
    if state.get("is_plan"):
        nodes.extend([
            "node_tool_weather", "node_tool_route", "node_tool_rail", "node_tool_poi",
            # 问题2（2026-09-10）：住宿/餐饮 POI（高德）——旧链路缺此项，行程里
            # "住宿/餐饮预算"只能写"以现场为准"，主人反映像"没调用高德"。
            "node_tool_stay_food",
        ])
    return nodes


def _plan_slot_gate_needed(state: UnifiedQueryGraphState) -> bool:
    """是否该先过规划槽位闸门。

    规划类问题（is_plan=True）必过；此外，只要本会话还挂着"已反问"标记，
    也一并过闸门——这是"用户补信息那一轮"重新接回规划链路的关键。
    """
    if state.get("domain") != "tourism":
        return False
    if not plan_slot_clarify_enabled():
        return False
    if state.get("is_plan"):
        return True
    return is_plan_pending(state.get("session_id") or "")


def after_attraction_confirm(state: UnifiedQueryGraphState):
    if state.get("answer"):
        return "node_answer_output_tourism"
    # G（2026-09-14 主人拍板第一步）：规划类问题先过「槽位反问闸门」——
    # 缺 目的地/出发地/天数/偏好 时先一次问全，齐了才检索/出行程。
    if _plan_slot_gate_needed(state):
        return "node_plan_slot_gate"
    return _tourism_parallel_nodes(state)


def after_plan_slot_gate(state: UnifiedQueryGraphState):
    """闸门后继：已反问完（not ready）则本轮结束；否则按老逻辑并行检索。"""
    if state.get("plan_slot_ready"):
        return _tourism_parallel_nodes(state)
    return "END"


# 文旅分支共用的边映射（主体确认 / 槽位闸门两处出口保持一致，避免漏挂节点）
_TOURISM_EDGE_MAP = {
    "node_answer_output_tourism": "node_answer_output_tourism",
    "node_search_embedding_tourism": "node_search_embedding_tourism",
    "node_search_embedding_hyde_tourism": "node_search_embedding_hyde_tourism",
    "node_keyword_search_tourism": "node_keyword_search_tourism",
    "node_web_search_mcp": "node_web_search_mcp",
    "node_tool_weather": "node_tool_weather",
    "node_tool_route": "node_tool_route",
    "node_tool_rail": "node_tool_rail",
    "node_tool_poi": "node_tool_poi",
    "node_tool_stay_food": "node_tool_stay_food",
    "node_plan_slot_gate": "node_plan_slot_gate",
    "END": END,
}


workflow.add_conditional_edges(
    "node_attraction_confirm",
    after_attraction_confirm,
    _TOURISM_EDGE_MAP,
)

workflow.add_conditional_edges(
    "node_plan_slot_gate",
    after_plan_slot_gate,
    _TOURISM_EDGE_MAP,
)


# 7. 联网搜索后的分支：闲聊直接进入闲聊回答节点；文旅汇聚到 RRF
def after_web_search(state: UnifiedQueryGraphState) -> str:
    if state.get("domain") == "chitchat":
        return "node_chitchat_answer"
    return "node_rrf"


workflow.add_conditional_edges(
    "node_web_search_mcp",
    after_web_search,
    {
        "node_chitchat_answer": "node_chitchat_answer",
        "node_rrf": "node_rrf",
    },
)


# 8. 静态边：多路并行的汇聚关系（RRF 作为同步屏障）
workflow.add_edge("node_search_embedding_tourism", "node_rrf")
workflow.add_edge("node_search_embedding_hyde_tourism", "node_rrf")
workflow.add_edge("node_keyword_search_tourism", "node_rrf")
# 天气/路线工具与检索路并行，同样在 RRF 屏障汇聚（RRF 只产出 rrf_chunks，tool_* 原样透传）
workflow.add_edge("node_tool_weather", "node_rrf")
workflow.add_edge("node_tool_route", "node_rrf")
workflow.add_edge("node_tool_rail", "node_rrf")
# T15-D1：景点 POI 工具与其余工具同构，静态边汇入 RRF 屏障
workflow.add_edge("node_tool_poi", "node_rrf")
# 问题2：住宿/餐饮 POI 工具同上，静态边汇入 RRF 屏障
workflow.add_edge("node_tool_stay_food", "node_rrf")
workflow.add_edge("node_rrf", "node_rerank")


# 9. 重排后的分支：规划类文旅问题走行程拼装，其余按 domain 分发到对应域的答案输出节点
def after_rerank(state: UnifiedQueryGraphState) -> str:
    if state.get("domain") == "tourism" and state.get("is_plan"):
        return "node_itinerary_generate"
    return {
        "tourism": "node_answer_output_tourism",
        "chitchat": "node_chitchat_answer",
    }.get(state.get("domain"), "node_answer_output_tourism")


workflow.add_conditional_edges(
    "node_rerank",
    after_rerank,
    {
        "node_answer_output_tourism": "node_answer_output_tourism",
        "node_chitchat_answer": "node_chitchat_answer",
        "node_itinerary_generate": "node_itinerary_generate",
    },
)


# 10. 结束边
workflow.add_edge("node_answer_output_tourism", END)
workflow.add_edge("node_chitchat_answer", END)
workflow.add_edge("node_itinerary_generate", END)
workflow.add_edge("node_datetime_answer", END)
workflow.add_edge("node_clarify", END)

# 11. 编译统一查询图
unified_query_app = workflow.compile()
# 兼容旧命名：统一图即原 kb 查询图
kb_query_app = unified_query_app


if __name__ == "__main__":
    import uuid

    logger.info("===== 开始统一查询图编译/冒烟测试 =====")
    state = {
        "session_id": f"test_unified_{uuid.uuid4().hex[:8]}",
        "original_query": "你好",
        "is_stream": False,
        "domain": "",
    }
    try:
        for step in unified_query_app.stream(state, stream_mode="updates"):
            if step:
                logger.info(f"节点执行完成:{list(step.keys())}")
    except Exception as e:
        logger.exception("统一查询图冒烟测试失败")
    logger.info("===== 统一查询图冒烟测试结束 =====")
