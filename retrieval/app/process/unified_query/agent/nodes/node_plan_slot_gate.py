"""规划槽位反问闸门节点（第一步：对话内反问，不换引擎）。

职责（只做闸门，不做规划）：
    行程规划链路在进入检索 / 行程生成之前先过这道闸门——
      ① 本会话已反问过（pending 标记在）→ 清标记直接放行，绝不连环追问；
      ② 否则判断五项（目的地 / 出发地 / 天数 / 偏好 / 预算）是否齐备：
           - 齐备 或 未达反问阈值 → 放行；
           - 缺项 → 写反问答案 + 落历史 + 写 pending 标记，本轮到此结束。

放行方式：
    写 `state["plan_slot_ready"] = True`，由 main_graph 的条件边
    `after_plan_slot_gate` 决定后继并行节点列表（复用 `_tourism_parallel_nodes`）。

开关：
    `PLAN_SLOT_CLARIFY_ENABLED`（默认 OFF）。OFF 时本节点恒放行，
    行为与接线前完全一致。

评测提示：
    本节点会让"缺项"的规划用例由"直接出行程"变成"先反问"，属主人拍板的
    预期行为变化；如需对照既有基线，把 `PLAN_SLOT_CLARIFY_MIN_MISSING` 设为 5
    （仅五项全缺才反问）即可最大收敛影响面。
"""
import sys

from app.infra.persistence import history_repository
from app.rag.tourism_query.attraction_confirm_service import load_history
from app.rag.tourism_query.plan_slot_service import (
    build_plan_clarify_answer,
    clear_plan_pending,
    extract_plan_slots,
    is_plan_pending,
    mark_plan_pending,
    missing_plan_slots,
    plan_clarify_required,
    plan_slot_clarify_enabled,
)
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.sse_utils import SSEEvent, push_to_session
from app.shared.utils.task_utils import add_done_task, add_running_task, set_task_result


@node_log("node_plan_slot_gate")
def node_plan_slot_gate(state):
    """规划槽位闸门：缺信息先反问一次；否则放行进入正常检索/行程生成。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 开关 OFF：恒放行（与接线前行为一致）
    if not plan_slot_clarify_enabled():
        state["plan_slot_ready"] = True
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return state

    session_id = state["session_id"]
    query = state.get("original_query") or state.get("rewritten_query") or ""

    # ① 已反问过 → 本轮无论补没补齐都放行（主人拍板：同一会话最多反问一次）
    if is_plan_pending(session_id):
        clear_plan_pending(session_id)
        logger.info(f"规划槽位闸门：本会话已反问过，直接放行不再追问，session_id={session_id}")
        state["plan_slot_ready"] = True
        add_done_task(session_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return state

    # ② 判断缺项（state["history"] 已由 confirm_attractions 回填，取不到再兜底读库）
    history_text = state.get("history") or load_history(session_id)
    flags = extract_plan_slots(query, history_text)
    missing = missing_plan_slots(flags)
    state["plan_slots"] = flags

    if not plan_clarify_required(flags):
        logger.info(f"规划槽位闸门：放行（缺项={missing or '无'}，未达反问阈值）")
        state["plan_slot_ready"] = True
        add_done_task(session_id, sys._getframe().f_code.co_name, state.get("is_stream"))
        return state

    # ③ 缺项 → 一次性问全，并记住"已问过"
    answer = build_plan_clarify_answer(missing)
    logger.info(f"规划槽位闸门：缺项={missing}，反问问清后再规划，session_id={session_id}")

    set_task_result(session_id, "answer", answer)
    if state.get("is_stream", False):
        push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
    history_repository.save_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query=query,
        item_names=state.get("item_names", []),
        image_urls=[],
        domain=state.get("domain", "tourism"),
        is_plan=True,
    )
    mark_plan_pending(session_id)

    state["plan_slot_ready"] = False
    state["answer"] = answer
    state["image_urls"] = []
    add_done_task(session_id, sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
