"""
文旅主体确认节点：改写问题 + 抽取主体（景点名/文化主题）+ tourism_entity 库确认。
无确认主体时降级为全库检索（不生成澄清话术）。
"""
import sys

from app.rag.tourism_query.attraction_confirm_service import confirm_attractions
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_attraction_confirm")
def node_attraction_confirm(state):
    """
    节点功能：确认用户问题中的核心主体（景点名/文化主题）。
    输入：state['original_query']
    输出：更新 state['item_names'] 与 state['rewritten_query']。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = confirm_attractions(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
