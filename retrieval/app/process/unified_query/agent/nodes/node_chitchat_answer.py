"""
闲聊回答节点：处理被路由为 chitchat 的日常对话。
只消费 node_web_search_mcp 的联网结果生成回答，不访问本地知识库。
"""
import sys

from app.rag.common.chitchat_answer_service import generate_chitchat_answer
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_chitchat_answer")
def node_chitchat_answer(state):
    """
    节点功能：闲聊回答生成（web 联网结果 + LLM）。
    输入：state['original_query']、state['web_search_docs']
    输出：更新 state['answer']。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = generate_chitchat_answer(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
