"""
文旅答案输出节点：最终答案生成、图片提取与历史落库。
使用 tourism/answer_out 模板。
"""
import sys

from app.rag.tourism_query.answer_output_service import generate_answer
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：文旅答案生成与输出。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = generate_answer(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
