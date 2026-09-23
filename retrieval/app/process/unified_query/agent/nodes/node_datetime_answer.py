"""
时间/日期确定性直答节点：处理被路由为纯时间询问（is_datetime_query=True）的输入。
与闲聊不同：不联网、不调 LLM，直接基于系统时钟给出确定性答案（杜绝幻觉日期）。
"""
import sys

from app.rag.common.datetime_answer_service import generate_datetime_answer
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_datetime_answer")
def node_datetime_answer(state):
    """
    节点功能：确定性回答"今天是几号/现在几点/明天星期几"等纯时间询问。
    输入：state['original_query']
    输出：更新 state['answer']（系统时钟计算，不经 LLM）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = generate_datetime_answer(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
