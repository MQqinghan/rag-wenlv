"""
澄清回答节点：处理被路由为困惑/澄清类输入（is_confusion=True）的消息。
用户对上一轮回答表示没看懂（"？""没明白""不对吧"）时，结合上一条助手回答与当前日期重新解释。
不联网、不走本地知识库检索，避免"？"被当作新问题把历史行程重生成一遍。
"""
import sys

from app.rag.common.clarify_answer_service import generate_clarify_answer
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_clarify")
def node_clarify(state):
    """
    节点功能：澄清上一条回答（含时间/日期错误纠正）。
    输入：state['original_query']
    输出：更新 state['answer']。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = generate_clarify_answer(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
