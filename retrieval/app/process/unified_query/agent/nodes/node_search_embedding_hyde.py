"""
文旅 HyDE 检索节点：生成假设性答案后对 tourism_chunks 执行增强检索。
"""
import sys

from app.rag.tourism_query.search_embedding_hyde_service import search_embedding_hyde
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：文旅 HyDE 增强检索（tourism_chunks）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result = search_embedding_hyde(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": result}
