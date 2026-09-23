"""
文旅普通向量检索节点：对 tourism_chunks 执行混合检索。
支持无主体全库检索兜底。
"""
import sys

from app.rag.tourism_query.search_embedding_service import search_embedding
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：文旅向量内容检索（tourism_chunks）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result = search_embedding(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": result}
