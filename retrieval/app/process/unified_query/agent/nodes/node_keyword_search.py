"""
文旅关键词检索节点：基于本地 BM25 索引对 tourism_chunks 做字面匹配召回。
与向量/多改写检索互补，结果进入 RRF 融合。
"""
import sys

from app.rag.tourism_query.search_keyword_service import search_keyword
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_keyword_search")
def node_keyword_search(state):
    """
    节点功能：文旅 BM25 关键词检索（tourism_chunks）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    result = search_keyword(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"keyword_chunks": result}
