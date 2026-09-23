"""
文旅全局重排节点：本地 RRF 融合结果 + 外网网页结果统一语义精排。
算法与公共层 common/rerank_service 共用。
"""
import sys

from app.rag.common.rerank_service import rerank_documents
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：文旅全局统一精排。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    reranked_docs = rerank_documents(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs": reranked_docs}
