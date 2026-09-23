"""
文旅联网检索节点：调用外部搜索引擎补充联网检索结果。
兜底策略：联网检索失败时降级为空结果，不阻断本地知识库主链路。
"""
import sys

from app.rag.common.web_search_service import search_web_documents
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充联网检索结果。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        pages = search_web_documents(state, count=10)
    except Exception as e:
        logger.warning(f"联网搜索失败,已降级为空结果,不影响本地知识库检索,错误信息:{str(e)}")
        pages = []
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"web_search_docs": pages}
