"""
文旅 RRF 融合节点：将本地两路检索结果（普通向量 + HyDE）倒数排名融合。
算法与公共层 common/rrf_service 共用。
"""
import sys

from app.rag.common.rrf_service import fuse_by_rrf
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rrf")
def node_rrf(state):
    """
    节点功能：RRF 倒数排名融合（文旅本地双路）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    rrf_chunks = fuse_by_rrf(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"rrf_chunks": rrf_chunks}
