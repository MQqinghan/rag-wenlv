"""
文旅路线规划工具节点（规划类问题专用）：调用高德驾车路径规划。
降级策略：未配置 Key / 规划失败时写入空简报（ok=False），不阻断主检索链路。
"""
import sys

from app.rag.tourism_query.route_tool_service import plan_route
from app.shared.runtime.logger import logger, node_log
from app.shared.runtime.action_guard import require_action
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_tool_route")
def node_tool_route(state):
    """
    节点功能：规划类问题并行分支之一，产出 state["tool_route"] 驾车路线简报。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        require_action("route_query")  # I：动作授权闸门（开关默认 OFF）
        brief = plan_route(state)
    except Exception as e:
        logger.warning(f"路线工具节点异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        brief = {"ok": False, "origin": "", "destination": "", "text": ""}
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"tool_route": brief}
