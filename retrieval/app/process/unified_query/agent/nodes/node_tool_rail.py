"""
文旅游铁路工具节点（规划类问题专用）：调用自部署 12306-MCP 查询跨城车次与票价。
降级策略：未配置/无法解析出发地目的地/查询失败时写入空简报（ok=False），不阻断主检索链路。
"""
import sys

from app.rag.tourism_query.rail_tool_service import get_rail_brief
from app.shared.runtime.logger import logger, node_log
from app.shared.runtime.action_guard import require_action
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_tool_rail")
def node_tool_rail(state):
    """
    节点功能：规划类问题并行分支之一，产出 state["tool_rail"] 铁路车次/票价简报。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        require_action("rail_query")  # I：动作授权闸门（开关默认 OFF）
        brief = get_rail_brief(state)
    except Exception as e:
        logger.warning(f"铁路工具节点异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        brief = {"ok": False, "origin": "", "destination": "", "date": "", "codes": [], "text": ""}
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"tool_rail": brief}
