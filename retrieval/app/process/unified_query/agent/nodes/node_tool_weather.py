"""
文旅天气工具节点（规划类问题专用）：调用和风天气 QWeather 查询目的地逐日天气。
降级策略：查询失败时写入空简报（ok=False），不阻断主检索链路。
"""
import sys

from app.rag.tourism_query.weather_tool_service import get_weather_brief
from app.shared.runtime.logger import logger, node_log
from app.shared.runtime.action_guard import require_action
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_tool_weather")
def node_tool_weather(state):
    """
    节点功能：规划类问题并行分支之一，产出 state["tool_weather"] 天气简报。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        require_action("weather_query")  # I：动作授权闸门（开关默认 OFF）
        brief = get_weather_brief(state)
    except Exception as e:
        logger.warning(f"天气工具节点异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        brief = {"ok": False, "destination": "", "text": ""}
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"tool_weather": brief}
