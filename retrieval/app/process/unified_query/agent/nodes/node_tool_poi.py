"""
文旅景点 POI 工具节点（规划类问题专用，T15-D1）：调用高德 POI 接口取目的地景点
结构化信息（地址 / 评分 / 人均 / 开放时间）与景点间距离。

降级策略：未配置 Key / 无法解析目的地 / 查询失败时写入空简报（ok=False），不阻断主检索链路。
"""
import sys

from app.rag.tourism_query.poi_tool_service import get_poi_brief
from app.shared.runtime.logger import logger, node_log
from app.shared.runtime.action_guard import require_action
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_tool_poi")
def node_tool_poi(state):
    """
    节点功能：规划类问题并行分支之一，产出 state["tool_poi"] 景点 POI 简报。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    try:
        require_action("poi_query")  # I：动作授权闸门（开关默认 OFF）
        brief = get_poi_brief(state)
    except Exception as e:
        logger.warning(f"景点POI工具节点异常,降级为空简报,不影响主链路,错误信息:{str(e)}")
        brief = {
            "ok": False, "destination": "", "city": "",
            "fetched_at": "", "names": [], "pois": [], "distances": [], "text": "",
        }
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"tool_poi": brief}
