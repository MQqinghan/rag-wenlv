"""历史消息拼接为 Prompt 纯文本的通用工具。

本工具与具体业务域无关，仅保留与域无关的通用历史文本构造逻辑，供文旅答案生成与行程拼装
复用（指代消解场景可放宽助手消息截断上限）。
"""
from app.shared.runtime.logger import step_log

# 单条历史消息参与 Prompt 的最大字符数（超长截断，防历史主体污染改写）
HISTORY_MSG_MAX_CHARS = 160
# 指代消解专用放宽值："上面提到的那个地方"这类问题要靠上一轮助手长回答补全地名，
# 160 字会把中后段的地名截掉，导致改写层补不出主体、后续检索全塌（多轮规划场景实测）。
HISTORY_DEIXIS_MAX_CHARS = 400


@step_log("build_history_text")
def build_history_text(history_messages, assistant_max_chars=None):
    """将历史消息拼接为适合 Prompt 使用的纯文本。

    Args:
        history_messages: 历史消息列表，包含角色、文本、改写问题和关联主体等字段。
        assistant_max_chars: 助手消息单独的截断上限；None 时与用户消息同用
            HISTORY_MSG_MAX_CHARS。仅在当前问题含指代需要靠上一轮长回答消解时传大值。

    Returns:
        str: 拼接后的历史上下文文本。
    """
    assistant_limit = assistant_max_chars or HISTORY_MSG_MAX_CHARS
    lines = []
    for msg in history_messages:
        content = msg.get("rewritten_query") if msg.get("role") == "user" else msg.get("text")
        content = str(content or "").strip()
        limit = assistant_limit if msg.get("role") != "user" else HISTORY_MSG_MAX_CHARS
        if len(content) > limit:
            content = content[:limit] + "…(已截断)"
        # Mongo 历史字段仍叫 item_names（兼容存量数据），读作关联主体展示
        item_names = "、".join(msg.get("item_names", []) or [])
        lines.append(f"角色:{msg.get('role', '')},内容:{content},关联主体: {item_names}")
    return "\n".join(lines)
