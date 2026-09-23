"""
纯时间/日期确定性回答服务：node_datetime_answer 专用。

不用 LLM、不走检索、不改写：直接基于系统时钟 datetime.now() 计算答案，
从根上杜绝"今天是几月几号"被模型答成训练记忆里的"8月28日"这类幻觉日期。
支持：今天/明天/后天/昨天/前天 ± 日期 / 星期几 / 几点钟 的组合询问；
以及"国庆节是几号"等带节日的日期询问（公历节日直接算本年日期，
农历节日只答农历日期并提示以日历为准——不做逐年农历换算，绝不编造公历日期）。
"""
import re
from datetime import datetime, timedelta

from app.infra.persistence.history_repository import history_repository
from app.shared.runtime.logger import step_log
from app.shared.utils.sse_utils import SSEEvent, push_to_session
from app.shared.utils.task_utils import set_task_result

# 中文星期映射（与 state.current_date_text 同源，独立维护避免跨层依赖）
_WEEKDAY_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# 日期询问词
_DATE_ASK = re.compile(r"几月几号|几号|多少号|什么日期|日期")
# 星期询问词
_WEEKDAY_ASK = re.compile(r"星期几|周几|礼拜几")
# 时钟询问词
_CLOCK_ASK = re.compile(r"几点钟?|几点了|现在时间|什么时间|时间是多少")

# 公历固定节日 → (月, 日)：直接计算本年对应公历日期与星期
_FIXED_FESTIVALS: list[tuple[str, int, int]] = [
    ("元旦", 1, 1), ("情人节", 2, 14), ("妇女节", 3, 8), ("植树节", 3, 12),
    ("劳动节", 5, 1), ("青年节", 5, 4), ("儿童节", 6, 1), ("建党节", 7, 1),
    ("建军节", 8, 1), ("教师节", 9, 10), ("国庆节", 10, 1), ("圣诞节", 12, 25),
]
# 农历节日 → 农历日期文案：逐年公历换算需农历库，这里只答农历并提示以日历为准，杜绝编造
_LUNAR_FESTIVALS: dict[str, str] = {
    "春节": "正月初一", "元宵节": "正月十五", "元宵": "正月十五",
    "端午节": "五月初五", "七夕节": "七月初七", "七夕": "七月初七",
    "中元节": "七月十五", "中秋节": "八月十五", "重阳节": "九月初九",
    "腊八节": "腊月初八", "除夕": "除夕（农历腊月最后一天）",
}


def _resolve_festival_text(question: str, now: datetime) -> str:
    """带节日的日期问句：公历节日给确定日期；农历节日给农历日期提示。无命中返回空串。"""
    if "清明" in question:
        return f"清明节是二十四节气之一，一般落在公历4月4日至6日之间（{now:%Y}年具体日期以日历为准）。"
    # 固定公历节日（按名称长度倒序匹配，避免"元宵"吞"元宵节"等子串问题）
    for name, month, day in sorted(_FIXED_FESTIVALS, key=lambda x: len(x[0]), reverse=True):
        if name in question:
            festival_date = now.replace(month=month, day=day)
            return (
                f"{now:%Y}年{name}是{festival_date.month}月{festival_date.day}日，"
                f"{_WEEKDAY_CN[festival_date.weekday()]}。"
            )
    # 农历节日
    for name, lunar_text in sorted(_LUNAR_FESTIVALS.items(), key=lambda x: len(x[0]), reverse=True):
        if name in question:
            return f"{name}是农历{lunar_text}，{now:%Y}年对应的公历日期请以日历为准。"
    return ""


@step_log("resolve_datetime_text")
def resolve_datetime_text(question: str) -> str:
    """解析纯时间/日期询问文本 → 确定性答案（基于系统时钟，不经 LLM）。"""
    now = datetime.now()

    # 带节日的日期问句优先走节日映射（不需要相对日期偏移）
    festival_answer = _resolve_festival_text(question, now)
    if festival_answer:
        return festival_answer

    # 目标日偏移（"明天是几号"问的是明天；"大后天"先于"后天"判断避免子串误吞）
    if "大大后天" in question:
        offset = 4
    elif "大后天" in question:
        offset = 3
    elif "后天" in question:
        offset = 2
    elif "明天" in question or "明日" in question:
        offset = 1
    elif "昨天" in question or "昨日" in question:
        offset = -1
    elif "前天" in question:
        offset = -2
    elif "大前天" in question:
        offset = -3
    else:
        offset = 0
    target = now + timedelta(days=offset)
    anchor = {4: "大大后天", 3: "大后天", 2: "后天", 1: "明天", 0: "今天", -1: "昨天", -2: "前天", -3: "大前天"}.get(offset, "今天")

    date_asked = bool(_DATE_ASK.search(question))
    weekday_asked = bool(_WEEKDAY_ASK.search(question))
    clock_asked = bool(_CLOCK_ASK.search(question))

    parts: list[str] = []
    if date_asked or weekday_asked:
        date_txt = f"{anchor}是{target.year}年{target.month}月{target.day}日" if date_asked else ""
        weekday_txt = _WEEKDAY_CN[target.weekday()] if weekday_asked else ""
        if date_txt and weekday_txt:
            parts.append(f"{date_txt}，{weekday_txt}。")
        elif date_txt:
            parts.append(f"{date_txt}。")
        else:
            parts.append(f"{anchor}是{weekday_txt}。")
    if clock_asked:
        parts.append(f"现在是{now:%H}:{now:%M}。")

    if not parts:  # 理论不可达（意图路由层已过滤），兜底给完整信息
        parts.append(f"今天是{now.year}年{now.month}月{now.day}日，{_WEEKDAY_CN[now.weekday()]}。")
    return "".join(parts)


@step_log("generate_datetime_answer")
def generate_datetime_answer(state: dict) -> dict:
    """
    时间直答主入口：确定性计算答案 → SSE 推送（流式）/task_result → 历史落库。

    Returns:
        dict: 更新 answer/image_urls 后的最新状态。
    """
    question = state.get("original_query") or state.get("rewritten_query") or ""
    if not question:
        raise ValueError("生成时间直答需要 original_query/rewritten_query")

    answer = resolve_datetime_text(question)
    session_id = state["session_id"]
    if state.get("is_stream", False):
        # 确定性直答没有逐字流，整段一次性推送（前端按 delta 增量渲染，等价于"打字机"效果）
        push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
    set_task_result(session_id, "answer", answer)
    state["answer"] = answer
    state["image_urls"] = []
    # 与闲聊回答一致：只落助手消息（role=user 由常规主体确认节点负责，本链路绕过它）
    history_repository.save_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query=question,
        item_names=[],
        image_urls=[],
        domain=state.get("domain", "chitchat"),
    )
    return state
