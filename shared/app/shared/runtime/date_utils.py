"""公共日期工具：当前日期文本与未来14天映射，供 Prompt 换算相对时间。"""
from datetime import datetime, timedelta

# 中文星期映射（datetime.strftime 的 %A 是英文，这里统一输出"周X"）
_WEEKDAY_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def current_date_text() -> str:
    """
    返回当前日期文本 + 未来14天日期映射表，如：
    "2026-09-11 周五；未来14天：明天=09-12，后天=09-13，下周一=09-14，下周二=09-15..."
    供 Prompt 直接查表换算"明天/下周一/下周六"等相对时间，避免 LLM 自行做日历算术出错。

    R15（2026-09-11 主人实测 E3「下周从深圳出发」被排到本周六 09-12）：
    - 表长由 7 天扩到 14 天，覆盖完整下一自然周，保证「下周X」「下周末」都有表可查；
    - 标签按下一次"周一"为界做自然周判定（本周=周X、下周=下周X、再下周=下下周X），
      替换旧逻辑"未来7天内首个该星期即为周X/下周X"——该口径会把"下周"与本周剩余
      日期混淆（周五问"下周"曾被映射到本周六）。
    """
    now = datetime.now()
    base = f"{now:%Y-%m-%d} {_WEEKDAY_CN[now.weekday()]}"
    next_monday = (now + timedelta(days=(7 - now.weekday()) % 7 or 7)).date()
    days = []
    for offset in range(1, 15):
        d = now + timedelta(days=offset)
        wd = _WEEKDAY_CN[d.weekday()]
        if offset == 1:
            tag = "明天"
        elif offset == 2:
            tag = "后天"
        elif d.date() < next_monday:
            tag = wd  # 本周剩余日期：周六/周日
        elif d.date() < next_monday + timedelta(days=7):
            tag = f"下{wd}"  # 下一自然周：下周一…下周日
        else:
            tag = f"下下{wd}"  # 再下一周
        days.append(f"{tag}={d:%m-%d}")
    return f"{base}；未来14天：{'，'.join(days)}"
