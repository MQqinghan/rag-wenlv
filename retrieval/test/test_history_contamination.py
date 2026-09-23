"""
历史会话污染改写问题的回归验证（2026-09-03 线上问题复现）：
现象：会话历史中存在"九寨沟"问答时，新问题"我周5打算从深圳去成都旅游，帮我规划一下"
被改写/主体抽取污染为九寨沟，导致天气定位九寨沟、路线工具误判无出发地。
验证：
1. 改写+主体抽取在污染历史下不被带偏（不得出现九寨沟）
2. 行程信息解析以原始问题为准（origin=深圳, destination=成都）
3. 长历史回答被截断（防止单条历史刷屏）
运行：python test/test_history_contamination.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.common.history_text_utils import (
    HISTORY_MSG_MAX_CHARS,
    build_history_text,
)
from app.rag.tourism_query.attraction_confirm_service import (
    rewrite_query_and_extract_attractions,
)
from app.rag.tourism_query.weather_tool_service import extract_destination_info

NEW_QUERY = "我周5打算从深圳去成都旅游，帮我规划一下"

# 模拟污染历史：上一轮问了九寨沟规划，助手返回了超长行程
POLLUTED_HISTORY = [
    {"role": "user", "text": "我明天想去九寨沟，帮我规划一下",
     "rewritten_query": "我明天想去九寨沟，帮我规划一下行程", "item_names": ["九寨沟"]},
    {"role": "assistant", "text":
        "行程概览：出行日期2026-09-04，成都→九寨沟，驾车约405公里/4小时38分，油费约253元。"
        "第一天：抵达九寨沟后游览树正沟，欣赏翠海与叠瀑；"
        "第二天：全天游览日则沟与则查洼沟，观赏诺日朗瀑布与五彩池；"
        "花销预估：门票+观光车+住宿+餐饮约800元/人；装备注意：山区早晚温差大，建议携带外套与雨具。" * 3,
     "item_names": ["九寨沟"]},
]


def test_history_truncation():
    text = build_history_text(POLLUTED_HISTORY)
    for line in text.splitlines():
        content = line.split("内容:")[1].split(",关联主体")[0] if "内容:" in line else ""
        assert len(content) <= HISTORY_MSG_MAX_CHARS + len("…(已截断)"), f"历史消息未截断: {len(content)}字"
    print(f"[PASS] 长历史回答已截断（单条上限 {HISTORY_MSG_MAX_CHARS} 字）")


def test_rewrite_not_polluted():
    result = rewrite_query_and_extract_attractions(POLLUTED_HISTORY, NEW_QUERY)
    attractions = result.get("attractions", [])
    rewritten = result.get("rewritten_query", "")
    assert "九寨沟" not in str(attractions), f"主体抽取被历史污染: {attractions}"
    assert "九寨沟" not in rewritten, f"改写问题被历史污染: {rewritten}"
    print(f"[PASS] 改写未受污染：attractions={attractions}, rewritten_query={rewritten}")
    return rewritten


def test_plan_extract_uses_original(rewritten: str):
    state = {
        "original_query": NEW_QUERY,
        "rewritten_query": rewritten,
        "is_plan": True,
    }
    info = extract_destination_info(state)
    assert info["origin"] == "深圳", f"出发地解析错误: {info}"
    assert "成都" in info["destination"], f"目的地解析错误: {info}"
    assert "九寨沟" not in str(info), f"行程解析被污染: {info}"
    print(f"[PASS] 行程解析以原始问题为准：origin={info['origin']}, destination={info['destination']}")


if __name__ == "__main__":
    test_history_truncation()
    rewritten = test_rewrite_not_polluted()
    test_plan_extract_uses_original(rewritten)
    print("历史污染回归验证全部通过")
