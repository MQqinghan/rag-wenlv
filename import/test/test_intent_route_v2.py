# -*- coding: utf-8 -*-
"""
B1 v2 意图路由（LLM 主导 + 正则护栏/降级）单测（T5 阶段验收）。

运行方式：
  python test/test_intent_route_v2.py

覆盖：
1) parse_route_v2_result：合法归一 / 缺 domains / 非法域 / tools 过滤 / policy 非法回落 kb；
2) v2 确定性护栏层优先于 LLM（datetime/confusion 在 LLM 调用前短路）；
3) LLM v2 成功 → state.route_info 写入 + domain 返回；
4) 知识型句式被 LLM 判闲聊 → 推翻为 tourism；
5) LLM v2 失败/非法 → 降级 match_by_rule → 旧兜底（不抛异常，返回合法域）；
6) dispatch 默认走 v1（INTENT_ROUTER_LLM_FIRST 缺省 false）。
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import app.rag.common.intent_route_service as irs  # noqa: E402
from app.rag.common.intent_route_service import (  # noqa: E402
    DOMAIN_CHITCHAT,
    DOMAIN_TOURISM,
)

# 保证开关按测试内环境变量重新读取
irs.INTENT_ROUTER_LLM_FIRST = os.environ.get("INTENT_ROUTER_LLM_FIRST", "false").lower() in ("1", "true", "yes", "on")


def _clear_llm():
    """后续用例通过替换模块函数控制 LLM 行为（不真实联网）。"""
    return None


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    # ---- 1) parse_route_v2_result ----
    ok = irs.parse_route_v2_result({
        "rewritten_query": "杭州三日怎么玩", "domains": ["tourism"],
        "source_policy": "kb_then_web", "tools": ["weather", "route"],
        "is_plan": True, "need_clarify": False, "reason": "r",
    })
    assert ok and ok["domains"] == ["tourism"] and ok["source_policy"] == "kb_then_web"
    assert ok["tools"] == ["weather", "route"] and ok["is_plan"] is True
    assert irs.parse_route_v2_result({"domains": []}) is None
    assert irs.parse_route_v2_result({"domains": ["music"]}) is None
    assert irs.parse_route_v2_result({"domains": ["chitchat"], "tools": ["jump", "none"], "source_policy": "bad"}) == {
        "rewritten_query": "", "domains": ["chitchat"], "source_policy": "kb",
        "tools": ["none"], "is_plan": False, "need_clarify": False, "reason": ""}
    assert irs.parse_route_v2_result("not-a-dict") is None
    results.append(("1 输出schema归一/非法回落", True, "ok"))

    # ---- 2) 护栏层短路（datetime/confusion）不触发 LLM ----
    calls = {"n": 0}
    orig = irs.classify_route_v2

    def counting_v2(*a, **k):
        calls["n"] += 1
        return None

    irs.classify_route_v2 = counting_v2
    try:
        assert irs.classify_intent_v2({"original_query": "今天是几号", "session_id": "x"}) == DOMAIN_CHITCHAT
        assert irs.classify_intent_v2({"original_query": "？", "session_id": "x"}) == DOMAIN_CHITCHAT
        assert irs.classify_intent_v2({"original_query": "今天天气不错啊，你好"}) == DOMAIN_CHITCHAT  # 纯寒暄走规则/LLM
        assert calls["n"] <= 1, "护栏层不应触发多余 LLM 调用"
        results.append(("2 护栏层优先短路", True, f"LLM 调用={calls['n']}"))
    finally:
        irs.classify_route_v2 = orig

    # ---- 3) LLM v2 成功 → route_info 写入 ----
    orig = irs.classify_route_v2

    def fake_route(*a, **k):
        return {"rewritten_query": "成都两天怎么安排", "domains": ["tourism"],
                "source_policy": "kb_then_web", "tools": ["route", "weather"],
                "is_plan": True, "need_clarify": False, "reason": "f"}

    irs.classify_route_v2 = fake_route
    try:
        st = {"original_query": "我下周去成都玩两天怎么安排？", "session_id": "x"}
        domain = irs.classify_intent_v2(st)
        assert domain == DOMAIN_TOURISM
        assert st["route_info"]["domains"] == ["tourism"]
        assert st["route_info"]["source_policy"] == "kb_then_web"
        assert st["is_plan"] is True
        results.append(("3 v2成功写route_info", True, f"tools={st['route_info']['tools']}"))
    finally:
        irs.classify_route_v2 = orig

    # ---- 4) 知识句式被 LLM 判闲聊 → 推翻 tourism ----
    orig = irs.classify_route_v2

    def fake_chat(*a, **k):
        return {"rewritten_query": "什么是 X", "domains": ["chitchat"],
                "source_policy": "web", "tools": ["none"], "is_plan": False,
                "need_clarify": False, "reason": "f"}

    irs.classify_route_v2 = fake_chat
    try:
        st = {"original_query": "什么是马尔代夫的海岛成因", "session_id": "x"}
        domain = irs.classify_intent_v2(st)
        assert domain == DOMAIN_TOURISM, f"知识句式应推翻为 tourism，实际 {domain}"
        results.append(("4 知识句式推翻闲聊", True, domain))
    finally:
        irs.classify_route_v2 = orig

    # ---- 5) LLM 失败 → 降级规则（"上海美食攻略"命中文旅关键词） ----
    orig = irs.classify_route_v2

    def failing_v2(*a, **k):
        return None

    irs.classify_route_v2 = failing_v2
    try:
        st = {"original_query": "上海美食攻略推荐", "session_id": "x"}
        domain = irs.classify_intent_v2(st)
        assert domain == DOMAIN_TOURISM, f"规则降级应到 tourism，实际 {domain}"
        results.append(("5 失败降级不抛且出合法域", True, f"{domain}"))
    finally:
        irs.classify_route_v2 = orig

    # ---- 6) dispatch 默认走 v2（T5 已拍板 INTENT_ROUTER_LLM_FIRST 默认 true）----
    os.environ.pop("INTENT_ROUTER_LLM_FIRST", None)
    importlib.reload(irs)
    assert irs.INTENT_ROUTER_LLM_FIRST is True
    assert irs.classify_intent_dispatch({"original_query": "成都有什么好吃的", "session_id": "x"}) in (
        DOMAIN_TOURISM, DOMAIN_CHITCHAT)
    results.append(("6 dispatch 默认 v2", True, "INTENT_ROUTER_LLM_FIRST=True"))

    # ---- 7) ⑤ 方案 C 守卫：LLM 判 is_plan=True 但问句无行程定义词 → 确定性降回 False ----
    # 背景见 docs/is_plan放宽评估.md：修好示例锚定后 LLM 转宽松，会把"咨询类"误判为行程；
    # 该守卫用可枚举规则收敛（LLM 宽召回 + 代码严收敛）。此处 LLM 一律返回 is_plan=True。
    orig = irs.classify_route_v2
    guard_cases = [
        # (问句, 期望最终 state.is_plan)
        ("我想去云南玩 5 天，昆明、大理、丽江怎么安排比较顺？", True),      # 阿拉伯数字天数
        ("想去厦门玩两三天，鼓浪屿和市区怎么安排？", True),                # 中文数词天数
        ("帮我安排一下从成都去九寨沟的行程", True),                        # "行程"
        ("我到张家界市区了，接下来天门山和森林公园怎么安排着玩？", True),   # "接下来"
        ("第一次去杭州，景点应该怎么安排？", False),                       # 咨询类（tr-008 判例）→ 降回
        ("成都有什么好吃的？", False),                                     # 知识咨询（示例B 口径）→ 降回
        ("云南什么季节去合适？", False),                                   # 季节咨询 → 降回
    ]

    def _fake_llm_plan(q: str) -> dict:
        return {"rewritten_query": q, "domains": ["tourism"], "source_policy": "kb",
                "tools": ["route", "weather"], "is_plan": True,
                "need_clarify": False, "reason": "f"}

    try:
        for q, expect in guard_cases:
            irs.classify_route_v2 = (lambda q=q, *a, **k: _fake_llm_plan(q))
            st = {"original_query": q, "session_id": "x"}
            irs.classify_intent_v2(st)
            got = bool(st.get("is_plan"))
            assert got == expect, f"守卫判定错: [{q}] 期望 is_plan={expect} 实得 {got}"
            if expect is False and st.get("route_info"):
                assert "route" not in st["route_info"]["tools"], \
                    f"降回后 tools 不应保留 route: [{q}] -> {st['route_info']['tools']}"
        results.append(("7 is_plan 守卫（LLM 宽召回→代码严收敛）", True, f"{len(guard_cases)}/{len(guard_cases)}"))
    finally:
        irs.classify_route_v2 = orig

    print("== intent_route_v2（B1）单测结果 ==")
    ok = True
    for name, passed, info in results:
        print(f"  {'✅' if passed else '❌'} {name}：{info}")
        ok = ok and passed
    print("全部通过 ✅" if ok else "存在失败 ❌")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
