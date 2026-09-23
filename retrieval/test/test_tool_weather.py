"""
规划类 API 工具端到端验证脚本（天气=和风天气 / 路线=高德，需 .env 配置 Key）：
1. 校验统一查询图包含工具节点、is_plan 并行分支、rerank 后行程拼装分流
2. 天气工具：真实调用 LLM 解析目的地 → 和风定位 → 逐日预报 → 简报组装
3. 路线工具：真实调用高德地理编码 + 驾车路径规划（含油费估算）→ 简报组装
4. 答案 Prompt 渲染：工具简报正确拼入，缺失时占位
5. 行程拼装：真实调用 LLM 生成结构化行程（KB 上下文用模拟数据）
运行：python test/test_tool_weather.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.process.unified_query.agent.main_graph import (
    after_attraction_confirm,
    after_rerank,
    unified_query_app,
)
from app.process.unified_query.agent.state import create_unified_default_state
from app.rag.tourism_query.itinerary_service import generate_itinerary
from app.rag.tourism_query.route_tool_service import plan_route
from app.rag.tourism_query.weather_tool_service import get_weather_brief

_FAKE_DOCS = [
    {"title": "九寨沟游玩攻略", "score": 0.9, "type": "milvus",
     "text": "九寨沟位于四川省阿坝藏族羌族自治州，以翠海、叠瀑、彩林、雪峰闻名。"
             "景区实行分时段预约入园，旺季建议早到。沟内以观光车代步，主要游览树正沟、日则沟、则查洼沟。"},
    {"title": "九寨沟门票信息", "score": 0.8, "type": "milvus",
     "text": "九寨沟门票与观光车票分旺季淡季，具体价格以景区官方公告为准，建议提前在线预约。"},
]


def test_graph_wiring():
    for node in ("node_tool_weather", "node_tool_route", "node_itinerary_generate"):
        assert node in unified_query_app.nodes, f"统一图中缺少 {node} 节点"
    plan_nodes = after_attraction_confirm({"is_plan": True})
    assert "node_tool_weather" in plan_nodes and "node_tool_route" in plan_nodes, "is_plan=True 时应追加工具分支"
    normal_nodes = after_attraction_confirm({"is_plan": False})
    assert "node_tool_weather" not in normal_nodes, "非规划问题不应调用工具"
    assert after_rerank({"domain": "tourism", "is_plan": True}) == "node_itinerary_generate", "规划类应走行程拼装"
    assert after_rerank({"domain": "tourism", "is_plan": False}) == "node_answer_output_tourism", "普通问题不受影响"
    print("[1/5] 图接线校验通过")


def test_weather_end_to_end():
    state = create_unified_default_state(original_query="我明天想去九寨沟，帮我规划一下行程。")
    brief = get_weather_brief(state)
    assert isinstance(brief, dict) and set(brief) >= {"ok", "destination", "text"}
    print(f"[2/5] 天气简报 ok={brief['ok']} destination={brief['destination']!r}")
    print("-" * 50)
    print(brief["text"])
    print("-" * 50)
    if brief["ok"]:
        assert "九寨沟" in brief["destination"] and "°C" in brief["text"]
        print("天气工具端到端通过")
    else:
        print("警告：未拿到天气数据（检查 QWEATHER_API_KEY / QWEATHER_API_HOST / GeoAPI 权限）")


def test_route_end_to_end():
    state = create_unified_default_state(original_query="我从成都出发，明天想去九寨沟，帮我规划一下行程。")
    brief = plan_route(state)
    assert isinstance(brief, dict) and set(brief) >= {"ok", "origin", "destination", "text"}
    print(f"[3/5] 路线简报 ok={brief['ok']} {brief.get('origin')!r} -> {brief.get('destination')!r}")
    print("-" * 50)
    print(brief["text"])
    print("-" * 50)
    if brief["ok"]:
        assert "公里" in brief["text"] and "油费" in brief["text"], "简报缺少距离或油费信息"
        print("路线工具端到端通过")
    else:
        print("警告：未拿到路线数据（检查 AMAP_API_KEY / 出发地是否被解析）")


def test_answer_prompt_render():
    from app.rag.tourism_query.answer_output_service import build_answer_prompt
    p1 = build_answer_prompt(
        _FAKE_DOCS, "我明天想去九寨沟帮我规划一下", ["九寨沟"], [],
        current_date="2026-09-03 周四",
        tool_weather="目的地:九寨沟\n2026-09-04:小雨,12~20°C",
        tool_route="成都→九寨沟:全程约405公里,驾车约4小时38分钟,预计油费约253元",
    )
    p2 = build_answer_prompt(_FAKE_DOCS, "九寨沟值得去吗", ["九寨沟"], [], current_date="2026-09-03 周四")
    assert "【实时天气参考】" in p1 and "小雨" in p1, "天气未拼入"
    assert "【路线参考】" in p1 and "405公里" in p1, "路线未拼入"
    assert "（本次无实时天气数据）" in p2 and "（本次无路线规划数据）" in p2, "空占位缺失"
    print("[4/5] 答案 Prompt 渲染校验通过")


def test_itinerary_generation():
    state = create_unified_default_state(original_query="我从成都出发，明天想去九寨沟，帮我规划一下行程。")
    state["reranked_docs"] = _FAKE_DOCS
    state["item_names"] = ["九寨沟"]
    state["tool_weather"] = {"ok": True, "destination": "九寨沟",
                             "text": "目的地:九寨沟\n2026-09-04:小雨,13~25°C,降水量2mm\n2026-09-05:阴,14~27°C,降水量0mm"}
    state["tool_route"] = {"ok": True, "origin": "成都", "destination": "九寨沟",
                           "text": "成都→九寨沟:全程约405公里,驾车约4小时38分钟,预计油费约253元（按百公里8L、油价7.8元/L估算）"}
    update = generate_itinerary(state)
    answer = update["answer"]
    assert isinstance(answer, str) and len(answer) > 50, "行程内容过短"
    assert "油费" in answer or "253" in answer, "行程未使用路线/油费数据"
    print("[5/5] 行程拼装端到端通过（前 400 字预览）")
    print("-" * 50)
    print(answer[:400])
    print("-" * 50)


if __name__ == "__main__":
    test_graph_wiring()
    test_weather_end_to_end()
    test_route_end_to_end()
    test_answer_prompt_render()
    test_itinerary_generation()
