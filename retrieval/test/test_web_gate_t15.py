"""
T15-D0 联网门禁 + 埋点 单测（零 LLM、零联网）。

验证 `after_attraction_confirm` 按 source_policy 门禁的逻辑：
    - policy=kb          → 不挂 node_web_search_mcp（省一次 29 元/千次 的联网）
    - policy=web         → 挂（库外/时效必须有网）
    - policy=kb_then_web → 挂
    - route_info 缺失     → 保守挂（不漏联网；兼容 v1 路由/降级链路）
    - answer 兜底直出     → 直接返回答案节点
    - is_plan → 仍正确追加 weather/route

同时验证 web_search_service 埋点计数。

运行：
    ./.venv/Scripts/python.exe test/test_web_gate_t15.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.process.unified_query.agent.main_graph import after_attraction_confirm  # noqa: E402
from app.rag.common import web_search_service as wss  # noqa: E402


def _nodes(policy="kb_then_web", *, is_plan=False, answer=None):
    state: dict = {}
    if policy is not None:
        state["route_info"] = {"source_policy": policy}
    if is_plan:
        state["is_plan"] = True
    if answer:
        state["answer"] = answer
    return after_attraction_confirm(state)


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, info: str = "") -> None:
        results.append((name, cond, info))

    WEB = "node_web_search_mcp"

    # 门禁核心
    check("policy=kb 跳过联网", WEB not in _nodes("kb"))
    check("policy=web 保留联网", WEB in _nodes("web"))
    check("policy=kb_then_web 保留联网", WEB in _nodes("kb_then_web"))
    check("route_info 缺失时保守保留联网", WEB in _nodes(None))

    # 兜底与规划分支未被破坏
    n = _nodes("kb", answer="已答")
    check("answer 兜底直出（不检索）", n == "node_answer_output_tourism", str(n))
    n = _nodes("web", is_plan=True)
    check("is_plan 分支仍完整",
          WEB in n and "node_tool_weather" in n and "node_tool_route" in n, str(n))
    n = _nodes("kb", is_plan=True)
    check("is_plan + policy=kb：规划仍挂工具、但不挂 web",
          WEB not in n and "node_tool_weather" in n and "node_tool_route" in n, str(n))

    # 埋点
    before = wss.web_search_metrics()["calls"]
    wss._record_web_call({"route_info": {"source_policy": "web"}, "rewritten_query": "q"}, 3)
    m = wss.web_search_metrics()
    check("埋点 calls 自增", m["calls"] == before + 1)
    check("埋点 by_policy 记录 web", m["by_policy"].get("web", 0) >= 1)
    check("埋点 total_pages 累计", m["total_pages"] >= 3)

    print("== T15-D0 联网门禁 + 埋点 单测 ==")
    passed = 0
    for name, ok, info in results:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{info}]" if info and not ok else ""))
        passed += 1 if ok else 0
    print(f"\n{passed}/{len(results)} 通过")
    if passed != len(results):
        raise SystemExit(1)
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
