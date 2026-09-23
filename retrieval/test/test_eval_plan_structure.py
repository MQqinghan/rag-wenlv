# -*- coding: utf-8 -*-
"""补评测盲区：plan 类结构断言离线单测（零外部请求、零 LLM）。

覆盖 run_eval.py 的三处新增能力：
  1. _plan_structure_signals(state)：结构信号判定
  2. l1_check(...)：plan 类的 hard_fail_plan_structure 硬信号
  3. judge_case(...)：plan 类 L1 前置结构门禁（该出行程却走普通问答 → accept=False）
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_eval_under_test", ROOT / "data" / "eval" / "run_eval.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# ============================================================
# 1. 结构信号
# ============================================================
print("== 1. _plan_structure_signals ==")
check("全空 state → structured=False", m._plan_structure_signals({})["structured"] is False)
check("is_plan=True → structured=True", m._plan_structure_signals({"is_plan": True})["structured"] is True)
check("is_plan=False → structured=False", m._plan_structure_signals({"is_plan": False})["structured"] is False)
check("map_data={} → structured=False", m._plan_structure_signals({"map_data": {}})["structured"] is False)
check("map_data 非空 → structured=True", m._plan_structure_signals({"map_data": {"points": [1]}})["structured"] is True)
check("tool_weather 命中 → structured=True", m._plan_structure_signals({"tool_weather": {"ok": True}})["structured"] is True)
check("tool_route 命中 → 计数=1", m._plan_structure_signals({"tool_route": {"ok": True}})["plan_tools"] == ["tool_route"])
check("tool_budget 命中 → structured=True", m._plan_structure_signals({"tool_budget": {"total": 1}})["structured"] is True)
check("命中列表保留全部键", m._plan_structure_signals({"tool_weather": 1, "tool_route": 1, "tool_budget": 1})["plan_tools"] == ["tool_weather", "tool_route", "tool_budget"])

# ============================================================
# 2. l1_check 结构信号
# ============================================================
print("== 2. l1_check ==")
plan_case = {"id": "plan-x", "expect_type": "plan", "query": "成都三日游怎么安排"}
answer_case = {"id": "ab-x", "expect_type": "answer", "query": "熊猫基地开放时间"}

l1 = m.l1_check(plan_case, {"answer": "有内容", "reranked_docs": [{"a": 1}]})
check("plan 无结构 → hard_fail_plan_structure=True", l1["hard_fail_plan_structure"] is True)
check("plan 无结构 → plan_structure 存在", "structured" in l1.get("plan_structure", {}))

l1b = m.l1_check(plan_case, {"answer": "有内容", "reranked_docs": [{"a": 1}], "is_plan": True})
check("plan 判为规划 → hard_fail_plan_structure=False", l1b["hard_fail_plan_structure"] is False)

l1c = m.l1_check(answer_case, {"answer": "有内容", "reranked_docs": [{"a": 1}]})
check("answer 类 → 不产 plan 结构信号", l1c["plan_structure"] == {} and l1c["hard_fail_plan_structure"] is False)

l1d = m.l1_check(plan_case, {"answer": "有内容", "reranked_docs": [], "map_data": {"x": 1}})
check("plan 有 map_data → 结构达标（但检索空仍 hard_fail_retrieval）", l1d["hard_fail_plan_structure"] is False and l1d["hard_fail_retrieval"] is True)

# ============================================================
# 3. judge_case 前置结构门禁
# ============================================================
print("== 3. judge_case ==")
# 3a. 触发门禁：plan + 无结构 + 非空回答 → 直接 accept=False，不调 judge
jd = m.judge_case(plan_case, {"answer": "我建议你三天玩遍成都各大景点……", "reranked_docs": [{"a": 1}]})
check("plan 无结构 → method=L1_plan_structure", jd.get("method") == "L1_plan_structure")
check("plan 无结构 → accept=False", jd.get("accept") is False)
check("plan 无结构 → 带 plan_structure 证据", "plan_structure" in jd)

# 3b. 结构化 plan → 放行到 judge（monkeypatch，不真调 LLM）
_orig_judge = m._call_judge
m._call_judge = lambda s, u: {"accept": True, "hallucination": False, "covered": [True], "reason": "ok"}
try:
    jd2 = m.judge_case(plan_case, {"answer": "第一天……", "reranked_docs": [{"a": 1}], "is_plan": True})
    check("plan 已结构化 → 走 judge（method=judge）", jd2.get("method") == "judge")
    check("plan 已结构化 → accept=True（透传 judge）", jd2.get("accept") is True)

    # 3c. 门禁开关关闭 → 结构化与否都放行
    m._PLAN_STRUCTURE_ENFORCE = False
    jd3 = m.judge_case(plan_case, {"answer": "随便答", "reranked_docs": [{"a": 1}]})
    check("门禁 OFF → 无结构也走 judge", jd3.get("method") == "judge")
    m._PLAN_STRUCTURE_ENFORCE = True

    # 3d. 非 plan 类不受门禁影响
    jd4 = m.judge_case(answer_case, {"answer": "有内容", "reranked_docs": [{"a": 1}]})
    check("answer 类不受 plan 门禁影响", jd4.get("method") == "judge")
finally:
    m._call_judge = _orig_judge

# 3e. 空回答优先于门禁
jd5 = m.judge_case(plan_case, {"answer": "", "reranked_docs": []})
check("空回答 → 优先判空（method=L1_only）", jd5.get("accept") is False and jd5.get("reason") == "回答为空")

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
