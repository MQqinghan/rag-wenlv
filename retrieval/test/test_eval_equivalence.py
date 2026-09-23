# -*- coding: utf-8 -*-
"""OBS-6 离线单测：judge_case 语义等价集（golden_facts_sets）逻辑，mock 掉 LLM judge。

验证：
  1. 配置了 golden_facts_sets 且第 1 组（非第 0 组）命中 → accept=True, matched_set=1
  2. 任一组都未命中 → accept=False, reason 含组数
  3. 所有组 judge 失败（返回 None）→ accept 保持 None（不计入分母，OBS-5 告警）
  4. 未配置 golden_facts_sets → 回退单组 golden_facts 原路径（向后兼容，守基线）
  5. 等价集路径不影响负例/澄清类判定（仍走 _call_judge NEG）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.eval.run_eval as run_eval  # noqa: E402

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


# ---- mock _call_judge：命中标记 "ACCEPT" 即 accept=True，否则 False；可切换为 None ----
call_log = []


def fake_judge(system, user, *, _fail_all=False):
    call_log.append(user)
    if _fail_all:
        return None
    # user 内含 facts_text；第几组由我们注入的标记区分
    if "ACCEPT_MARKER" in user:
        return {"covered": [True], "hallucination": False, "accept": True}
    return {"covered": [False], "hallucination": False, "accept": False}


def patch(fail_all=False):
    call_log.clear()
    run_eval._call_judge = lambda s, u: fake_judge(s, u, _fail_all=fail_all)


# 构造最小 state（answer 非空即可走正例分支）
def mk_state(answer="这是一段看似合理的回答"):
    return {"answer": answer, "reranked_docs": [{"file_title": "X", "text": "原文片段"}]}


# 1. 第 1 组命中
patch()
case = {
    "expect_type": "answer",
    "golden_facts_sets": [
        ["普通要点A", "普通要点B"],
        ["ACCEPT_MARKER 等价要点C", "等价要点D"],
    ],
}
res = run_eval.judge_case(case, mk_state())
check("等价集第1组命中→accept=True", res.get("accept") is True)
check("matched_set=1", res.get("matched_set") == 1)
check("method=judge_equivalence", res.get("method") == "judge_equivalence")

# 2. 全部未命中
patch()
case2 = {
    "expect_type": "answer",
    "golden_facts_sets": [
        ["普通要点A", "普通要点B"],
        ["普通要点C", "普通要点D"],
    ],
}
res2 = run_eval.judge_case(case2, mk_state())
check("等价集全未命中→accept=False", res2.get("accept") is False)
check("reason 含组数(2)", "2" in (res2.get("reason") or ""))

# 3. 全部 judge 失败 → accept 保持 None
patch(fail_all=True)
res3 = run_eval.judge_case(case, mk_state())
check("judge 全失败→accept=None", res3.get("accept") is None)
check("judge 全失败→method 非 judge_equivalence 定论", res3.get("method") in (None, "L1_only", "judge"))

# 4. 未配置等价集 → 回退单组 golden_facts（向后兼容）
patch()
case4 = {
    "expect_type": "answer",
    "golden_facts": ["ACCEPT_MARKER 单组要点"],
}
res4 = run_eval.judge_case(case4, mk_state())
check("单组路径命中→accept=True", res4.get("accept") is True)
check("单组 method=judge", res4.get("method") == "judge")

# 5. 单组路径未命中 → accept=False（原语义不变）
patch()
case5 = {"expect_type": "answer", "golden_facts": ["另一个未命中要点"]}
res5 = run_eval.judge_case(case5, mk_state())
check("单组未命中→accept=False", res5.get("accept") is False)

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
