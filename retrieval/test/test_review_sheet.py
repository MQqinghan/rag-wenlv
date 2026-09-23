# -*- coding: utf-8 -*-
"""L3 人工抽检工具单测（data/eval/make_review_sheet.py）—— 离线、纯文件。

覆盖：
  1. _aggregate：与 run_eval 口径一致的四格/Acc/P/R/F1（含负类 accept 语义）
  2. _pick：必含全部失败项 + 随机补足至 ≥ratio；同 seed 可复现
  3. build：生成 review_sheet.md（含失败项与金标要点）与 review_sample.json 留痕
  4. apply_verdicts：人工结论回填后修正指标（FN→TP / FP→TN）与推翻条数
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "make_review_sheet", str(ROOT / "data" / "eval" / "make_review_sheet.py"))
mrs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mrs)

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


def case(cid, etype, accept):
    return {
        "id": cid, "query": f"问题-{cid}", "expect_type": etype,
        "domain_route": "tourism", "answer": f"答案正文-{cid}",
        "answer_judge": {"method": "judge", "accept": accept,
                         "judge_raw": {"reason": f"理由-{cid}"}, "covered": [True]},
        "retrieval": {"golden_file_titles": ["某文件"]},
        "error": None, "elapsed_s": 1.0,
    }


CASES = [
    case("a1", "answer", True),    # TP
    case("a2", "answer", False),   # FN
    case("a3", "plan", True),      # TP
    case("a4", "plan", False),     # FN
    case("c1", "chitchat", True),  # TN
    case("c2", "chitchat", False), # FP
    case("c3", "refuse", True),    # TN
    case("c4", "clarify", False),  # FP
]

# ============================================================
# 1. _aggregate 口径
# ============================================================
print("== 1. _aggregate 口径 ==")
base = mrs._aggregate(CASES, None)
check("四格 TP2/FP2/FN2/TN2",
      (base["tp"], base["fp"], base["fn"], base["tn"]) == (2, 2, 2, 2))
check("Acc=0.5", base["accuracy"] == 0.5)
check("P=R=F1=0.5", base["precision"] == 0.5 and base["recall"] == 0.5 and base["f1"] == 0.5)

_neg = mrs._aggregate([case("n1", "chitchat", True)], None)
check("负类 accept=True 记 TN（不是 TP）", _neg["tn"] == 1 and _neg["tp"] == 0)
check("accept=None 记 unresolved",
      mrs._aggregate([{"id": "x", "expect_type": "answer",
                       "answer_judge": {"accept": None}}], None)["unresolved"] == 1)

# ============================================================
# 2. _pick 抽样
# ============================================================
print("== 2. _pick 抽样 ==")
picked, fails, need = mrs._pick(CASES, 0.25, 42)
check("need=ceil(0.25*8)=2", need == 2)
check("全部失败项必含", {r["id"] for r in fails}.issubset({r["id"] for r in picked}))
check("抽检数 ≥ 目标", len(picked) >= need)
check("抽检保持原始顺序",
      [r["id"] for r in picked] == sorted([r["id"] for r in picked],
                                          key=lambda i: [c["id"] for c in CASES].index(i)))

_few_fail = CASES[:1] + [case(f"p{i}", "answer", True) for i in range(7)]
p1, _, need1 = mrs._pick(_few_fail, 0.25, 42)
p2, _, _ = mrs._pick(_few_fail, 0.25, 42)
check("失败不足时从通过项补足", len(p1) >= need1 == 2)
check("同 seed 可复现", [r["id"] for r in p1] == [r["id"] for r in p2])

# ============================================================
# 3. build
# ============================================================
print("== 3. build ==")
tmp = Path(tempfile.mkdtemp(prefix="review_test_"))
(tmp / "metrics.json").write_text(
    json.dumps({"summary": {}, "args": {"cases": "x.json"}, "cases": CASES}, ensure_ascii=False),
    encoding="utf-8")
cases_file = tmp / "cases.json"
cases_file.write_text(json.dumps(
    [{"id": c["id"], "golden_facts": [f"要点-{c['id']}"],
      "golden_file_titles": ["某文件"]} for c in CASES], ensure_ascii=False), encoding="utf-8")

mrs.build(str(tmp), 0.25, 42, str(cases_file))
sheet = (tmp / "review_sheet.md")
sample = tmp / "review_sample.json"
check("产出 review_sheet.md", sheet.exists())
check("产出 review_sample.json", sample.exists())

st = sheet.read_text(encoding="utf-8")
check("复核单含全部失败项 id", all(f"`{i}`" in st for i in ("a2", "a4", "c2", "c4")))
check("复核单含金标要点", "要点-a2" in st)
check("复核单含 judge 理由", "理由-a2" in st)
check("复核单含人工判定栏", "▢correct ▢wrong" in st)

sm = json.loads(sample.read_text(encoding="utf-8"))
check("留痕含抽样比例与种子", sm["ratio"] == 0.25 and sm["seed"] == 42 and sm["total"] == 8)
check("留痕 picked 与失败集合一致",
      set(sm["picked_fail"]) == {"a2", "a4", "c2", "c4"})
check("留痕含 judge 原始指标", sm["judge_metrics"]["accuracy"] == 0.5)

# ============================================================
# 4. apply_verdicts
# ============================================================
print("== 4. apply_verdicts ==")
vp = tmp / "review_result.json"
vp.write_text(json.dumps({"a2": "correct", "c2": "correct", "c3": "wrong"}),
              encoding="utf-8")
mrs.apply_verdicts(str(tmp), str(vp))
cm = json.loads((tmp / "corrected_metrics.json").read_text(encoding="utf-8"))
cor = cm["corrected"]
check("a2 FN→TP", cor["tp"] == 3 and cor["fn"] == 1)
check("c2 FP→TN 与 c3 TN→FP 相抵（TN/FP 各 2）", cor["tn"] == 2 and cor["fp"] == 2)
check("修正后 Acc=0.625", cor["accuracy"] == 0.625)
check("修正后 P/R/F1", (cor["precision"], cor["recall"], cor["f1"]) == (0.6, 0.75, 0.6667))
check("记录人工推翻条数", cor["human_flipped"] == 3)
check("保留 judge 基线", cm["judge_baseline"]["accuracy"] == 0.5)

print()
print(f"== review_sheet 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
