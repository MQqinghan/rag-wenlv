# -*- coding: utf-8 -*-
"""评测 A/B 对比工具单测（data/eval/compare_eval.py）—— 离线、纯文件。

覆盖：
  1. _agg：四格与指标口径（与 run_eval 一致）
  2. compare：指标 Δ 正确、翻转分类（改善/回归）、返回码（有回归 → 1）
  3. 用例集合不一致：仅对共同用例计算，并单列独有 id
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "compare_eval", str(ROOT / "data" / "eval" / "compare_eval.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

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


def case(cid, etype, accept, reason=""):
    return {
        "id": cid, "expect_type": etype, "answer_judge": {
            "method": "judge", "accept": accept,
            "judge_raw": {"reason": reason or f"r-{cid}"},
        },
    }


def write_dir(root: Path, name: str, cases: list) -> str:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(
        json.dumps({"summary": {}, "args": {}, "cases": cases}, ensure_ascii=False),
        encoding="utf-8")
    return str(d)


# ============================================================
# 1. _agg 口径
# ============================================================
print("== 1. _agg 口径 ==")
g = ce._agg([case("a", "answer", True), case("b", "answer", False),
             case("c", "chitchat", True), case("d", "chitchat", False)])
check("四格 TP1/FP1/FN1/TN1", (g["tp"], g["fp"], g["fn"], g["tn"]) == (1, 1, 1, 1))
check("Acc=0.5 / P=R=F1=0.5",
      g["accuracy"] == 0.5 and g["precision"] == 0.5 and g["recall"] == 0.5 and g["f1"] == 0.5)
check("accept=None 记 unresolved",
      ce._agg([{"id": "x", "expect_type": "answer", "answer_judge": {"accept": None}}])["unresolved"] == 1)

# ============================================================
# 2. compare 正常对比
# ============================================================
print("== 2. compare 正常对比 ==")
tmp = Path(tempfile.mkdtemp(prefix="cmp_test_"))
base = write_dir(tmp, "base", [case("a1", "answer", True), case("a2", "answer", False),
                               case("c1", "chitchat", True), case("c2", "chitchat", False)])
new = write_dir(tmp, "new", [case("a1", "answer", True), case("a2", "answer", True),
                             case("c1", "chitchat", False), case("c2", "chitchat", False)])
jd = str(tmp / "diff.json")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = ce.compare(base, new, show_flips=True, json_out=jd)
out = buf.getvalue()

check("有回归 → 返回码 1", rc == 1)
check("识别 2 条翻转", "翻转用例 2 条" in out)
check("含改善与回归各 1", "改善 1 / 回归 1" in out)
check("输出含 Δ 行", "Δ" in out)

detail = json.loads(Path(jd).read_text(encoding="utf-8"))
check("明细 common=4", detail["common"] == 4)
# a2 改善(FN→TP) 与 c1 回归(TN→FP) 相抵 → Acc 净不变；但 TP 增加使 Recall/F1 上升
check("Δ accuracy 净不变（改喜与回归相抵）", detail["delta"]["accuracy"] == 0.0)
check("Δ recall > 0（TP 增加）", detail["delta"]["recall"] > 0)
check("Δ f1 > 0", detail["delta"]["f1"] > 0)
flips = {f["id"]: f["direction"] for f in detail["flips"]}
check("a2 判为改善", flips.get("a2", "").startswith("改善"))
check("c1 判为回归", flips.get("c1", "").startswith("回归"))

# 无回归时返回 0
new_ok = write_dir(tmp, "new_ok", [case("a1", "answer", True), case("a2", "answer", True),
                                   case("c1", "chitchat", True), case("c2", "chitchat", False)])
with contextlib.redirect_stdout(io.StringIO()):
    rc2 = ce.compare(base, new_ok, show_flips=False, json_out=None)
check("无回归 → 返回码 0", rc2 == 0)

# ============================================================
# 3. 用例集合不一致
# ============================================================
print("== 3. 用例集合不一致 ==")
part = write_dir(tmp, "part", [case("a1", "answer", True), case("a2", "answer", False),
                               case("zz", "answer", True)])
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    rc3 = ce.compare(base, part, show_flips=False, json_out=str(tmp / "d2.json"))
o2 = buf2.getvalue()
check("提示仅对共同用例计算", "仅对共同用例计算" in o2)
d2 = json.loads((tmp / "d2.json").read_text(encoding="utf-8"))
check("common=2（a1,a2）", d2["common"] == 2)
check("new 独有 zz 单列", d2["only_new"] == ["zz"])
check("base 独有 c1/c2 单列", set(d2["only_base"]) == {"c1", "c2"})

print()
print(f"== compare_eval 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
