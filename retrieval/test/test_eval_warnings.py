# -*- coding: utf-8 -*-
"""OBS-5：评测 unresolved(accept=None) 显著告警 · 离线单测（零外部请求、零 LLM）。

覆盖 data/eval/run_eval.py 的 collect_eval_warnings：
  1. 无 unresolved → 返回空列表（不干扰正常报告）
  2. 存在 unresolved → 返回显著告警，含 case_id 与条数
  3. 多条 unresolved → 占比计算正确
  4. aggregate_answer_metrics 口径不变（unresolved 仍不计入分母，仅单独统计）
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


def _rec(cid, expect_type, accept):
    return {
        "id": cid,
        "expect_type": expect_type,
        "answer_judge": {"accept": accept},
    }


# ============================================================
# 1. 无 unresolved → 空告警
recs_clean = [
    _rec("a-001", "answer", True),
    _rec("a-002", "answer", False),   # 真实 FN
    _rec("neg-001", "refuse", True),  # 真实 TN
]
warns_clean = m.collect_eval_warnings(recs_clean, m.aggregate_answer_metrics(recs_clean))
check("无 unresolved → 告警列表为空", warns_clean == [])
check("无告警时不含未判定字样", not any("未判定" in w for w in warns_clean))

# ============================================================
# 2. 单条 unresolved → 显著告警
recs_1 = recs_clean + [_rec("xr-004", "answer", None)]  # judge 失败
warns_1 = m.collect_eval_warnings(recs_1, m.aggregate_answer_metrics(recs_1))
check("1 条 unresolved → 产出 1 条告警", len(warns_1) == 1)
check("告警含 case_id(xr-004)", "xr-004" in warns_1[0])
check("告警含『未判定』标识", "未判定" in warns_1[0])
check("告警含占比说明", "25.0%" in warns_1[0] or "25%" in warns_1[0])

# ============================================================
# 3. 多条 unresolved → 条数与占比
recs_2 = [
    _rec("a-001", "answer", True),
    _rec("xr-002", "answer", None),
    _rec("xr-004", "answer", None),
    _rec("xr-008", "answer", None),
]
warns_2 = m.collect_eval_warnings(recs_2, m.aggregate_answer_metrics(recs_2))
check("3 条 unresolved → 产出 1 条告警", len(warns_2) == 1)
check("告警含全部 3 个 case_id",
      all(cid in warns_2[0] for cid in ("xr-002", "xr-004", "xr-008")))
check("告警含占比 75%", "75.0%" in warns_2[0] or "75%" in warns_2[0])

# ============================================================
# 4. 口径不变：unresolved 不计入分母，仅单独统计
agg = m.aggregate_answer_metrics(recs_2)  # 1 tp, 3 unresolved
check("unresolved 单独计数=3", agg["unresolved"] == 3)
check("unresolved 不计入四格分母(n=tp+fp+fn+tn=1)",
      agg["tp"] + agg["fp"] + agg["fn"] + agg["tn"] == 1)
check("真实 FN 不被 unresolved 覆盖(tp=1, fn=0)",
      agg["tp"] == 1 and agg["fn"] == 0)

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
