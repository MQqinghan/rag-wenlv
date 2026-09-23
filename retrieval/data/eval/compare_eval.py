#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测 A/B 对比：比较两个评测输出目录，给出指标差与逐用例翻转。

用途：
    - 指标决策（如 B1 开关：INTENT_ROUTE_LLM_FIRST ON vs OFF）的**对照证据**；
    - 代码改动前后的**回归定位**（哪些用例从对变错 / 从错变对）。

用法：
    python data/eval/compare_eval.py --base output/eval/A --new output/eval/B
    python data/eval/compare_eval.py --base A --new B --show-flips   # 打印翻转用例明细
    python data/eval/compare_eval.py --base A --new B --json out.json

口径说明：
    accept=True 表示该用例答案整体可接受（与 run_eval 一致）。
    「翻转」= 同一 id 在两次评测中 accept 不同。
    仅比较**两次都出现**的用例；各自独有的用例单列（避免指标被样本差污染）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_METRIC_KEYS = ("accuracy", "precision", "recall", "f1")


def _load(out_dir: str) -> dict:
    p = Path(out_dir) / "metrics.json"
    if not p.exists():
        raise SystemExit(f"[错误] 未找到 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _index(d: dict) -> dict:
    return {c.get("id"): c for c in (d.get("cases") or [])}


def _accept(rec: dict):
    return (rec.get("answer_judge") or {}).get("accept")


def _agg(recs: list) -> dict:
    """按 run_eval 口径重算（仅用传入的用例集合）。"""
    tp = fp = fn = tn = unresolved = 0
    for r in recs:
        acc = _accept(r)
        if acc is None:
            unresolved += 1
            continue
        pos = r.get("expect_type") in ("answer", "plan")
        if pos and acc:
            tp += 1
        elif pos and not acc:
            fn += 1
        elif (not pos) and acc:
            tn += 1
        else:
            fp += 1
    n = tp + fp + fn + tn
    acc_ = (tp + tn) / n if n else 0.0
    p_ = tp / (tp + fp) if (tp + fp) else 0.0
    r_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unresolved": unresolved,
        "accuracy": round(acc_, 4), "precision": round(p_, 4),
        "recall": round(r_, 4), "f1": round(f1_, 4),
    }


def compare(base_dir: str, new_dir: str, show_flips: bool, json_out: str | None) -> int:
    bd, nd = _load(base_dir), _load(new_dir)
    bi, ni = _index(bd), _index(nd)
    common = [i for i in bi if i in ni]
    only_b = sorted(set(bi) - set(ni))
    only_n = sorted(set(ni) - set(bi))

    b_common = [bi[i] for i in common]
    n_common = [ni[i] for i in common]
    A = _agg(b_common)
    B = _agg(n_common)

    flips = []
    for i in common:
        a, b = _accept(bi[i]), _accept(ni[i])
        if a != b:
            flips.append({
                "id": i,
                "expect_type": bi[i].get("expect_type"),
                "base": a, "new": b,
                "direction": "改善 错→对" if (a is False and b is True) else
                             ("回归 对→错" if (a is True and b is False) else "涉及未判"),
                "base_reason": ((bi[i].get("answer_judge") or {}).get("judge_raw") or {}).get("reason", ""),
                "new_reason": ((ni[i].get("answer_judge") or {}).get("judge_raw") or {}).get("reason", ""),
            })
    improved = [f for f in flips if f["direction"].startswith("改善")]
    regressed = [f for f in flips if f["direction"].startswith("回归")]

    print("=" * 62)
    print("评测 A/B 对比")
    print("=" * 62)
    print(f"  base = {base_dir}")
    print(f"  new  = {new_dir}")
    print(f"  共同用例 {len(common)} 条（base 独有 {len(only_b)} / new 独有 {len(only_n)}）")
    print("-" * 62)
    print(f"  {'指标':<12}{'base':>10}{'new':>10}{'Δ':>10}")
    for k in _METRIC_KEYS:
        d = round(B[k] - A[k], 4)
        dtxt = f"{d:+}" if d != 0 else "0"
        flag = "" if d == 0 else ("  ▲" if d > 0 else "  ▼")
        print(f"  {k:<12}{A[k]:>10}{B[k]:>10}{dtxt:>10}{flag}")
    a4 = f"{A['tp']}/{A['fp']}/{A['fn']}/{A['tn']}"
    b4 = f"{B['tp']}/{B['fp']}/{B['fn']}/{B['tn']}"
    print(f"  {'TP/FP/FN/TN':<12}{a4:>10}{b4:>10}")
    print("-" * 62)
    print(f"  翻转用例 {len(flips)} 条：改善 {len(improved)} / 回归 {len(regressed)}")
    for f in flips:
        print(f"    [{f['direction']}] {f['id']} ({f['expect_type']})  {f['base']} -> {f['new']}")
        if show_flips and f["new_reason"]:
            print(f"        new judge: {f['new_reason'][:110]}")

    if only_b or only_n:
        print("-" * 62)
        print(f"  [提示] 用例集合不一致，已仅对共同用例计算指标：")
        if only_b:
            print(f"    base 独有：{only_b[:12]}{' …' if len(only_b) > 12 else ''}")
        if only_n:
            print(f"    new 独有：{only_n[:12]}{' …' if len(only_n) > 12 else ''}")

    payload = {
        "base_dir": base_dir, "new_dir": new_dir,
        "common": len(common), "only_base": only_b, "only_new": only_n,
        "base_metrics": A, "new_metrics": B,
        "delta": {k: round(B[k] - A[k], 4) for k in _METRIC_KEYS},
        "flips": flips,
    }
    if json_out:
        Path(json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] 明细已写入 {json_out}")

    # 退出码：出现「回归」返回 1，便于 CI 拦截
    return 1 if regressed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="评测 A/B 对比")
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--show-flips", action="store_true", help="打印翻转用例的 judge 理由")
    ap.add_argument("--json", default=None, help="把对比明细写入 json")
    args = ap.parse_args()
    sys.exit(compare(args.base, args.new, args.show_flips, args.json))


if __name__ == "__main__":
    main()
