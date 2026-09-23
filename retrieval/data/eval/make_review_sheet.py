#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3 人工抽检辅助：从评测输出生成「复核单」，并把人工结论回填为修正指标。

背景（docs/回归策略.md）：
    L2（LLM-as-Judge）**双向失守率约 17.5%**（假阴 = 模型其实答对却判错；
    假阳 = 答得像样却漏判幻觉），因此终值**必须**由 L3 人工抽检 ≥25% 复核。
    此前这道工序**无工具支撑**，全靠手工挑样本、手工记账。

本脚本把它变成两步行之可复现的工序：
    1) build：抽检 = 全部失败项 + 随机补足至 ≥ratio（同 seed 可复现）；
    2) apply：回填人工结论 → 输出修正后的 Acc/Precision/Recall/F1（与原判定对照）。

用法：
    # 生成复核单（写 <out>/review_sheet.md 与 <out>/review_sample.json）
    python data/eval/make_review_sheet.py --out output/eval/<ts> [--ratio 0.25] [--seed 42]

    # 回填人工结论（写 <out>/corrected_metrics.json）
    python data/eval/make_review_sheet.py --apply output/eval/<ts> \
        --verdicts output/eval/<ts>/review_result.json
    # review_result.json 形如：{"plan-001": "wrong", "neg-003": "correct"}
    #   人工判定：correct = 模型答对（judge 原判可信）；wrong = 模型答错（judge 判决有误）
    #   未给出结论的用例，沿用 judge 原判（不修改）。

口径（与 run_eval.aggregate_answer_metrics 严格一致）：
    正类 pos = expect_type ∈ {answer, plan}；accept=True 表示该答案整体可接受。
    pos & accept → TP；pos & !accept → FN；!pos & accept → TN；!pos & !accept → FP。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ANSWER_CAP = 1800  # 复核单里答案的展示上限（超长截断，避免成文几万行）


def _load_metrics(out_dir: str) -> dict:
    p = Path(out_dir) / "metrics.json"
    if not p.exists():
        raise SystemExit(f"[错误] 未找到 {p}（请先跑评测产出该目录）")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_cases(cases_path: str) -> dict:
    p = Path(cases_path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("cases", [])
    return {r.get("id"): r for r in rows if r.get("id")}


def _pos(rec: dict) -> bool:
    return rec.get("expect_type") in ("answer", "plan")


def _aggregate(recs: list, verdicts: dict | None) -> dict:
    """按 run_eval 口径重算四格；verdicts 给出的用例用人工结论覆盖 judge 判定。"""
    verdicts = verdicts or {}
    tp = fp = fn = tn = unresolved = 0
    flipped = 0
    for r in recs:
        aj = r.get("answer_judge") or {}
        acc = aj.get("accept")
        vid = r.get("id")
        if vid in verdicts:
            human = str(verdicts[vid]).strip().lower()
            new_acc = human in ("correct", "true", "1", "对", "pass", "ok")
            if new_acc is not acc:
                flipped += 1
            acc = new_acc
        if acc is None:
            unresolved += 1
            continue
        p = _pos(r)
        if p and acc:
            tp += 1
        elif p and not acc:
            fn += 1
        elif (not p) and acc:
            tn += 1
        else:
            fp += 1
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unresolved": unresolved,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "human_flipped": flipped,
    }


def _pick(recs: list, ratio: float, seed: int) -> tuple[list, list, int]:
    """返回 (picked, fails, need)。picked 保持原用例顺序。"""
    fails = [r for r in recs if (r.get("answer_judge") or {}).get("accept") is False]
    passes = [r for r in recs if (r.get("answer_judge") or {}).get("accept") is True]
    need = max(1, math.ceil(ratio * len(recs))) if recs else 0
    need_pass = max(0, need - len(fails))
    rng = random.Random(seed)
    sampled = rng.sample(passes, min(need_pass, len(passes))) if need_pass else []
    order = {r.get("id"): i for i, r in enumerate(recs)}
    picked = sorted(fails + sampled, key=lambda r: order.get(r.get("id"), 10 ** 9))
    return picked, fails, need


def _fmt_answer(text: str) -> str:
    t = (text or "").strip()
    if len(t) > _ANSWER_CAP:
        t = t[:_ANSWER_CAP] + f"\n…（已截断，完整 {len(text)} 字）"
    return t


def build(out_dir: str, ratio: float, seed: int, cases_path: str | None) -> None:
    d = _load_metrics(out_dir)
    recs = d.get("cases", [])
    if not recs:
        raise SystemExit("[错误] metrics.json 内无 cases，无法生成复核单")
    cases_path = cases_path or (d.get("args") or {}).get("cases") or str(ROOT / "data" / "eval_cases.json")
    golden = _load_cases(cases_path)

    picked, fails, need = _pick(recs, ratio, seed)
    base = _aggregate(recs, None)

    lines: list[str] = []
    lines.append("# L3 人工抽检复核单")
    lines.append("")
    lines.append(f"- 评测目录：`{out_dir}`")
    lines.append(f"- 用例来源：`{cases_path}`")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 全量样本：**{len(recs)}** 条；judge 判失败：**{len(fails)}** 条；"
                 f"抽检目标 ≥{ratio:.0%}（{need} 条）；**实际抽检 {len(picked)} 条**"
                 f"（{len(picked) / max(len(recs), 1):.1%}）")
    lines.append(f"- 抽样随机种子：`{seed}`（同目录同种子可复现同一组样本）")
    lines.append(f"- judge 原始指标：Acc {base['accuracy']} / P {base['precision']} / "
                 f"R {base['recall']} / F1 {base['f1']}"
                 f"（TP{base['tp']} FP{base['fp']} FN{base['fn']} TN{base['tn']}）")
    lines.append("")
    lines.append("> **复核口径**：L2 judge 双向都会失守（约 17.5%）。请在每条下填写被抽中用例的人工结论——"
                 "**模型答对填 `correct`，模型答错填 `wrong`**，写入同目录 `review_result.json` 后用 "
                 "`--apply` 生成修正指标。")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(picked, 1):
        aj = r.get("answer_judge") or {}
        judge_txt = {True: "✔︎ 通过", False: "✘ 失败", None: "— 未判"}.get(aj.get("accept"), "— 未判")
        lines.append(f"### {i}. `{r.get('id')}` · 期望={r.get('expect_type')} · 路由={r.get('domain_route')} · "
                     f"judge={judge_txt} · **人工判定：▢correct ▢wrong**")
        lines.append("")
        lines.append(f"**问题**：{r.get('query', '')}")
        lines.append("")
        a = _fmt_answer(r.get("answer", ""))
        lines.append("**模型答案**：")
        lines.append("")
        for al in a.splitlines() or [""]:
            lines.append(f"> {al}")
        lines.append("")
        g = golden.get(r.get("id")) or {}
        facts = g.get("golden_facts") or []
        covered = aj.get("covered") or []
        if facts:
            lines.append("**金标要点**（judge covered 标记）：")
            for j, f in enumerate(facts):
                mark = "✔︎" if (j < len(covered) and covered[j]) else "✘"
                lines.append(f"- {mark} {f}")
            lines.append("")
        if g.get("golden_file_titles"):
            lines.append(f"**金标文件**：{'、'.join(g['golden_file_titles'])}")
            lines.append("")
        jr = aj.get("judge_raw") or {}
        if jr.get("reason") or aj.get("reason"):
            lines.append(f"**judge 理由**：{jr.get('reason') or aj.get('reason')}")
            lines.append("")
        if r.get("error"):
            lines.append(f"**执行错误**：`{r['error']}`")
            lines.append("")
        lines.append("---")
        lines.append("")

    sheet = Path(out_dir) / "review_sheet.md"
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample = {
        "out_dir": out_dir,
        "cases_path": cases_path,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ratio": ratio, "seed": seed,
        "total": len(recs), "n_fail": len(fails), "need": need,
        "picked": [r.get("id") for r in picked],
        "picked_fail": [r.get("id") for r in fails],
        "judge_metrics": base,
    }
    (Path(out_dir) / "review_sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 复核单已生成：{sheet}")
    print(f"     抽检 {len(picked)}/{len(recs)} = {len(picked) / max(len(recs), 1):.1%}"
          f"（其中 FAIL {len(fails)} 条必含）")
    print(f"[OK] 抽样留痕：{Path(out_dir) / 'review_sample.json'}")


def apply_verdicts(out_dir: str, verdicts_path: str) -> None:
    d = _load_metrics(out_dir)
    recs = d.get("cases", [])
    vp = Path(verdicts_path)
    if not vp.exists():
        raise SystemExit(f"[错误] 未找到人工结论文件 {vp}")
    verdicts = json.loads(vp.read_text(encoding="utf-8"))
    unknown = [k for k in verdicts if k not in {r.get("id") for r in recs}]
    if unknown:
        print(f"[警告] {len(unknown)} 个结论 id 不在本次评测中：{unknown[:6]}")

    before = _aggregate(recs, None)
    after = _aggregate(recs, verdicts)

    report = {
        "out_dir": out_dir,
        "verdicts_file": str(vp),
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_verdicts": len(verdicts),
        "judge_baseline": before,
        "corrected": after,
        "unknown_ids": unknown,
    }
    outp = Path(out_dir) / "corrected_metrics.json"
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== judge 原判 vs 人工修正 ==")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:>10}: {before[k]}  ->  {after[k]}")
    print(f"  四格: TP{before['tp']} FP{before['fp']} FN{before['fn']} TN{before['tn']}"
          f"  ->  TP{after['tp']} FP{after['fp']} FN{after['fn']} TN{after['tn']}")
    print(f"  人工推翻 judge 的条数：{after['human_flipped']}")
    print(f"[OK] 修正指标已写入：{outp}")


def main() -> None:
    ap = argparse.ArgumentParser(description="L3 人工抽检辅助（复核单生成 / 结论回填）")
    ap.add_argument("--out", required=True, help="评测输出目录（含 metrics.json）")
    ap.add_argument("--ratio", type=float, default=0.25, help="抽检比例下限（默认 0.25）")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    ap.add_argument("--cases", default=None, help="评测用例文件（默认取 metrics.json 的 args.cases）")
    ap.add_argument("--apply", action="store_true", help="回填人工结论模式")
    ap.add_argument("--verdicts", default=None, help="--apply 时的人工结论 json 路径")
    args = ap.parse_args()

    if args.apply:
        if not args.verdicts:
            raise SystemExit("[错误] --apply 需同时给出 --verdicts <json>")
        apply_verdicts(args.out, args.verdicts)
    else:
        build(args.out, args.ratio, args.seed, args.cases)


if __name__ == "__main__":
    main()
