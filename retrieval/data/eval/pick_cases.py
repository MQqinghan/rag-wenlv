"""
按改动影响面自动挑选回归用例，替代「每次都全量 28 题」。

设计背景：
    全量 28 题约 6.6 分钟 + 28 次 LLM judge 调用，日常每个任务都跑既费时又耗
    token 额度（额度紧张时 judge 会无响应，造成假 FN 抖动）。本脚本按「改动影响
    面 → 用例标签」的映射选出最小必跑集，全量改为里程碑才跑。

用法：
    # 改了意图路由（intent_route_service / 路由正则 / intent_route_v2.prompt）
    python data/eval/pick_cases.py --impact routing
    # 改了检索或重排
    python data/eval/pick_cases.py --impact retrieval
    # 改了答案 prompt（影响面最大，建议并上 core 基线）
    python data/eval/pick_cases.py --impact generation --core
    # 自定义标签（并集）
    python data/eval/pick_cases.py --tag web,guard
    # 只看清单不取 id
    python data/eval/pick_cases.py --impact core --list
    # 直接喂给 run_eval（bash）
    IDS=$(python data/eval/pick_cases.py --impact routing)
    python data/eval/run_eval.py --cases data/eval_cases.json --out output/eval/xxx --ids "$IDS"

标签含义见 data/eval_cases.json 的 meta.tag_schema。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "eval_cases.json"

# 改动影响面 → 对应标签集合（并集）
IMPACT_MAP: dict[str, tuple[str, ...]] = {
    "routing": ("routing",),
    "retrieval": ("retrieval",),
    "generation": ("generation",),
    "plan": ("plan",),
    "web": ("web",),
    "guard": ("guard",),
    "core": ("core",),
    "all": (),  # 空表示全选
}

# 单条用例平均耗时（秒），取自 output/eval/20260910_t7_full50 实测：698.2s / 50 条
AVG_SEC_PER_CASE = 14.0


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", data)


def pick(cases: list[dict], impact: str | None, tags: list[str], with_core: bool) -> list[dict]:
    if impact == "all":
        return list(cases)

    wanted: set[str] = set(tags)
    if impact:
        if impact not in IMPACT_MAP:
            raise SystemExit(f"未知 impact: {impact}，可选 {sorted(IMPACT_MAP)}")
        wanted |= set(IMPACT_MAP[impact])
    if with_core:
        wanted.add("core")
    if not wanted:
        raise SystemExit("未指定任何筛选条件（用 --impact 或 --tag）")

    return [c for c in cases if wanted & set(c.get("tags") or [])]


def main() -> None:
    ap = argparse.ArgumentParser(description="按改动影响面挑选回归用例")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--impact", default="", choices=sorted(IMPACT_MAP),
                    help="改动影响面；all 表示全量")
    ap.add_argument("--tag", default="", help="额外标签，逗号分隔（与 impact 取并集）")
    ap.add_argument("--core", action="store_true", help="强制并入 core 核心基线集")
    ap.add_argument("--list", action="store_true", help="打印清单而非输出 id 串")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    extra = [t.strip() for t in args.tag.split(",") if t.strip()]
    selected = pick(cases, args.impact or None, extra, args.core)

    if args.list:
        print(f"命中 {len(selected)} 条（共 {len(cases)} 条），预估 {len(selected) * AVG_SEC_PER_CASE / 60:.1f} 分钟")
        print("-" * 78)
        for c in selected:
            print(f"{c['id']:9s} [{'/'.join(c.get('tags') or [])}]")
            print(f"          {c['query'][:60]}")
        return

    # 默认输出逗号分隔 id，便于 shell 直接传给 run_eval --ids
    ids = ",".join(c["id"] for c in selected)
    print(ids)
    # 辅助信息走 stderr，不污染 id 串
    print(f"# {len(selected)}/{len(cases)} 条，预估 {len(selected) * AVG_SEC_PER_CASE / 60:.1f} 分钟",
          file=sys.stderr)


if __name__ == "__main__":
    main()
