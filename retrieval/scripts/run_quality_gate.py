#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""质量门禁（CI-ready）：语法编译 → 架构依赖 lint → 离线单测。

设计目标（对齐《分布式开发拆分计划》阶段 A 的契约守护）：
  一条命令守护「代码可编译 / 依赖方向合规 / 行为不回归」三道基线，
  可直接接入 CI 或 pre-push；发现任一失败即返回非零退出码。

用法：
    python scripts/run_quality_gate.py            # 全量：编译 + lint + 全部 test_*.py
    python scripts/run_quality_gate.py --fast     # 快跑：仅纯离线单测（不依赖 Mongo/Milvus/Redis）
    python scripts/run_quality_gate.py --lint-only
    python scripts/run_quality_gate.py --timeout 600

注意（重要）：
    默认「全量」模式会跑**需要真实 LLM 的测试**（如 test_tool_weather.py / test_trip_plan*.py）。
    若此时正在跑 `data/eval/run_eval.py`，两者会**争抢同一 LLM 配额 → 触发 429**，
    表现为测试失败（假阳性，非代码回归）。**评测进行中请改用 `--fast`。**
    如需永久排除个别 LLM 依赖测试，加进 --exclude 或改 OFFLINE_TESTS 列表即可。

退出码：0 全绿；1 存在失败。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 纯离线单测（零外部服务）：--fast 只跑这些，用于本地快验/未启服务时
OFFLINE_TESTS = [
    "test_context_compressor.py",
    "test_llm_reliability.py",
    "test_loop_engineering.py",
    "test_intent_route_llm.py",
    "test_eval_plan_structure.py",
    "test_review_sheet.py",
    "test_compare_eval.py",
    "test_framework_integration.py",
    "test_stay_food_distance.py",
    "test_mobile_gateway.py",
    "test_mobile_gateway_stream.py",
    "test_mobile_gateway_auth.py",
    "test_mobile_auth_rate_limit.py",
    "test_trip_route.py",
    "test_eval_warnings.py",
    "test_generic_place.py",
    "test_eval_equivalence.py",
    "test_plan_slot_gate.py",
    "test_trip_plan_bridge.py",
    "test_trip_plan_review.py",
]

_COUNT_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
# 兼容「27 通过 / 0 失败」这类表述
_COUNT_RE2 = re.compile(r"(\d+)\s*通过\s*/\s*(\d+)\s*失败")


def _run(cmd: list[str], timeout: int | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"[超时] 超过 {timeout}s 未结束"
    except Exception as e:  # noqa: BLE001
        return 125, f"[执行异常] {e!r}"


def step_compile() -> tuple[bool, str]:
    rc, out = _run([PY, "-m", "compileall", "-q", "app", "test"], timeout=300)
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-3:]
    return rc == 0, "；".join(tail) if tail else "全部模块编译通过"


def step_lint() -> tuple[bool, str]:
    rc, out = _run([PY, "scripts/check_import_direction.py"], timeout=120)
    info = next((ln for ln in out.splitlines() if ln.startswith("[")), out.strip()[:160])
    return rc == 0, info


def _discover_tests(fast: bool) -> list[str]:
    if fast:
        return [t for t in OFFLINE_TESTS if (ROOT / "test" / t).exists()]
    return sorted(p.name for p in (ROOT / "test").glob("test_*.py"))


def step_tests(names: list[str], timeout: int) -> tuple[bool, list[tuple[str, bool, str]]]:
    rows: list[tuple[str, bool, str]] = []
    for name in names:
        rc, out = _run([PY, f"test/{name}"], timeout=timeout)
        ok = rc == 0
        # 仅从「结果」汇总行提取 P/Q（避免命中日志里的端口号等无关数字）
        detail = ""
        for line in out.splitlines():
            if "单测结果" in line or "结果" in line or "通过" in line:
                m2 = _COUNT_RE2.search(line)
                if m2:
                    # 「P 通过 / Q 失败」→ 统一成 P/(P+Q)
                    detail = f"{m2.group(1)}/{int(m2.group(1)) + int(m2.group(2))}"
                    continue
                m = _COUNT_RE.search(line)
                if m:
                    detail = f"{m.group(1)}/{m.group(2)}"
        if not detail:
            detail = "OK" if ok else f"exit={rc}"
        rows.append((name, ok, detail))
    return all(r[1] for r in rows), rows


def main() -> None:
    ap = argparse.ArgumentParser(description="质量门禁：编译 + 依赖 lint + 单测")
    ap.add_argument("--fast", action="store_true", help="仅跑纯离线单测")
    ap.add_argument("--lint-only", action="store_true", help="只跑编译 + 依赖 lint")
    ap.add_argument("--timeout", type=int, default=600, help="单个测试的超时秒数（默认 600）")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 64)
    print("质量门禁 · Quality Gate")
    print("=" * 64)

    results: list[tuple[str, bool, str]] = []

    ok, info = step_compile()
    results.append(("① 语法编译 compileall", ok, info))
    print(f"[{'OK' if ok else 'FAIL'}] ① 语法编译：{info}")

    ok, info = step_lint()
    results.append(("② 依赖方向 lint", ok, info))
    print(f"[{'OK' if ok else 'FAIL'}] ② 依赖方向：{info}")

    if not args.lint_only:
        names = _discover_tests(args.fast)
        print(f"[..] ③ 单测：开始（{len(names)} 个{'·离线' if args.fast else ''}文件）")
        all_ok, rows = step_tests(names, args.timeout)
        for name, tok, detail in rows:
            print(f"     [{'OK' if tok else 'FAIL'}] {name:<34} {detail}")
        results.append((f"③ 单测（{len(names)} 个）", all_ok, "见上表"))

    elapsed = time.time() - t0
    print("-" * 64)
    failed = [n for n, ok, _ in results if not ok]
    for n, ok, info in results:
        print(f"  {'✅' if ok else '❌'} {n}")
    print(f"耗时 {elapsed:.1f}s")
    if failed:
        print(f"门禁未通过 ❌：{', '.join(failed)}")
        sys.exit(1)
    print("门禁全绿 ✅")
    sys.exit(0)


if __name__ == "__main__":
    main()
