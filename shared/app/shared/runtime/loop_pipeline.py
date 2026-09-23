# -*- coding: utf-8 -*-
"""Loop Engineering D3：自动优化管道骨架。

闭环：提出优化提案 → 触发评测回归（门禁：Accuracy 不低于基线）→
      审批 → 灰度/回滚 → 审计。

设计原则（对齐《企业 Agent 落地十二大工程框架》Loop Engineering）：
- "错误归因先于优化"：提案须关联 12 类归因（attribution_id / attr_class）。
- 任何改动须经：评测回归 + 可审批 + 可灰度 + 可回滚 + 可审计。
- 本模块为**骨架**：不自动改业务代码，只管理提案生命周期与回归门禁；
  真正落码由人工/后续 agent 执行，但每次落码都需经本管道留痕。

存储：logs/optimization_proposals.jsonl（追加 + 全量重写更新，便于审计回溯）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from app.shared.runtime.loop_attribution import ATTRIBUTION_CLASSES

_PROJ = Path(__file__).resolve().parents[3]
_PROPOSAL_FILE = _PROJ / "logs" / "optimization_proposals.jsonl"
_EVAL_SCRIPT = _PROJ / "data" / "eval" / "run_eval.py"
_DEFAULT_CASES = _PROJ / "data" / "eval_cases.json"

PROPOSAL_STATUS = (
    "proposed",          # 已提出，未回归
    "regression_passed", # 回归达标，待审批
    "rejected",          # 回归未达标，驳回
    "approved",          # 已审批
    "gray_released",     # 已灰度发布
    "rolled_back",       # 已回滚
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _ensure_file() -> None:
    _PROPOSAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _PROPOSAL_FILE.exists():
        _PROPOSAL_FILE.write_text("", encoding="utf-8")


def _load_all() -> list:
    _ensure_file()
    out = []
    for line in _PROPOSAL_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _save_all(rows: list) -> None:
    _ensure_file()
    with open(_PROPOSAL_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def create_proposal(
    title: str,
    target: str,
    change_summary: str,
    attr_class: str,
    baseline_acc: float,
    attribution_id: str = "",
    proposed_by: str = "human",
    note: str = "",
) -> dict:
    """创建一条优化提案。attr_class 须属于 12 类归因。"""
    if attr_class not in ATTRIBUTION_CLASSES:
        attr_class = "data"
    rows = _load_all()
    existing = {r.get("proposal_id") for r in rows}
    base = time.strftime("%Y%m%d-%H%M%S") + "-" + f"{int((time.time() % 1) * 1000):03d}"
    pid = "opt-" + base
    suffix = 1
    while pid in existing:  # 同一毫秒内连续创建时保证唯一（原实现可碰撞）
        suffix += 1
        pid = f"opt-{base}-{suffix}"
    rec = {
        "proposal_id": pid,
        "title": title,
        "target": target,
        "change_summary": change_summary,
        "attr_class": attr_class,
        "attribution_id": attribution_id,
        "baseline_acc": float(baseline_acc),
        "proposed_by": proposed_by,
        "status": "proposed",
        "regression_acc": None,
        "regression_report": "",
        "note": note,
        "created_at": _now(),
        "updated_at": _now(),
        "audit": [{"ts": _now(), "action": "create", "by": proposed_by}],
    }
    rows.append(rec)
    _save_all(rows)
    return rec


def get_proposal(pid: str) -> Optional[dict]:
    for r in _load_all():
        if r.get("proposal_id") == pid:
            return r
    return None


def list_proposals(status_filter: Optional[str] = None) -> list:
    rows = _load_all()
    if status_filter:
        rows = [r for r in rows if r.get("status") == status_filter]
    return rows


def _parse_acc_from_metrics(metrics_path: Path) -> Optional[float]:
    """从 run_eval 输出的 metrics.json 解析全域 Accuracy。"""
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(data["summary"]["answer_global"]["accuracy"])
    except Exception:
        return None


def run_regression(
    pid: str,
    cases_path: Optional[str] = None,
    ids: Optional[str] = None,
    out_dir: Optional[str] = None,
    skip_run: bool = False,
    mock_acc: Optional[float] = None,
) -> dict:
    """对提案触发评测回归，并以 Accuracy 为门禁更新状态。

    - skip_run=True 时跳过真实评测，直接用 mock_acc（骨架/测试用）。
    - 否则调用 data/eval/run_eval.py（--ids 可定向回归），解析 metrics.json。
    - 门禁：regression_acc >= baseline_acc → regression_passed；否则 rejected。
    返回更新后的提案记录。
    """
    rec = get_proposal(pid)
    if rec is None:
        raise ValueError(f"proposal not found: {pid}")

    acc: Optional[float] = None
    report = ""
    if skip_run:
        acc = mock_acc
        report = "skipped real eval (skeleton)"
    else:
        cases = cases_path or str(_DEFAULT_CASES)
        out = out_dir or str(_PROJ / "output" / "eval" / f"regr-{pid}")
        cmd = [sys.executable, str(_EVAL_SCRIPT), "--cases", cases, "--out", out, "--no-ablation"]
        if ids:
            cmd += ["--ids", ids]
        try:
            subprocess.run(cmd, cwd=str(_PROJ), check=True, capture_output=True, text=True, timeout=3600)
            acc = _parse_acc_from_metrics(Path(out) / "metrics.json")
            report = f"eval ok, metrics={Path(out)/'metrics.json'}"
        except Exception as e:  # noqa: BLE001
            report = f"eval failed: {e!r}"

    rows = _load_all()
    for r in rows:
        if r.get("proposal_id") == pid:
            r["regression_acc"] = acc
            r["regression_report"] = report
            if acc is None:
                r["status"] = "rejected"
                r["regression_report"] += " | acc 解析失败，按驳回处理"
            elif acc >= r["baseline_acc"]:
                r["status"] = "regression_passed"
            else:
                r["status"] = "rejected"
            r["updated_at"] = _now()
            r["audit"].append({
                "ts": _now(), "action": "regression",
                "acc": acc, "gate": r["status"],
            })
            break
    _save_all(rows)
    return get_proposal(pid)


def _transition(pid: str, target_status: str, by: str, reason: str = "") -> dict:
    rec = get_proposal(pid)
    if rec is None:
        raise ValueError(f"proposal not found: {pid}")
    rows = _load_all()
    for r in rows:
        if r.get("proposal_id") == pid:
            r["status"] = target_status
            r["updated_at"] = _now()
            r["audit"].append({
                "ts": _now(), "action": target_status, "by": by, "reason": reason,
            })
            break
    _save_all(rows)
    return get_proposal(pid)


def approve_proposal(pid: str, by: str = "human") -> dict:
    """审批通过（须先 regression_passed）。"""
    rec = get_proposal(pid)
    if rec is None:
        raise ValueError(f"proposal not found: {pid}")
    if rec["status"] != "regression_passed":
        raise ValueError(f"cannot approve in status {rec['status']}")
    return _transition(pid, "approved", by)


def gray_release_proposal(pid: str, by: str = "human") -> dict:
    """灰度发布（须先 approved）。"""
    rec = get_proposal(pid)
    if rec is None:
        raise ValueError(f"proposal not found: {pid}")
    if rec["status"] != "approved":
        raise ValueError(f"cannot gray_release in status {rec['status']}")
    return _transition(pid, "gray_released", by)


def rollback_proposal(pid: str, by: str = "human", reason: str = "") -> dict:
    """回滚（从 approved / gray_released 回到 rolled_back）。"""
    rec = get_proposal(pid)
    if rec is None:
        raise ValueError(f"proposal not found: {pid}")
    if rec["status"] not in ("approved", "gray_released"):
        raise ValueError(f"cannot rollback in status {rec['status']}")
    return _transition(pid, "rolled_back", by, reason)
