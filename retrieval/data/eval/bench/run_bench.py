"""
轻量可复现基准（bench）：一条命令评估 RAG 问答服务的 速度 / 准确率 / 真实性。

定位：与正式评测（data/eval/run_eval.py，L2 judge + L3 人工抽检）互补的轻量冒烟基准，
不追求评测深度，追求「5-10 分钟跑完、报告可直接对比、环境信息齐全可复现」。

用法（需先启动查询服务，默认 8001）：
    .venv/Scripts/python data/eval/bench/run_bench.py                 # 默认 repeat=2
    .venv/Scripts/python data/eval/bench/run_bench.py --repeat 3      # 多轮取稳定值
    .venv/Scripts/python data/eval/bench/run_bench.py --base http://127.0.0.1:8001

指标定义：
- 响应速度：每条 /query 非流式墙钟延迟（ms），汇总 mean / P50 / P95（规划类含天气/路线外部 API，波动属正常）
- 准确率：judge 按金标 golden_facts 评定「被答案支持的要点比例」fact_hit_rate ∈ [0,1]（无金标用例不计）
- 真实性：judge 判答案是否编造具体事实（fabricated），真实性 = 非编造用例占比
- 通过判定：answer/plan → fact_hit_rate>=0.5 且未编造；refuse → 未编造即过（对齐 run_eval 负类口径
  「无资料不编造」，模板化/说明性回复不算失败，refused 字段仅作形态展示）；clarify → 澄清反问即过；
  chitchat → 未编造即过（域外联网作答属设计行为）

可复现保障：用例 id 名单入库（bench_cases.json，金标单一来源 eval_cases.json）；
报告头记录 时间 / git commit / 服务地址 / judge 模型 / repeat 参数；results.json 留全量原始明细。
"""
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # data/eval/bench
EVAL_DIR = BASE_DIR.parent                           # data/eval
PROJECT_ROOT = EVAL_DIR.parents[1]                   # 项目根
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.shared.config.lm_config import lm_config    # noqa: E402
from app.shared.model.lm_utils import get_llm_client  # noqa: E402

PASS_HIT_THRESHOLD = 0.5        # answer/plan 类通过线：金标要点命中率
JUDGE_RETRY = 1                 # judge 解析失败重试次数
QUERY_TIMEOUT_SECONDS = 300     # 规划类含外部 API，放宽


# ---------------------------------------------------------------- 用例加载
def load_cases(bench_file: Path):
    """按 bench_cases.json 的 include_ids 从正式评测集取用例（金标单一大来源）。"""
    if not bench_file.exists():
        raise SystemExit(f"[bench] 用例名单文件不存在：{bench_file}")
    bench = json.loads(bench_file.read_text(encoding="utf-8"))
    ids = bench["include_ids"]
    if not ids:
        raise SystemExit(f"[bench] 用例名单为空（include_ids 无条目）：{bench_file}")
    all_cases = json.loads((EVAL_DIR.parent / "eval_cases.json").read_text(encoding="utf-8"))["cases"]
    by_id = {c["id"]: c for c in all_cases}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"[bench] 评测集缺少用例：{missing}，请核对 bench_cases.json")
    return [by_id[i] for i in ids]


# ---------------------------------------------------------------- 问答调用
def ask_question(base: str, query: str, session_id: str) -> dict:
    """POST /query 非流式，返回 {latency_ms, answer, domain, done_count}，失败抛异常。"""
    payload = json.dumps({"query": query, "session_id": session_id, "is_stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/query", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - start) * 1000)
    return {
        "latency_ms": latency_ms,
        "answer": data.get("answer", "") or "",
        "domain": data.get("domain", ""),
        "done_count": len(data.get("done_list", []) or []),
    }


# ---------------------------------------------------------------- judge 评估
JUDGE_SYSTEM = (
    "你是 RAG 问答质量评审员。根据【问题】【金标要点】【系统答案】输出 JSON 评审结果，"
    "只输出 JSON，不要多余文字。字段：\n"
    '1. "fact_hit_rate"：金标要点中被答案正确支持的比率，0.0~1.0 一位小数；金标为空则填 null\n'
    '2. "fabricated"：答案是否编造了无中生有的具体事实（编造书目/演播者/景点/价格/班次/数据等），true/false；'
    "答案明确表示无法确认/无资料/建议核实不算编造\n"
    '3. "refused"：答案是否为拒答或澄清反问（明确表示无法回答、无对应资料、请用户补充信息），true/false\n'
    '4. "reason"：一句话依据'
)


def _parse_json_loose(text: str):
    """容错解析 judge 输出：剥 ```json 包裹后抽第一个 {...}。"""
    s = re.sub(r"```(?:json)?", "", text or "").strip().strip("`")
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError(f"judge 输出无 JSON：{text[:120]}")
    return json.loads(m.group(0))


def judge_answer(query: str, golden_facts: list, answer: str) -> dict | None:
    """一次 LLM 调用同时产出 准确率(fact_hit_rate) 与 真实性(fabricated/refused)；失败返回 None。"""
    client = get_llm_client(json_mode=True)
    user_msg = (
        f"【问题】{query}\n"
        f"【金标要点】{'；'.join(golden_facts) if golden_facts else '（无，fact_hit_rate 填 null）'}\n"
        f"【系统答案】\n{answer[:3000]}"
    )
    for _ in range(JUDGE_RETRY + 1):
        try:
            resp = client.invoke(
                [("system", JUDGE_SYSTEM), ("user", user_msg)]
            )
            result = _parse_json_loose(resp.content)
            hit = result.get("fact_hit_rate")
            return {
                "fact_hit_rate": None if hit is None else max(0.0, min(1.0, float(hit))),
                "fabricated": bool(result.get("fabricated", False)),
                "refused": bool(result.get("refused", False)),
                "reason": str(result.get("reason", ""))[:200],
            }
        except Exception as e:  # noqa: BLE001 - judge 失败降级，不阻断跑批
            print(f"    [judge 重试] {e}")
    return None


def judge_pass(case: dict, judge: dict | None) -> bool | None:
    """按 expect_type 判定用例通过；judge 失败返回 None（不计入通过率分母）。"""
    if judge is None:
        return None
    expect = case.get("expect_type", "answer")
    if judge["fabricated"]:
        return False
    if expect in ("answer", "plan"):
        return judge["fact_hit_rate"] is not None and judge["fact_hit_rate"] >= PASS_HIT_THRESHOLD
    if expect == "refuse":
        # 对齐 run_eval 负类口径（产品决策「无资料不编造」）：模板化/说明性回复不算失败，
        # 只要不编造具体事实即过；是否明确拒答用 judge.refused 字段在明细中展示形态。
        return True
    if expect == "clarify":
        return judge["refused"] or judge["fact_hit_rate"] is None  # 澄清反问即过
    if expect == "chitchat":
        return True  # 域外/闲聊：不编造即过
    return False


# ---------------------------------------------------------------- 统计与报告
def pct(values, q):
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def judge_model_label() -> str:
    model = lm_config.zhipu_llm_model if lm_config.llm_provider == "zhipu" else lm_config.llm_model
    return f"{model}({lm_config.llm_provider})"


def build_report(out_dir: Path, rounds: list, args) -> Path:
    """rounds: [{round, items:[{case, latency, judge, pass, error}]}]，汇总写 report.md + results.json。"""
    all_items = [it for r in rounds for it in r["items"]]
    latencies = [it["latency"] for it in all_items if it["latency"] is not None]
    hits = [it["judge"]["fact_hit_rate"] for it in all_items
            if it["judge"] and it["judge"]["fact_hit_rate"] is not None]
    judged = [it for it in all_items if it["judge"] is not None]
    fabricated_cnt = sum(1 for it in judged if it["judge"]["fabricated"])
    passes = [it["pass"] for it in all_items if it["pass"] is not None]
    pass_cnt = sum(1 for p in passes if p)

    # 分域延迟
    dom_lat = {}
    for it in all_items:
        if it["latency"] is not None:
            dom_lat.setdefault(it["case"]["domain"], []).append(it["latency"])

    last = rounds[-1]
    lines = [
        "# RAG 轻量基准报告（bench）",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 服务：{args.base} | repeat：{len(rounds)} | 用例：{len(rounds[0]['items'])} 条/轮",
        f"- 环境：git={git_commit()} | judge模型={judge_model_label()} | 通过线=hit>={PASS_HIT_THRESHOLD}",
        "",
        "## 总评（全部轮次汇总）",
        "| 指标 | 值 |",
        "|---|---|",
        (f"| 响应延迟 mean / P50 / P95 / max (ms) | "
         f"{statistics.mean(latencies):.0f} / {pct(latencies, 0.5)} / {pct(latencies, 0.95)} / {max(latencies)} |"
         if latencies else "| 响应延迟 | 无数据（无成功请求） |"),
        f"| 准确率 mean(fact_hit_rate) | {statistics.mean(hits):.3f} |" if hits else "| 准确率 | 无金标判定 |",
        f"| 用例通过率 | {pass_cnt}/{len(passes)}（{pass_cnt / len(passes) * 100:.1f}%） |" if passes else "| 通过率 | 无 |",
        f"| 真实率（非编造占比） | {len(judged) - fabricated_cnt}/{len(judged)}"
        f"（{(len(judged) - fabricated_cnt) / len(judged) * 100:.1f}%） |" if judged else "| 真实率 | 无 |",
        "",
        "## 分域延迟 (ms)",
        "| 领域 | n | mean | P95 |",
        "|---|---|---|---|",
    ]
    for dom, lats in sorted(dom_lat.items()):
        lines.append(f"| {dom} | {len(lats)} | {statistics.mean(lats):.0f} | {pct(lats, 0.95)} |")

    lines += ["", f"## 明细（第 {last['round']} 轮）",
              "| id | 期望 | 实际域 | 延迟ms | hit | 编造 | 通过 | 说明 |", "|---|---|---|---|---|---|---|---|"]
    for it in last["items"]:
        j = it["judge"]
        hit_s = "-" if not j or j["fact_hit_rate"] is None else f"{j['fact_hit_rate']:.1f}"
        fab_s = "-" if not j else ("是" if j["fabricated"] else "否")
        pass_s = {True: "✅", False: "❌", None: "⚠️"}[it["pass"]]
        reason = (j["reason"] if j else it.get("error") or "judge 失败")[:60]
        lines.append(
            f"| {it['case']['id']} | {it['case']['expect_type']} | {it.get('domain') or '-'} "
            f"| {it['latency'] if it['latency'] is not None else '-'} | {hit_s} | {fab_s} | {pass_s} | {reason} |"
        )
    lines += ["", "> 注：judge 为 LLM 初筛口径，存在误判可能；正式结论请以 run_eval.py 全量评测 + 人工抽检为准。",
              "> 规划类延迟含和风天气/高德外部 API，天然波动。", ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"args": vars(args), "rounds": rounds}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return report_path


# ---------------------------------------------------------------- 主流程
def main():
    import argparse

    parser = argparse.ArgumentParser(description="轻量可复现基准：速度/准确率/真实性")
    parser.add_argument("--base", default="http://127.0.0.1:8001", help="查询服务地址")
    parser.add_argument("--repeat", type=int, default=2, help="重复轮数（默认 2）")
    parser.add_argument("--cases", default=str(BASE_DIR / "bench_cases.json"), help="用例名单文件")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error(f"--repeat 必须 >= 1，当前：{args.repeat}")

    cases = load_cases(Path(args.cases))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rounds = []
    print(f"[bench] 用例 {len(cases)} 条 × {args.repeat} 轮 → {args.base}")
    for r in range(1, args.repeat + 1):
        items = []
        print(f"--- 第 {r}/{args.repeat} 轮 ---")
        for case in cases:
            session_id = f"bench-{ts}-r{r}-{case['id']}"
            item = {"case": {"id": case["id"], "expect_type": case["expect_type"],
                             "domain": case["domain"], "query": case["query"]},
                    "latency": None, "judge": None, "pass": None, "error": ""}
            try:
                result = ask_question(args.base, case["query"], session_id)
                item["latency"] = result["latency_ms"]
                item["domain"] = result["domain"]
                item["answer"] = result["answer"]
                item["done_count"] = result["done_count"]
                print(f"  {case['id']:<10} {result['latency_ms']:>6}ms domain={result['domain']:<10} "
                      f"answer={len(result['answer'])}字")
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError, UnicodeDecodeError) as e:
                # JSON/编码两类：服务返回 200 但 body 非 JSON（如网关 HTML 错误页）时降级为单条失败
                item["error"] = f"请求失败：{e}"
                print(f"  {case['id']:<10} ❌ {item['error']}")
            if item.get("answer") or not item["error"]:
                item["judge"] = judge_answer(case["query"], case.get("golden_facts") or [],
                                             item.get("answer", ""))
                item["pass"] = judge_pass(case, item["judge"])
            items.append(item)
        rounds.append({"round": r, "items": items})
        # 服务不可达等场景：第一轮全部失败则早退，避免空跑 judge 并生成无意义报告
        if r == 1 and items and all(it["error"] for it in items):
            raise SystemExit(f"[bench] 全部用例请求失败，请确认查询服务已启动：{args.base}")

    out_dir = PROJECT_ROOT / "output" / "bench" / ts
    report = build_report(out_dir, rounds, args)
    print(f"\n[bench] 报告：{report}")


if __name__ == "__main__":
    main()
