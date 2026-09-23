# -*- coding: utf-8 -*-

"""

RAG 系统评测执行器（文旅 + 行程规划）



依据 docs/评估方案与报告.md：

- 逐条调用 unified_query_app（LangGraph 统一查询图，is_stream=False），

  从节点流式 update 中抓取各阶段真实中间产物（零额外检索成本）：

    embedding_chunks / hyde_embedding_chunks / keyword_chunks → rrf_chunks → reranked_docs

- 检索层指标（top5）：Precision / Recall / F1 / Hit，分别在

    B1 仅向量路（embedding_chunks[:5]）

    B2 多路+RRF（rrf_chunks[:5]）

    B3 全链路+Rerank（reranked_docs[:5]）

  三档计算 → 消融对照（量化 RRF / Rerank 增量）

- 问答层指标：按 expect_type 构建四格（TP/FP/FN/TN）→ Accuracy / Precision / Recall / F1。

  判定 = L1 规则硬校验 + L2 LLM-as-Judge（json_mode，逐条核对 golden_facts 与幻觉），分歧人工复核。



用法（需在项目 .venv 环境、服务在线）：

  python data/eval/run_eval.py --cases data/eval_cases.json --out output/eval/<ts>

  python data/eval/run_eval.py --cases data/eval_cases.json --limit 3   # 冒烟（不落盘 judge 也可 --no-judge）

  python data/eval/run_eval.py --ids plan-003,neg-005,neg-006,xr-009  # 定向回归（只跑指定用例）

"""

from __future__ import annotations



import argparse

import json

import os

import re

import sys

import time

import uuid

from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



# ---------- 常量 ----------

TOP_K = 5  # 检索层 top-k（与答案生成取 reranked_docs[:5] 一致）

SOURCE_MARKERS = ["参考资料", "内容来自", "资料：", "来源："]  # 答案中“带出处”的判据

REFUSE_MARKERS = ["无法", "不能", "不适合", "暂不", "没有找到", "未找到", "抱歉", "不好意思", "知识库", "建议您"]

JUDGE_TIMEOUT_S = 60

# plan 类结构门禁开关（补评测盲区）：金标为行程规划、但生成全程未进入行程链路

# （未判 is_plan、无行程地图、规划工具全空）时直接判不通过——judge 只看要点覆盖会漏判。

_PLAN_STRUCTURE_ENFORCE = True



# ---------- Loop 闭环（D1 错误归因，纯增量日志，不影响判定） ----------

from app.shared.runtime.loop_attribution import record_attribution, suggest_attr_class  # noqa: E402





# ============================================================

# 工具函数

# ============================================================



def doc_file_title(doc: dict) -> str:

    """从候选文档里取来源 file_title（兼容不同字段名/嵌套）。"""

    if not isinstance(doc, dict):

        return ""

    for key in ("file_title", "source_file", "filename", "book_name"):

        v = doc.get(key)

        if isinstance(v, str) and v.strip():

            return v.strip()

    meta = doc.get("meta")

    if isinstance(meta, dict):

        v = meta.get("file_title")

        if isinstance(v, str) and v.strip():

            return v.strip()

    return ""





def doc_text(doc: dict, limit: int = 500) -> str:

    """取候选文本（text/content 兼容），供 judge 核对原文。"""

    for key in ("text", "content", "page_content"):

        v = doc.get(key)

        if isinstance(v, str) and v.strip():

            v = re.sub(r"\s+", " ", v).strip()

            return v[:limit]

    return ""





def path_file_titles(docs, top: int = TOP_K) -> list[str]:

    """取 docs[:top] 的 file_title（去重保序）。"""

    seen: list[str] = []

    for d in (docs or [])[:top]:

        t = doc_file_title(d)

        if t and t not in seen:

            seen.append(t)

    return seen





def compute_ret_metrics(golden: set[str], hit_titles: list[str], top: int = TOP_K) -> dict:

    """单条检索层指标：候选=hit_titles[:top]，相关集=golden。"""

    a = hit_titles[:top]

    inter = [t for t in a if t in golden]

    p = len(inter) / len(a) if a else 0.0

    r = len(inter) / len(golden) if golden else 0.0

    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "hit": 1 if inter else 0}





def macro_mean(rows: list[dict], keys=("precision", "recall", "f1", "hit")) -> dict:

    if not rows:

        return {k: 0.0 for k in keys}

    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}





def micro_ret(rows: list[dict]) -> dict:

    """micro：合并混淆再算。hit 用命中 case 数/总 case 数近似 Acc。"""

    tp = sum(r["hit"] for r in rows)  # 检索层 Hit 口径

    return {"acc_micro": round(tp / len(rows), 4) if rows else 0.0}





# ============================================================

# L1 规则硬校验

# ============================================================



def _plan_structure_signals(state: dict) -> dict:

    """plan 类结构信号 —— 纯读 state，零生产改动。



    判据均取自行程链路的客观产物（不做语义猜测）：

      - is_plan      ：意图路由判为行程规划（intent_route_service 写的独立键）

      - has_map_data ：行程生成产出地图数据（map_data 非空，只有真跑行程才有）

      - plan_tools   ：规划分支特有工具命中（天气/路线/铁路/住宿餐饮/预算/交通）

    structured=True 当且仅当三者至少命中其一（行程链路的结构性必要条件）。

    """

    is_plan = bool(state.get("is_plan"))

    has_map = bool(state.get("map_data"))

    tool_keys = (

        "tool_weather",

        "tool_route",

        "tool_rail",

        "tool_stay_food",

        "tool_budget",

        "tool_transport",

    )

    hit = [k for k in tool_keys if state.get(k)]

    return {

        "is_plan": is_plan,

        "has_map_data": has_map,

        "plan_tools": hit,

        "structured": bool(is_plan or has_map or hit),

    }





def l1_check(case: dict, state: dict) -> dict:

    """L1 只产出硬信号，不做最终判决；返回给 judge/汇总使用。"""

    exp = case.get("expect_type", "answer")

    answer = (state.get("answer") or "").strip()

    reranked = state.get("reranked_docs") or []

    out = {

        "answer_empty": not answer,

        "has_source_marker": any(m in answer for m in SOURCE_MARKERS),

        "retrieved": bool(reranked),

        "domain": state.get("domain", ""),

    }

    # 正类硬信号：期望实质回答，但没有任何检索文档 → 结构上不可能答好

    if exp in ("answer", "plan") and not reranked:

        out["hard_fail_retrieval"] = True

    else:

        out["hard_fail_retrieval"] = False

    # 规划类结构信号：该出行程却走普通问答 → 结构上不可能达标（补评测盲区）

    if exp == "plan":

        ps = _plan_structure_signals(state)

        out["plan_structure"] = ps

        out["hard_fail_plan_structure"] = not ps["structured"]

    else:

        out["plan_structure"] = {}

        out["hard_fail_plan_structure"] = False

    return out





# ============================================================

# L2 LLM-as-Judge

# ============================================================



JUDGE_POS_SYSTEM = (

    "你是 RAG 系统回答质量评审员。只依据【标准要点】与【参考原文】核对模型回答，"

    "不要引入外部知识。逐条判定要点是否被覆盖；再判定回答里是否存在参考原文中完全没有依据的"

    "事实断言（幻觉）。输出严格 JSON：{\"covered\": [true/false,...], "

    "\"hallucination\": true/false, \"accept\": true/false, \"reason\": \"一句话\"}。"

    "accept=true 当且仅当：全部要点 covered=true 或未覆盖要点属于合理省略（回答未自相矛盾），"

    "且 hallucination=false。"

)



JUDGE_POS_USER = (

    "【标准要点】\n{facts}\n\n"

    "【参考原文（检索命中的知识库片段，供核对幻觉）】\n{evidence}\n\n"

    "【模型回答】\n{answer}\n\n"

    "请给出 JSON 判定。"

)



JUDGE_NEG_SYSTEM = (

    "你是 RAG 系统行为评审员。该问题属于【不应给出知识库实质内容】的类型（需拒答/澄清/闲聊/无库内答案）。"

    "请判定模型回答是否属于\"编造了看似来自知识库的实质内容\"，以下任一情况均判 fabricated=true："

    "1) 以肯定口吻给出具体事实/行程/推荐并伪装有来源；"

    "2) 输出了具体行程安排（具体地点串场、交通班次号、出发到达时刻、逐日安排）——即使措辞泛化也算；"

    "3) 对具体对象（游戏/产品/目的地/书）做出明确的质量结论或事实断言（如\"好玩\"\"值得去\""

    "\"某年某榜单\"），而非泛泛而谈。"

    "纯粹的寒暄、明确表示不知道/无法回答/库内无资料、或反问澄清均不算（fabricated=false）。"

    "输出严格 JSON：{\"fabricated\": true/false, \"accept\": true/false, \"reason\": \"一句话\"}。"

    "accept=true 当且仅当 fabricated=false。"

)



JUDGE_NEG_USER = "【问题】\n{q}\n\n【模型回答】\n{answer}\n\n请给出 JSON 判定。"





def _call_judge(system: str, user: str) -> dict | None:

    from langchain_core.messages import HumanMessage, SystemMessage



    from app.infra.llm import llm_provider



    try:

        llm = llm_provider.chat(json_mode=True)

        resp = llm.invoke(

            [

                SystemMessage(content=system),

                HumanMessage(content=user),

            ]

        )

        raw = (resp.content or "").strip()

        m = re.search(r"\{.*\}", raw, re.S)

        if not m:

            return None

        data = json.loads(m.group(0))

        if isinstance(data, dict):

            return data

    except Exception as e:  # noqa: BLE001

        print(f"    [judge 调用失败] {e!r}")

    return None





def judge_case(case: dict, state: dict) -> dict:

    exp = case.get("expect_type", "answer")

    answer = (state.get("answer") or "").strip()

    reranked = state.get("reranked_docs") or []

    facts = case.get("golden_facts") or []

    result: dict = {"method": "L1_only", "accept": None, "judge_raw": None}



    # 空回答：无论正负都不合格

    if not answer:

        result.update({"accept": False, "reason": "回答为空"})

        return result



    # plan 结构门禁（L1 前置）：金标为行程规划，但全程未进入行程链路 →

    # 「该出行程却走普通问答」。judge 只看要点覆盖会漏判（答得像样就给过），故在此拦截。

    if exp == "plan" and _PLAN_STRUCTURE_ENFORCE:

        ps = _plan_structure_signals(state)

        if not ps["structured"]:

            result.update({

                "method": "L1_plan_structure",

                "accept": False,

                "reason": (

                    "该出行程却走普通问答：未判为规划(is_plan=False)、"

                    "无行程地图(map_data 空)、规划工具全空"

                ),

                "plan_structure": ps,

            })

            return result



    if exp in ("answer", "plan"):
        evidence = "\n".join(
            f"[{doc_file_title(d)}] {doc_text(d, 400)}" for d in reranked[:3]
        )

        # ---- OBS-6：语义等价集（同一要点有多种合理写法，避免 judge 因措辞抖动误判） ----
        # golden_facts_sets = 多组「可接受的要点集」；任意一组被全覆盖且无幻觉即判过。
        # 未配置时回退单组 golden_facts，行为与原版完全一致（向后兼容，守基线）。
        facts_sets = case.get("golden_facts_sets")
        if facts_sets and isinstance(facts_sets, list) and facts_sets:
            matched_idx = None
            last_jd = None
            for idx, fset in enumerate(facts_sets):
                if not isinstance(fset, list):
                    continue
                facts_text = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(fset))
                jd = _call_judge(
                    JUDGE_POS_SYSTEM,
                    JUDGE_POS_USER.format(
                        facts=facts_text, evidence=evidence or "（无检索片段）", answer=answer
                    ),
                )
                if jd:
                    last_jd = jd
                    if bool(jd.get("accept")):
                        matched_idx = idx
                        break
            if matched_idx is not None:
                jd = last_jd
                result.update({
                    "method": "judge_equivalence",
                    "judge_raw": jd,
                    "accept": True,
                    "hallucination": bool(jd.get("hallucination")),
                    "covered": jd.get("covered"),
                    "matched_set": matched_idx,
                    "set_count": len(facts_sets),
                })
            elif last_jd is not None:
                result.update({
                    "method": "judge_equivalence",
                    "judge_raw": last_jd,
                    "accept": False,
                    "hallucination": bool(last_jd.get("hallucination")),
                    "covered": last_jd.get("covered"),
                    "set_count": len(facts_sets),
                    "reason": f"所有 {len(facts_sets)} 组语义等价要点均未被全覆盖",
                })
            # last_jd 为 None（judge 全失败）→ accept 保持 None（命中 OBS-5 告警，不计入分母）
            return result

        # ---- 原版单组 golden_facts 路径（向后兼容） ----
        if not facts:
            # 无 golden_facts：退化为“必须有出处 + 无幻觉”的弱判据（交给 judge 核对原文）
            if not evidence:
                result.update({"accept": False, "reason": "无检索文档且无标准要点，无法证明基于知识库"})
                return result
            system = JUDGE_POS_SYSTEM.replace("全部要点 covered=true 或", "")
            user = (
                "【标准要点】（无，判据=回答应与参考原文一致且无超出原文的断言）\n（无）\n\n"
                f"【参考原文】\n{evidence}\n\n【模型回答】\n{answer}\n\n请给出 JSON 判定。"
            )
            jd = _call_judge(system, user)
            if jd:
                result.update({
                    "method": "judge",
                    "judge_raw": jd,
                    "accept": bool(jd.get("accept")),
                    "hallucination": bool(jd.get("hallucination")),
                })
            return result

        facts_text = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
        jd = _call_judge(JUDGE_POS_SYSTEM, JUDGE_POS_USER.format(facts=facts_text, evidence=evidence or "（无检索片段）", answer=answer))
        if jd:
            result.update({
                "method": "judge",
                "judge_raw": jd,
                "accept": bool(jd.get("accept")),
                "hallucination": bool(jd.get("hallucination")),
                "covered": jd.get("covered"),
            })
        return result



    # 负例/闲聊/澄清类

    jd = _call_judge(JUDGE_NEG_SYSTEM, JUDGE_NEG_USER.format(q=case.get("query", ""), answer=answer))

    if jd:

        result.update({

            "method": "judge",

            "judge_raw": jd,

            "accept": bool(jd.get("accept")),

            "fabricated": bool(jd.get("fabricated")),

        })

    return result





# ============================================================

# 主流程

# ============================================================



def _maybe_record_attribution(case: dict, rec: dict) -> None:

    """Loop D1：评测未通过（accept=False）自动写一条 12 类错误归因；纯日志，失败静默。"""

    aj = rec.get("answer_judge") or {}

    if aj.get("accept") is not False:

        return

    reason = aj.get("reason") or ""

    note = " | ".join([

        f"expect={case.get('expect_type')}",

        f"route={rec.get('domain_route')}",

        f"method={aj.get('method')}",

        f"reason={reason}",

        f"answer_head={(rec.get('answer') or '')[:120]}",

    ])

    text = " ".join([case.get("query", ""), rec.get("answer") or "", reason])

    cls = suggest_attr_class(text)

    record_attribution(

        case_id=str(case.get("id", "")),

        attr_class=cls,

        note=note,

        source="eval",

        predicted=str(aj.get("accept")),

    )





def run_case(case: dict) -> dict:

    """执行单条用例：跑图 + 抓中间态 + 判定。"""

    q = case.get("query", "")

    # current_date 必须显式注入：生产路径由 create_default_state 填充，

    # 评测直接 invoke 绕过了工厂，缺失会让规划类用例的 Prompt 拿到"未知"日期（诱发日期幻觉，与生产不一致）

    from app.shared.runtime.date_utils import current_date_text  # noqa: PLC0415



    state_in = {

        "session_id": f"eval_{uuid.uuid4().hex[:12]}",

        "original_query": q,

        "is_stream": False,

        "domain": "",

        "current_date": current_date_text(),

    }

    error: str | None = None

    t0 = time.time()



    # 延迟导入，保证 CLI --help 不触发模型/依赖

    from app.process.unified_query.agent.main_graph import unified_query_app  # noqa: PLC0415



    try:

        # 直接 invoke：LangGraph 返回最终完整 state，含各阶段产物键

        # （embedding_chunks / hyde_embedding_chunks / keyword_chunks / rrf_chunks /

        #   reranked_docs / domain / tool_weather / tool_route / item_names ...）

        merged = unified_query_app.invoke(state_in)

        if not isinstance(merged, dict):

            merged = {}

    except Exception as e:  # noqa: BLE001

        error = f"{type(e).__name__}: {e}"

        print(f"    [图执行异常] {error}")

        merged = {}



    elapsed = round(time.time() - t0, 2)

    answer = (merged.get("answer") or "").strip()



    # ---- 检索层三档切片（真实中间产物，均取自最终 state） ----

    embed_titles = path_file_titles(merged.get("embedding_chunks"))

    rrf_titles = path_file_titles(merged.get("rrf_chunks"))

    rerank_titles = path_file_titles(merged.get("reranked_docs"))

    golden = set(case.get("golden_file_titles") or [])



    rec = {

        "id": case.get("id"),

        "query": q,

        "expect_type": case.get("expect_type"),

        "intent": case.get("intent"),

        "domain_route": merged.get("domain", ""),

        "elapsed_s": elapsed,

        "error": error,

        "answer": answer,

        "answer_len": len(answer),

        "retrieval": {

            "golden_file_titles": sorted(golden),

            "b1_vector_titles": embed_titles,

            "b2_rrf_titles": rrf_titles,

            "b3_rerank_titles": rerank_titles,

            "b1": compute_ret_metrics(golden, embed_titles),

            "b2": compute_ret_metrics(golden, rrf_titles),

            "b3": compute_ret_metrics(golden, rerank_titles),

        },

        "tool_weather": merged.get("tool_weather"),

        "tool_route": merged.get("tool_route"),

    }



    # 规划类额外记录：出发地解析是否合理（不做自动判分，仅留证供人工）

    if case.get("expect_type") == "plan":

        ps = _plan_structure_signals(merged)

        rec["plan_evidence"] = {

            "is_plan": ps["is_plan"],

            "has_map_data": ps["has_map_data"],

            "plan_tools": ps["plan_tools"],

            "structured": ps["structured"],

            "tool_weather": merged.get("tool_weather"),

            "tool_route": merged.get("tool_route"),

            "rewritten_query": merged.get("rewritten_query"),

            "answer_head": answer[:300],

        }



    # ---- L1 + L2 ----

    l1 = l1_check(case, merged)

    rec["l1"] = l1

    if error:

        rec["answer_judge"] = {"method": "graph_error", "accept": False, "reason": f"图执行异常 {error}"}

    else:

        rec["answer_judge"] = judge_case(case, merged)



    # === D1 Loop 闭环：评测未通过自动记录 12 类错误归因（纯日志，不影响判定） ===

    _maybe_record_attribution(case, rec)

    return rec





def aggregate_answer_metrics(recs: list[dict]) -> dict:

    """问答层四格（正类=期望实质回答 answer/plan；负类=chitchat/refuse/clarify）。"""

    tp = fp = fn = tn = 0

    unresolved = 0

    for r in recs:

        aj = r.get("answer_judge") or {}

        acc = aj.get("accept")

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

    acc_all = (tp + tn) / n if n else 0.0

    p = tp / (tp + fp) if (tp + fp) else 0.0

    r = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {

        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "unresolved": unresolved,

        "accuracy": round(acc_all, 4), "precision": round(p, 4),

        "recall": round(r, 4), "f1": round(f1, 4),

    }





def aggregate_by_domain(recs: list[dict]) -> dict:

    out = {}

    for dom in ("tourism", "plan", "chitchat"):

        sub = [r for r in recs if (r.get("expect_type") == "plan" and dom == "plan") or

               (r.get("expect_type") != "plan" and r.get("domain_route") == dom) or

               (dom == "chitchat" and r.get("expect_type") in ("chitchat", "refuse", "clarify"))]

        if sub:

            out[dom] = aggregate_answer_metrics(sub)

    return out





def collect_eval_warnings(recs: list[dict], summary: dict) -> list[str]:
    """OBS-5：judge 失败(accept=None)曾静默剔出分母致 Accuracy 虚高；此处显式告警。

    不改口径（unresolved 仍不计入分母），仅提升可见性，便于区分
    「真实 FN」与「judge 超时/失败导致的未判定」。
    """
    warnings: list[str] = []
    unresolved_ids = [
        r.get("id") for r in recs
        if (r.get("answer_judge") or {}).get("accept") is None
    ]
    if unresolved_ids:
        ratio = len(unresolved_ids) / max(1, len(recs))
        warnings.append(
            "⚠️ 未判定(unresolved)用例 {n} 条（占比 {p:.1%}）：case_id={ids}。"
            "原因=judge 调用失败/超时(accept=None)，这些用例**不计入分母**，"
            "会使 Accuracy 虚高，易与真实 FN 混淆。建议：复跑该批用例或先排查 judge 可用性，"
            "勿直接采信当前 Accuracy。".format(
                n=len(unresolved_ids), p=ratio, ids=unresolved_ids
            )
        )
    return warnings


def render_md(recs: list[dict], summary: dict, args) -> str:

    lines = []

    lines.append("# 文旅单域 RAG 评测运行报告")

    lines.append("")

    lines.append(f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append(f"- 评测集：{args.cases}")

    lines.append(f"- case 数：{len(recs)}")

    lines.append(f"- 消融：{'开' if args.ablation else '关'}，LLM-Judge：{'开' if args.judge else '关'}")

    lines.append("")

    lines.append("## 问答层指标（全域）")

    lines.append("")

    qa = summary["answer_global"]

    lines.append("| TP | FP | FN | TN | 未判定 | Accuracy | Precision | Recall | F1 |")

    lines.append("|---|---|---|---|---|---|---|---|---|")

    lines.append(

        f"| {qa['tp']} | {qa['fp']} | {qa['fn']} | {qa['tn']} | {qa['unresolved']} "

        f"| {qa['accuracy']} | {qa['precision']} | {qa['recall']} | {qa['f1']} |"

    )

    lines.append("")

    warns = collect_eval_warnings(recs, summary)
    if warns:
        lines.append("## ⚠️ 评测告警（OBS-5）")
        lines.append("")
        for _w in warns:
            lines.append(f"- {_w}")
        lines.append("")

    lines.append("## 分域问答指标")

    lines.append("")

    lines.append("| 域 | N | Accuracy | Precision | Recall | F1 |")

    lines.append("|---|---|---|---|---|---|")

    for dom, d in summary["answer_by_domain"].items():

        n = d["tp"] + d["fp"] + d["fn"] + d["tn"]

        lines.append(f"| {dom} | {n} | {d['accuracy']} | {d['precision']} | {d['recall']} | {d['f1']} |")

    lines.append("")

    if args.ablation:

        lines.append("## 检索层消融（macro，候选=各阶段 top5 的 file_title）")

        lines.append("")

        lines.append("| 阶段 | Precision@5 | Recall@5 | F1@5 | Hit@5 |")

        lines.append("|---|---|---|---|---|")

        for key, name in (("b1", "B1 仅向量路"), ("b2", "B2 多路+RRF"), ("b3", "B3 全链路(+Rerank)")):

            # 只统计"有金标来源"的正类用例：无金标负例永远 0 分，混入会系统性拉低 macro

            rows = [r["retrieval"][key] for r in recs

                    if r["retrieval"].get(key) and r["retrieval"]["golden_file_titles"]]

            if rows:

                m = macro_mean(rows)

                lines.append(f"| {name} | {m['precision']} | {m['recall']} | {m['f1']} | {m['hit']} |")

        lines.append("")

        lines.append("> Hit@5 = 该阶段 top5 是否含至少一个 golden 来源（命中率，等价检索准确率）。")

        lines.append("")

    lines.append("## 单条明细")

    lines.append("")

    lines.append("| id | expect | 路由domain | 判定 | accept | 检索b3 F1 | 耗时s | 备注 |")

    lines.append("|---|---|---|---|---|---|---|---|")

    for r in recs:

        aj = r.get("answer_judge") or {}

        ret = r["retrieval"]["b3"] if r["retrieval"].get("b3") else {}

        note = (aj.get("reason") or r.get("error") or "")[:60]

        lines.append(

            f"| {r['id']} | {r['expect_type']} | {r.get('domain_route','')} | {aj.get('method','')} "

            f"| {aj.get('accept')} | {ret.get('f1','-')} | {r['elapsed_s']} | {note} |"

        )

    return "\n".join(lines)





def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--cases", default=str(ROOT / "data" / "eval_cases.json"))

    ap.add_argument("--out", default="")

    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")

    ap.add_argument("--ids", default="", help="只跑指定 id 的用例（逗号分隔，第三轮定向回归用）")

    ap.add_argument("--ablation", action="store_true", default=True)

    ap.add_argument("--no-ablation", dest="ablation", action="store_false")

    ap.add_argument("--judge", action="store_true", default=True)

    ap.add_argument("--no-judge", dest="judge", action="store_false")

    ap.add_argument("--workers", type=int, default=1,

                    help="并发执行用例数，默认 1（串行）。注意：并发会提高触发 LLM 供应商"

                         "限流（如智谱 429）的概率，导致 judge 无响应而产生假 FN。"

                         "建议仅在需要提速且额度充足时使用 2~3。")

    args = ap.parse_args()



    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")



    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]

    if args.limit:

        cases = cases[: args.limit]

    if args.ids:

        wanted = [s.strip() for s in args.ids.split(",") if s.strip()]

        by_id = {c.get("id"): c for c in cases}

        missing = [i for i in wanted if i not in by_id]

        if missing:

            raise SystemExit(f"--ids 中存在未知用例 id: {missing}")

        cases = [by_id[i] for i in wanted]

    print(f"== 评测开始：{len(cases)} 条 ==")



    recs = []

    if args.workers > 1 and len(cases) > 1:

        # 并发模式：IO 密集型（LLM/联网）适合线程池；错峰启动 + 安全上限 3，

        # 降低瞬间打满 LLM 供应商触发 429 的概率（详见 --workers 帮助文本）。

        from concurrent.futures import ThreadPoolExecutor, as_completed

        import threading

        max_workers = min(args.workers, 3)

        lock = threading.Lock()

        done = 0

        holder = [None] * len(cases)

        print(f"== 并发模式 workers={max_workers}（错峰启动，429 风险自负）==")



        def _run(idx: int, c: dict):

            time.sleep(idx * 0.25)  # 错峰，避免首拍并发瞬时打满

            return run_case(c)



        with ThreadPoolExecutor(max_workers=max_workers) as ex:

            fut_to_idx = {ex.submit(_run, i, c): i for i, c in enumerate(cases)}

            for fut in as_completed(fut_to_idx):

                idx = fut_to_idx[fut]

                rec = fut.result()

                holder[idx] = rec

                with lock:

                    done += 1

                    aj = rec.get("answer_judge") or {}

                    print(f"[{done}/{len(cases)}] {cases[idx]['id']} "

                          f"route={rec.get('domain_route','')} judge={aj.get('accept')} "

                          f"t={rec['elapsed_s']}s")

        recs = holder

    else:

        for i, c in enumerate(cases, 1):

            print(f"[{i}/{len(cases)}] {c['id']} {c['query'][:40]}...")

            rec = run_case(c)

            recs.append(rec)

            aj = rec.get("answer_judge") or {}

            print(f"    -> route={rec.get('domain_route','')} judge={aj.get('accept')} "

                  f"({aj.get('method')}) b3F1={rec['retrieval'].get('b3',{}).get('f1')} "

                  f"len(answer)={rec['answer_len']} t={rec['elapsed_s']}s")



    summary = {

        "answer_global": aggregate_answer_metrics(recs),

        "answer_by_domain": aggregate_by_domain(recs),

    }

    warnings = collect_eval_warnings(recs, summary)
    print("\n== 全域问答指标 ==")
    print(json.dumps(summary["answer_global"], ensure_ascii=False, indent=2))
    if warnings:
        print("\n" + "\n".join(warnings))

    if args.ablation:

        rows = {"b1": [], "b2": [], "b3": []}

        pos_recs = [r for r in recs if r["expect_type"] in ("answer", "plan") and r["retrieval"]["golden_file_titles"]]

        for r in pos_recs:

            for k in rows:

                if r["retrieval"].get(k):

                    rows[k].append(r["retrieval"][k])

        print("== 检索层消融（macro top5，仅正类有金标用例）==")

        for k, name in (("b1", "B1 仅向量路"), ("b2", "B2 多路+RRF"), ("b3", "B3 全链路(+Rerank)")):

            if rows[k]:

                print(name, macro_mean(rows[k]))



    out_dir = args.out or str(ROOT / "output" / "eval" / time.strftime("%Y%m%d_%H%M%S"))

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    payload = {"summary": summary, "warnings": warnings, "args": vars(args), "cases": recs}

    Path(out_dir, "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_md(recs, summary, args)

    Path(out_dir, "report.md").write_text(md, encoding="utf-8")

    print(f"\n== 结果已写入 {out_dir} ==")





if __name__ == "__main__":

    main()

