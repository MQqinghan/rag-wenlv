# -*- coding: utf-8 -*-
"""Loop Engineering 单测（D1 错误归因 + 隐式反馈；D3 优化提案管道）。

离线、零外部请求、不污染真实 logs/（日志文件路径被重定向到临时目录）。

覆盖：
  D1 loop_attribution
    1. 12 类归因枚举与描述完整性
    2. record_attribution：正常落盘 / 非法归因类收敛 / JSON 可解析
    3. record_implicit_feedback：正常落盘 / 非法类型收敛 / query 截断
    4. suggest_attr_class：关键词命中优先级 / 无命中归 data
    5. detect_implicit_feedback：rewrite / abandon / follow_up / None
  D3 loop_pipeline
    6. create_proposal：id 唯一 / 初始状态 / 审计 / 非法归因类
    7. get_proposal / list_proposals 过滤
    8. run_regression 门禁：达标 / 未达标 / acc 解析失败
    9. 状态机门禁：approve←regression_passed、gray←approved、rollback←approved|gray_released
   10. _parse_acc_from_metrics：正常 / 缺字段
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.runtime import loop_attribution as la
from app.shared.runtime import loop_pipeline as lp

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


_TMP = Path(tempfile.mkdtemp(prefix="loop_test_"))
_ATTR = _TMP / "error_attribution.jsonl"
_FB = _TMP / "loop_feedback.jsonl"
_PROP = _TMP / "optimization_proposals.jsonl"

# ============================================================
# D1-1 枚举完整性
# ============================================================
print("== D1-1 归因枚举 ==")
check("12 类归因", len(la.ATTRIBUTION_CLASSES) == 12)
check("归因类与描述一一对应", all(c in la.ATTRIBUTION_DESC for c in la.ATTRIBUTION_CLASSES))
check("4 类隐式反馈", la.FEEDBACK_TYPES == ("follow_up", "abandon", "reject", "rewrite"))

with mock.patch.object(la, "_ATTR_FILE", _ATTR), mock.patch.object(la, "_FEEDBACK_FILE", _FB):
    # ============================================================
    # D1-2 record_attribution
    # ============================================================
    print("== D1-2 record_attribution ==")
    rec = la.record_attribution(
        "c1", "retrieval", note="召回为空", source="eval", predicted="p", golden="g"
    )
    check("字段完整", rec["attr_class"] == "retrieval"
          and rec["attr_desc"] == la.ATTRIBUTION_DESC["retrieval"]
          and rec["source"] == "eval" and rec["auto"] is True)
    check("落盘一行", _ATTR.read_text(encoding="utf-8").count("\n") == 1)

    rec2 = la.record_attribution("c2", "不存在的归因类")
    check("非法归因类收敛为 data", rec2["attr_class"] == "data")

    lines = [json.loads(x) for x in _ATTR.read_text(encoding="utf-8").splitlines() if x.strip()]
    check("落盘 JSON 可解析且顺序正确", len(lines) == 2 and lines[0]["case_id"] == "c1"
          and lines[1]["attr_class"] == "data")

    # ============================================================
    # D1-3 record_implicit_feedback
    # ============================================================
    print("== D1-3 record_implicit_feedback ==")
    fb = la.record_implicit_feedback("s1", "rewrite", detail="纠正", query="x" * 300)
    check("返回记录", isinstance(fb, dict) and fb["feedback_type"] == "rewrite")
    check("query 截断到 200", len(fb["query"]) == 200)

    fb2 = la.record_implicit_feedback("s2", "非法类型")
    check("非法类型收敛为 follow_up", fb2["feedback_type"] == "follow_up")
    check("反馈落盘两行", _FB.read_text(encoding="utf-8").count("\n") == 2)

# ============================================================
# D1-4 suggest_attr_class
# ============================================================
print("== D1-4 suggest_attr_class ==")
check("召回词 → retrieval", la.suggest_attr_class("这次召回错了，没找到资料") == "retrieval")
check("上下文词 → context", la.suggest_attr_class("上下文丢了，忘了之前说的") == "context")
check("格式词 → prompt", la.suggest_attr_class("输出格式不对，json 坏了") == "prompt")
check("工具词 → tool", la.suggest_attr_class("高德接口超时了") == "tool")
check("时效词 → data", la.suggest_attr_class("知识库数据过期了") == "data")
check("无命中 → data", la.suggest_attr_class("完全无关的一句话zzz") == "data")
check("空串 → data", la.suggest_attr_class("") == "data")

# ============================================================
# D1-5 detect_implicit_feedback
# ============================================================
print("== D1-5 detect_implicit_feedback ==")
check("纠错词 → rewrite", la.detect_implicit_feedback("我要去杭州，不是成都") == "rewrite")
check("换词 → rewrite", la.detect_implicit_feedback("把第三天改成博物馆") == "rewrite")
check("有上文+放弃词 → abandon", la.detect_implicit_feedback("算了，不用了", "成都三日游") == "abandon")
check("有上文+衔接词短句 → follow_up",
      la.detect_implicit_feedback("那门票多少钱", "成都三日游行程") == "follow_up")
check("无上文 → None", la.detect_implicit_feedback("那门票多少钱") is None)
check("普通长句 → None",
      la.detect_implicit_feedback("请详细介绍一下成都的旅游资源分布情况", "上轮") is None)
check("空串 → None", la.detect_implicit_feedback("") is None)

# ============================================================
# D3-6 create_proposal
# ============================================================
print("== D3-6 create_proposal ==")
with mock.patch.object(lp, "_PROPOSAL_FILE", _PROP):
    p1 = lp.create_proposal("t1", "intent_route", "改路由", "retrieval", 0.9)
    check("id 前缀 opt-", p1["proposal_id"].startswith("opt-"))
    check("初始 status=proposed", p1["status"] == "proposed")
    check("baseline_acc 浮点化", isinstance(p1["baseline_acc"], float) and p1["baseline_acc"] == 0.9)
    check("审计含 create", p1["audit"][0]["action"] == "create" and p1["audit"][0]["by"] == "human")
    check("regression_acc 初始为 None", p1["regression_acc"] is None)

    p2 = lp.create_proposal("t2", "answer", "改答案", "非法类", 0.9)
    check("非法归因类收敛为 data", p2["attr_class"] == "data")

    p3 = lp.create_proposal("t3", "x", "y", "prompt", 0.9)
    check("连续创建 id 唯一",
          len({p1["proposal_id"], p2["proposal_id"], p3["proposal_id"]}) == 3)

    # ============================================================
    # D3-7 查询
    # ============================================================
    print("== D3-7 查询 ==")
    check("get_proposal 命中", lp.get_proposal(p1["proposal_id"])["title"] == "t1")
    check("get_proposal 未命中 → None", lp.get_proposal("opt-not-exist") is None)
    check("list 全量", len(lp.list_proposals()) == 3)
    check("list 状态过滤", len(lp.list_proposals("proposed")) == 3
          and len(lp.list_proposals("approved")) == 0)

    # 强制同一毫秒连续创建 —— 原实现（毫秒内不比对）会碰撞，现要求自动加后缀去重
    with mock.patch.object(lp.time, "strftime", return_value="20260101-000000"), \
            mock.patch.object(lp.time, "time", return_value=1.250):
        d1 = lp.create_proposal("d1", "x", "y", "prompt", 0.9)
        d2 = lp.create_proposal("d2", "x", "y", "prompt", 0.9)
        d3 = lp.create_proposal("d3", "x", "y", "prompt", 0.9)
        check("同毫秒强制创建 id 唯一",
              len({d1["proposal_id"], d2["proposal_id"], d3["proposal_id"]}) == 3)
        check("冲突时按 base / -2 / -3 递增",
              d1["proposal_id"] == "opt-20260101-000000-250"
              and d2["proposal_id"] == "opt-20260101-000000-250-2"
              and d3["proposal_id"] == "opt-20260101-000000-250-3")

    # ============================================================
    # D3-8 run_regression 门禁
    # ============================================================
    print("== D3-8 run_regression 门禁 ==")
    r = lp.run_regression(p1["proposal_id"], skip_run=True, mock_acc=0.95)
    check("达标 → regression_passed",
          r["status"] == "regression_passed" and r["regression_acc"] == 0.95)
    check("回归写入审计", any(a.get("action") == "regression" for a in r["audit"]))

    r = lp.run_regression(p2["proposal_id"], skip_run=True, mock_acc=0.5)
    check("未达标 → rejected", r["status"] == "rejected")

    r = lp.run_regression(p3["proposal_id"], skip_run=True, mock_acc=None)
    check("acc 解析失败 → rejected 且留说明",
          r["status"] == "rejected" and "解析失败" in r["regression_report"])

    try:
        lp.run_regression("opt-not-exist", skip_run=True, mock_acc=1.0)
        check("不存在提案应抛错", False)
    except ValueError:
        check("不存在提案应抛错", True)

    # ============================================================
    # D3-9 状态机门禁
    # ============================================================
    print("== D3-9 状态机门禁 ==")
    try:
        lp.approve_proposal(p2["proposal_id"])
        check("rejected 不可审批", False)
    except ValueError:
        check("rejected 不可审批", True)

    a = lp.approve_proposal(p1["proposal_id"], by="master")
    check("regression_passed → approved", a["status"] == "approved")
    check("审批记录操作人", a["audit"][-1]["by"] == "master")

    try:
        lp.gray_release_proposal(p3["proposal_id"])
        check("未审批不可灰度", False)
    except ValueError:
        check("未审批不可灰度", True)

    g = lp.gray_release_proposal(p1["proposal_id"])
    check("approved → gray_released", g["status"] == "gray_released")

    rb = lp.rollback_proposal(p1["proposal_id"], reason="线上异常")
    check("gray_released → rolled_back", rb["status"] == "rolled_back")
    check("回滚记录原因", rb["audit"][-1]["reason"] == "线上异常")

    try:
        lp.rollback_proposal(p1["proposal_id"])
        check("已回滚不可再回滚", False)
    except ValueError:
        check("已回滚不可再回滚", True)

# ============================================================
# D3-10 _parse_acc_from_metrics
# ============================================================
print("== D3-10 解析 metrics ==")
mp = _TMP / "metrics.json"
mp.write_text(json.dumps({"summary": {"answer_global": {"accuracy": 0.9321}}}), encoding="utf-8")
check("正常解析", lp._parse_acc_from_metrics(mp) == 0.9321)
mp.write_text(json.dumps({"summary": {}}), encoding="utf-8")
check("缺字段 → None", lp._parse_acc_from_metrics(mp) is None)
check("文件不存在 → None", lp._parse_acc_from_metrics(_TMP / "nope.json") is None)

print()
print(f"== loop_engineering 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
