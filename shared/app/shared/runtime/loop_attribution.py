# -*- coding: utf-8 -*-
"""Loop Engineering 基础：12 类错误归因框架 + 隐式反馈捕获。

框架来源：《企业 Agent 落地十二大工程框架（上篇）》Loop Engineering ——
"错误归因先于优化"；"隐式反馈（追问/放弃/驳回/改写）> 显式点赞"。

本模块为**纯增量、零风险**：仅向 logs/ 追加 jsonl 日志，不改动任何主链路
判定/生成逻辑；所有写入均在 try/except 中，失败绝不影响业务。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 12 类错误归因（对齐框架定义）
# ---------------------------------------------------------------------------
ATTRIBUTION_CLASSES = [
    "prompt", "context", "retrieval", "ontology", "process",
    "skill", "agent", "tool", "permission", "safety", "product", "data",
]

ATTRIBUTION_DESC = {
    "prompt": "提示词指令/格式/拒答规则问题",
    "context": "上下文给错/缺失/未压缩/记忆错误",
    "retrieval": "检索召回/切分/向量/BM25 问题",
    "ontology": "业务语义/实体/标签建模问题",
    "process": "业务流程/协作/状态机问题",
    "skill": "能力封装/Schema/版本问题",
    "agent": "规划/决策/多Agent协作问题",
    "tool": "工具调用/参数/外部接口问题",
    "permission": "权限/越权/脱敏问题",
    "safety": "安全合规/护栏问题",
    "product": "产品体验/UX/前端问题",
    "data": "数据质量/源/版本/有效期问题",
}

# 隐式反馈信号类型
FEEDBACK_TYPES = ("follow_up", "abandon", "reject", "rewrite")

_LOOP_DIR = Path(__file__).resolve().parents[3] / "logs"
_ATTR_FILE = _LOOP_DIR / "error_attribution.jsonl"
_FEEDBACK_FILE = _LOOP_DIR / "loop_feedback.jsonl"


def _ensure_dir() -> None:
    _LOOP_DIR.mkdir(parents=True, exist_ok=True)


def record_attribution(
    case_id: str,
    attr_class: str,
    note: str = "",
    source: str = "eval",
    predicted: Optional[str] = None,
    golden: Optional[str] = None,
    auto: bool = True,
) -> dict:
    """记录一条错误归因。attr_class 必须属于 ATTRIBUTION_CLASSES（否则归 data）。"""
    if attr_class not in ATTRIBUTION_CLASSES:
        attr_class = "data"
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_id": str(case_id),
        "attr_class": attr_class,
        "attr_desc": ATTRIBUTION_DESC.get(attr_class, ""),
        "source": source,
        "auto": auto,
        "note": note,
        "predicted": predicted,
        "golden": golden,
    }
    try:
        _ensure_dir()
        with open(_ATTR_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 归因日志失败绝不影响主链路
    return rec


def record_implicit_feedback(
    session_id: str,
    feedback_type: str,
    detail: str = "",
    query: str = "",
) -> Optional[dict]:
    """记录一条隐式反馈信号。feedback_type ∈ FEEDBACK_TYPES。"""
    if feedback_type not in FEEDBACK_TYPES:
        feedback_type = "follow_up"
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": str(session_id),
        "feedback_type": feedback_type,
        "detail": detail,
        "query": (query or "")[:200],
    }
    try:
        _ensure_dir()
        with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 轻量启发式（人工可覆写）
# ---------------------------------------------------------------------------
_KEYWORD_MAP = {
    "retrieval": ["召回", "检索", "chunk", "向量", "bm25", "rerank", "未召回", "召回错", "没找到", "搜不到"],
    "context": ["上下文", "记忆", "历史", "压缩", "截断", "忘了", "记错"],
    "prompt": ["格式", "instruction", "拒答", "指令遵循", "json", "输出格式"],
    "data": ["过期", "时效", "知识库", "源数据", "版本", "旧了", "已更新"],
    "tool": ["工具", "高德", "天气", "路线", "铁路", "api", "调用失败", "超时"],
    "ontology": ["标签", "实体", "类型", "分类", "分错"],
    "safety": ["越权", "脱敏", "泄露", "权限"],
    "product": ["前端", "页面", "按钮", "体验", "界面"],
}


def suggest_attr_class(text: str) -> str:
    """从失败文本启发式推断归因类（首命中优先；都不中归 data）。"""
    t = (text or "").lower()
    for cls, kws in _KEYWORD_MAP.items():
        if any(kw in t for kw in kws):
            return cls
    return "data"


# 改写/纠正信号词（用于 D2 隐式反馈）
_REWRITE_MARKERS = ["不是", "改成", "换成", "改为", "应该是", "错了", "不对", "我要去", "重新", "纠正"]


def detect_implicit_feedback(query: str, prev_query: str = "") -> Optional[str]:
    """从用户 query 文本启发式识别隐式反馈信号类型，无则返回 None。

    - 改写/纠正（rewrite）：含「不是/改成/换成/我要去X不是Y…」等显式纠错词。
    - 追问（follow_up）：在已有上文前提下，query 较短或以衔接词开头。
    - 放弃（abandon）：出现「算了/不用了/换个话题/不聊了」等。
    说明：abandon/follow_up 的完整判定依赖会话历史；此处仅做文本层粗筛，
    历史级精细判定在后续接入会话状态后增强。
    """
    q = (query or "").strip()
    if not q:
        return None
    if any(m in q for m in _REWRITE_MARKERS):
        return "rewrite"
    if prev_query:
        if any(w in q for w in ["算了", "不用了", "不聊了", "换个话题", "不要了"]):
            return "abandon"
        connectors = ("那", "这个", "上面", "再", "然后", "还有", "接着", "它", "她", "他")
        if len(q) <= 18 and q[:1] in connectors:
            return "follow_up"
    return None
