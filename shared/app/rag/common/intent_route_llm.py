# -*- coding: utf-8 -*-
"""B1 意图路由 LLM 主导化：单次多路决策路由（增量，不破坏旧链路）。

设计（对齐《模块拆分与意图路由升级规划》B1）：
- 旧的「正则为主 + LLM 兜底」保留为 legacy 路径，由特性开关 INTENT_ROUTE_LLM_FIRST 控制。
- 新路径（LLM 主导）：一次 LLM 调用同时产出 domain / rewritten / source_policy /
  model_tier / tools / is_plan / confidence（多路决策，呼应 Hermes 否决式路由 + 模型档位）。
- 安全正则降级为「前置短路 + 硬覆盖 + 兜底」：知识型句式 LLM 判闲聊时强制 tourism；
  LLM 返回非法域时回退 legacy 路径。绝不交出安全关键判定。

特性开关（.env）：
  INTENT_ROUTE_LLM_FIRST=true  启用 LLM 主导（默认 false，保证评测基线不破）。
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm import llm_provider
from app.shared.config.common import env_bool
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.logger import logger, step_log

# 复用旧路由服务的域枚举与安全正则（避免重复定义、保持单一真相）
from app.rag.common.intent_route_service import (  # noqa: F401
    DOMAIN_TOURISM,
    DOMAIN_CHITCHAT,
    match_by_rule,
    inherit_domain_if_followup,
    classify_by_llm,
    _KNOWLEDGE_QUERY_PATTERN,
    _route_plan_signal_hit,
)

_ALLOWED_DOMAINS = (DOMAIN_TOURISM, DOMAIN_CHITCHAT)
_ALLOWED_TIERS = ("lite", "standard", "longctx")
_ALLOWED_TOOLS = ("weather", "route", "rail", "poi")
_ALLOWED_SOURCE = ("kb", "web", "kb_then_web")


def _normalize_led(raw: dict, query: str) -> dict:
    """把 LLM 原始输出收敛到合法值域；非法域/缺字段都有兜底。"""
    domain = raw.get("domain", "")
    if domain not in _ALLOWED_DOMAINS:
        domain = ""
    rewritten = (raw.get("rewritten") or "").strip() or query
    source = raw.get("source_policy", "")
    if source not in _ALLOWED_SOURCE:
        source = "kb_then_web" if domain in (DOMAIN_TOURISM, DOMAIN_CHITCHAT) else "kb"
    tier = raw.get("model_tier", "")
    if tier not in _ALLOWED_TIERS:
        tier = "standard"
    tools = [t for t in (raw.get("tools") or []) if t in _ALLOWED_TOOLS]
    is_plan = bool(raw.get("is_plan", False))
    try:
        conf = float(raw.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {
        "domain": domain,
        "rewritten": rewritten,
        "source_policy": source,
        "model_tier": tier,
        "tools": tools,
        "is_plan": is_plan,
        "confidence": conf,
    }


def _call_led_model(query: str, history_text: str) -> dict:
    """执行一次 LLM 多路决策（内部函数，供缓存包装）。"""
    client = llm_provider.chat(json_mode=True)
    prompt = load_prompt(
        "tourism/intent_route_v3",
        history_text=history_text,
        query=query,
    )
    messages = [
        SystemMessage(content="你是智能问答系统的意图路由决策器，只能输出合法 JSON。"),
        HumanMessage(content=prompt),
    ]
    result = (client | JsonOutputParser()).invoke(messages)
    if not isinstance(result, dict):
        raise ValueError(f"LLM 路由返回非 dict: {type(result)}")
    return result


@step_log("route_by_llm_led")
def route_by_llm_led(query: str, history_text: str = "") -> dict:
    """LLM 主导单次多路决策路由。失败/异常时返回空域（交由编排层兜底）。"""
    try:
        raw = cached_invoke(
            namespace="intent_route_v3",
            cache_parts=(query, history_text),
            producer=lambda: _call_led_model(query, history_text),
            cache_label=query[:20],
            semantic=True,
        )
        led = _normalize_led(raw, query)
        # 安全硬覆盖：知识型问句不可判闲聊
        if led["domain"] == DOMAIN_CHITCHAT and _KNOWLEDGE_QUERY_PATTERN.search(query):
            logger.warning(f"LLM 判闲聊但命中知识型句式，推翻为 tourism: [{query[:30]}]")
            led["domain"] = DOMAIN_TOURISM
            led["source_policy"] = "kb_then_web"
        if not led["domain"]:
            logger.warning("LLM 路由未给出合法域，交由 legacy 兜底")
            led["domain"] = None  # type: ignore[assignment]
        return led
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM 主导路由失败，交由 legacy 兜底：{e}")
        return {"domain": None, "rewritten": query, "source_policy": "kb",
                "model_tier": "standard", "tools": [], "is_plan": False, "confidence": 0.0}


def _legacy_domain(query: str, session_id: str = "") -> str:
    """复现旧链路域判定：正则 → 追问继承 → LLM 兜底。"""
    domain = match_by_rule(query)
    if not domain:
        inh, _ = inherit_domain_if_followup(query, session_id)
        domain = inh or classify_by_llm(query)
    return domain or DOMAIN_TOURISM


@step_log("route_intent")
def route_intent(
    query: str,
    history_text: str = "",
    session_id: str = "",
    use_llm_first: Optional[bool] = None,
) -> dict:
    """统一路由入口（B1 新入口）。

    - use_llm_first=None 时读 .env INTENT_ROUTE_LLM_FIRST（默认 false）。
    - false：精确复现旧链路（仅返回 domain，其余字段给保守默认），保证评测基线不破。
    - true ：LLM 主导多路决策；LLM 失败/非法域时自动回退 legacy 求 domain。
    """
    flag = env_bool("INTENT_ROUTE_LLM_FIRST", False) if use_llm_first is None else use_llm_first
    if not flag:
        domain = _legacy_domain(query, session_id)
        return {
            "domain": domain,
            "rewritten": query,
            "source_policy": "kb_then_web",
            "model_tier": "standard",
            "tools": [],
            "is_plan": False,
            "confidence": 1.0,
            "route_mode": "legacy",
        }

    led = route_by_llm_led(query, history_text)
    if not led.get("domain"):
        # LLM 没给合法域：legacy 兜底求 domain，其余字段保留 LLM 的改写/工具
        led["domain"] = _legacy_domain(query, session_id)
    # 规划信号硬覆盖：LLM 漏判 is_plan 时用确定性规则补回（对齐 legacy 守卫）
    if not led.get("is_plan") and _route_plan_signal_hit(query):
        led["is_plan"] = True
    led["route_mode"] = "llm_led"
    return led
