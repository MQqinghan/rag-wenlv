# -*- coding: utf-8 -*-
"""E·Harness 接线层：把 B1 输出的 model_tier 与 LLM 可靠性包装接进答案生成主链路。

本模块只做「接线」，不改变任何默认行为——两个特性开关均默认 false：
- LLM_TIER_ROUTING=true ：答案生成按 state.route_info.model_tier 选模型（chat_by_tier）；
- LLM_RELIABILITY=true  ：答案生成的非流式调用套超时/熔断/降级（reliable_llm_call）。

开关 OFF 时，answer_llm_client 等价于 `llm_provider.chat().bind(max_tokens=...)`，
reliable_invoke 等价于 `client.invoke(prompt)`，即行为与接线前一致（零基线风险）。
"""
from __future__ import annotations

from app.infra.llm import llm_provider
from app.infra.llm_reliability import reliable_llm_call
from app.shared.config.common import env_bool, env_str
from app.shared.runtime.logger import logger

_TIERS = ("lite", "standard", "longctx")


def resolve_tier(state: dict, default: str = "standard") -> str:
    """从 route_info.model_tier 取模型档位；非法/缺失回默认。"""
    tier = ((state or {}).get("route_info") or {}).get("model_tier")
    return tier if tier in _TIERS else default


def answer_llm_client(state: dict, max_tokens: int, tier_default: str = "standard"):
    """返回答案生成用的 LLM 客户端；LLM_TIER_ROUTING=true 时按档位选模型，失败回默认。"""
    if env_bool("LLM_TIER_ROUTING", False):
        tier = resolve_tier(state, tier_default)
        try:
            return llm_provider.chat_by_tier(tier).bind(max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Harness] 档位模型获取失败，回落默认模型：{e}")
    return llm_provider.chat().bind(max_tokens=max_tokens)


def reliable_invoke(client, prompt, state: dict, tier_default: str = "standard"):
    """非流式答案调用：LLM_RELIABILITY=true 时套超时/熔断/降级，否则直调（等价现状）。"""
    if env_bool("LLM_RELIABILITY", False):
        try:
            timeout = float(env_str("LLM_TIMEOUT_SECONDS", "60") or 60)
        except Exception:  # noqa: BLE001
            timeout = 60.0
        return reliable_llm_call(
            client, prompt, tier=resolve_tier(state, tier_default), timeout=timeout
        )
    return client.invoke(prompt)
