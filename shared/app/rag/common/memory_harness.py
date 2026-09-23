# -*- coding: utf-8 -*-
"""F·Context 接线层：长期记忆(user_id) 与长对话压缩接入答案生成主链路。

特性开关（.env，均默认 false——OFF 时行为与接线前一致，零基线风险）：
- USER_MEMORY_ENABLED=true        ：按 state.user_id 读取长期画像注入答案 Prompt；
                                    行程生成成功后回写 history_itineraries。
- CONTEXT_COMPRESSION_ENABLED=true：对 state.history 做四阶段压缩（首尾保护 + 中间摘要）。

全部异常安全：读取 / 压缩 / 回写失败都不影响主链路。
"""
from __future__ import annotations

import re

from app.rag.common.context_compressor import compress_messages
from app.rag.common.user_memory import user_memory_store
from app.shared.config.common import env_bool
from app.shared.runtime.logger import logger

_DAYS_PATTERN = re.compile(r"(\d{1,2})\s*(?:天|日)")


def maybe_compress_history(state: dict, history: list) -> list:
    """CONTEXT_COMPRESSION_ENABLED=true 时压缩 history；关闭/失败时原样返回。"""
    if not env_bool("CONTEXT_COMPRESSION_ENABLED", False):
        return history
    try:
        return compress_messages(history)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Context] 长对话压缩失败，保留原历史：{e}")
        return history


def memory_block(state: dict) -> str:
    """USER_MEMORY_ENABLED=true 时返回用户长期画像文本块（无 user_id / 无记忆 → 空串）。"""
    if not env_bool("USER_MEMORY_ENABLED", False):
        return ""
    user_id = (state or {}).get("user_id")
    if not user_id:
        return ""
    try:
        profile = user_memory_store.get_profile(user_id) or {}
        interests = user_memory_store.get_interests(user_id) or {}
        recent = user_memory_store.get_recent_itineraries(user_id, limit=3) or []
        if not profile and not interests and not recent:
            return ""
        lines = ["【用户长期画像】（跨会话记忆，仅供参考；与本次问题无关时忽略，勿生硬复述）"]
        if profile:
            lines.append("基本信息：" + "；".join(f"{k}={v}" for k, v in profile.items() if v))
        for kind, vals in interests.items():
            if vals:
                lines.append(f"偏好-{kind}：" + "、".join(str(v) for v in list(vals)[:8]))
        if recent:
            lines.append(
                "最近行程："
                + "；".join(f"{r.get('city') or ''}{str(r.get('days') or '')}天" for r in recent)
            )
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Memory] 读取长期记忆失败，忽略：{e}")
        return ""


def record_plan_memory(state: dict) -> None:
    """USER_MEMORY_ENABLED=true 时把本次行程回写长期记忆（目的地 + 天数，失败忽略）。"""
    if not env_bool("USER_MEMORY_ENABLED", False):
        return
    user_id = (state or {}).get("user_id")
    if not user_id:
        return
    try:
        destination = ""
        try:
            from app.rag.tourism_query.weather_tool_service import extract_destination_info

            info = extract_destination_info(state) or {}
            destination = (info.get("destination") or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Memory] 目的地解析失败，跳过行程记忆：{e}")
        if not destination:
            return
        query = state.get("original_query") or state.get("rewritten_query") or ""
        m = _DAYS_PATTERN.search(query)
        days = int(m.group(1)) if m else 0
        user_memory_store.record_itinerary(user_id, destination, days, note=query[:40])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Memory] 回写行程记忆失败，忽略：{e}")
