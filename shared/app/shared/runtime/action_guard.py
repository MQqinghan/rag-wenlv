# -*- coding: utf-8 -*-
"""I·Action 风控接线：工具执行前的动作授权闸门。

特性开关 ACTION_GUARD_ENABLED（默认 false）：OFF 时 require_action 零行为，
主链路与接线前完全一致（零基线风险）；ON 时未登记/未启用/需确认的动作会被拒绝。

设计：闸门以「异常」表达拒绝，复用各工具节点既有的「异常 -> 降级 ok=False 不阻断」
契约，因此拒绝时主链路仍安全降级，不会抛穿。
"""
from __future__ import annotations

from app.shared.config.common import env_bool
from app.shared.runtime.action_registry import registry
from app.shared.runtime.logger import logger


class ActionDeniedError(PermissionError):
    """动作未授权（未登记 / 未启用 / 需人工确认 / 前置条件不满足）。"""


def require_action(name: str) -> None:
    """ACTION_GUARD_ENABLED=true 时校验动作授权；被拒则抛 ActionDeniedError。"""
    if not env_bool("ACTION_GUARD_ENABLED", False):
        return
    allowed, reason = registry.authorize(name)
    if not allowed:
        logger.warning(f"[Action] 动作被拒绝：{name}（{reason}）")
        raise ActionDeniedError(f"{name}: {reason}")
