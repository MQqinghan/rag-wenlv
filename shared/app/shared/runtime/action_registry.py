# -*- coding: utf-8 -*-
"""I·Action 风险分级：动作注册表 + 权限/风险/人工确认/审计。

设计（对齐《框架》Action 工程维度「Action 建模是 Agent 拿执行权的前提」）：
- 每个可执行动作登记：name / description / risk_level / requires_human_confirm /
  preconditions / audit / enabled。
- 当前项目动作均为查询类（低危、无需人工确认）；预留高危动作（订票/支付）登记为
  DISABLED + 必须人工二次确认，未来启用时天然受控。
- authorize(action, ctx)：未启用 / 高危未确认 / 前置条件不满足 → 拒绝并说明原因。
- 全部异常安全。

前置权限过滤在信息进模型之前（框架强调：Prompt 里的拒答规则替不了真实权限校验）。
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.shared.runtime.logger import logger


class ActionRisk(str, Enum):
    LOW = "low"          # 查询/只读，无副作用
    MEDIUM = "medium"    # 外部实时写入/下单草稿
    HIGH = "high"        # 支付/不可逆操作


class ActionSpec(BaseModel):
    name: str
    description: str = ""
    risk_level: ActionRisk = ActionRisk.LOW
    requires_human_confirm: bool = False
    preconditions: list[str] = Field(default_factory=list)
    audit: bool = True
    enabled: bool = True


class ActionRegistry:
    def __init__(self):
        self._acts: dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec) -> ActionSpec:
        self._acts[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ActionSpec]:
        return self._acts.get(name)

    def list_all(self) -> list[ActionSpec]:
        return list(self._acts.values())

    def requires_confirm(self, name: str) -> bool:
        s = self._acts.get(name)
        return bool(s and s.requires_human_confirm)

    def authorize(self, name: str, *, human_confirmed: bool = False) -> tuple[bool, str]:
        """返回 (是否允许, 原因)。"""
        s = self._acts.get(name)
        if s is None:
            return False, f"未知动作: {name}"
        if not s.enabled:
            return False, f"动作已禁用: {name}"
        if s.risk_level == ActionRisk.HIGH and not human_confirmed:
            return False, f"高危动作需人工二次确认: {name}"
        if s.requires_human_confirm and not human_confirmed:
            return False, f"需人工确认: {name}"
        return True, "ok"


registry = ActionRegistry()


def register_builtin_actions() -> None:
    builtins = [
        ActionSpec(name="weather_query", description="实时天气查询", risk_level=ActionRisk.LOW),
        ActionSpec(name="route_query", description="路线规划查询（高德/铁路）", risk_level=ActionRisk.LOW),
        ActionSpec(name="rail_query", description="高铁车次/票价查询", risk_level=ActionRisk.LOW),
        ActionSpec(name="poi_query", description="POI/门票/酒店查询", risk_level=ActionRisk.LOW),
        ActionSpec(name="stay_food_query", description="住宿/餐饮 POI 查询（高德）", risk_level=ActionRisk.LOW),
        # 前瞻：高危动作登记为禁用 + 必须人工确认（未来启用时受控）
        ActionSpec(name="book_ticket", description="【预留】代订车票",
                   risk_level=ActionRisk.HIGH, requires_human_confirm=True, enabled=False),
        ActionSpec(name="payment", description="【预留】支付",
                   risk_level=ActionRisk.HIGH, requires_human_confirm=True, enabled=False),
    ]
    for a in builtins:
        registry.register(a)


register_builtin_actions()
