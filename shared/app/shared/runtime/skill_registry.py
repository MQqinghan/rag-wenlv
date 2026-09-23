# -*- coding: utf-8 -*-
"""G·Skill 封装升级：轻量 Registry + 12 要素 Skill 描述模型。

设计（对齐《框架》Skill 工程维度与主人「可管理可迁移」初衷）：
- 每个能力（trip_plan / 意图路由 / 天气 / 铁路 / POI / 文旅 / 闲聊）登记为一条 SkillSpec，
  含 12 要素：name / version / description / domain / triggers / inputs / outputs /
  dependencies / permissions / risk_level / sla / owner / eval_hook / rollback。
- 与「运行时代码」解耦：Registry 只是能力目录（元数据 + 调用入口引用），不搬运实现。
- 复用率 / 版本 / 风险可回溯：任何能力的启用、版本、风险评估都在这里一眼可见。
- 全部异常安全：注册/查询失败不影响主链路。

后续可扩展：从本 Registry 自动生成用户级 Skill 卡片（~/.workbuddy/skills/...），实现「可迁移」。
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.shared.runtime.logger import logger


class RiskLevel(str, Enum):
    LOW = "low"        # 纯查询/只读，无副作用
    MEDIUM = "medium"  # 依赖外部实时数据，偶发不准
    HIGH = "high"      # 涉及下单/支付/写操作（当前项目暂无，预留）


class SkillSpec(BaseModel):
    """12 要素 Skill 描述。"""
    name: str
    version: str = "0.1.0"
    description: str = ""
    domain: str = ""                       # tourism / chitchat / cross / system
    triggers: list[str] = Field(default_factory=list)   # 触发句式的语义描述
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # 依赖的其他 skill / 网关
    permissions: list[str] = Field(default_factory=list)  # 所需权限（当前多为空）
    risk_level: RiskLevel = RiskLevel.LOW
    sla: str = ""                          # 时延/可用性预期
    owner: str = "team"
    eval_hook: Optional[str] = None        # 关联评测集/用例 id
    rollback: str = "feature-flag"         # 回滚方式
    enabled: bool = True
    entry: Optional[str] = None            # 代码入口引用（模块.函数），仅供参考


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> SkillSpec:
        if spec.name in self._skills:
            logger.warning(f"[SkillRegistry] 覆盖已存在 skill: {spec.name}")
        self._skills[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[SkillSpec]:
        return self._skills.get(name)

    def list_all(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def enabled(self) -> list[SkillSpec]:
        return [s for s in self._skills.values() if s.enabled]

    def by_domain(self, domain: str) -> list[SkillSpec]:
        return [s for s in self._skills.values() if s.domain == domain]

    def as_catalog_json(self) -> str:
        import json
        return json.dumps(
            [s.model_dump() for s in self._skills.values()],
            ensure_ascii=False, indent=2,
        )


# 进程内单例
registry = SkillRegistry()


def register_builtin_skills() -> None:
    """登记本项目内置能力（12 要素）。幂等，可重复调用。"""
    builtins = [
        SkillSpec(
            name="trip_plan", version="1.0.0", domain="tourism",
            description="端到端行程规划：目的地市内行程 + 出发地往返交通（高德/铁路真实数据）",
            triggers=["规划", "行程", "攻略", "几天", "路线", "怎么安排"],
            inputs=["city", "origin(可选)", "travel_days", "transportation", "start_date", "end_date"],
            outputs=["每日行程", "往返交通(RouteInfo)", "酒店", "天气", "POI"],
            dependencies=["weather_gateway", "amap_gateway", "rail_gateway", "poi_gateway"],
            risk_level=RiskLevel.LOW, sla="检索+LLM 约 10-30s", owner="rag-team",
            eval_hook="eval_cases(is_plan)", rollback="feature-flag(INTENT/TRIP 开关)",
            entry="app.process.trip_plan.agent.main_graph.trip_plan_app",
        ),
        SkillSpec(
            name="intent_route", version="2.0.0", domain="system",
            description="意图路由：LLM 主导单次多路决策（domain/改写/source_policy/model_tier/tools/is_plan）",
            triggers=["任意用户问句"],
            inputs=["query", "history_text", "session_id"],
            outputs=["domain", "rewritten", "model_tier", "tools", "is_plan"],
            dependencies=["llm_provider", "llm_reliability"],
            risk_level=RiskLevel.LOW, sla="<2s（命中缓存<100ms）", owner="rag-team",
            eval_hook="eval_cases(全40/50题 Acc>=1.0)", rollback="feature-flag(INTENT_ROUTE_LLM_FIRST=False)",
            entry="app.rag.common.intent_route_llm.route_intent",
        ),
        SkillSpec(
            name="weather_query", version="1.0.0", domain="tourism",
            description="实时天气查询（和风主/高德备，自动降级）",
            triggers=["天气", "下雨", "台风", "适合出行吗"],
            inputs=["city", "date"], outputs=["逐日天气/温度/降水"],
            dependencies=["weather_gateway"], risk_level=RiskLevel.MEDIUM,
            sla="<1s", owner="infra-team", eval_hook=None, rollback="降级为无天气",
        ),
        SkillSpec(
            name="rail_query", version="1.0.0", domain="tourism",
            description="真实高铁车次/耗时/票价（12306-MCP，仅支持提前14天）",
            triggers=["高铁", "火车", "车次", "票价"],
            inputs=["origin", "destination", "travel_date"], outputs=["车次/耗时/各席别票价"],
            dependencies=["rail_gateway"], risk_level=RiskLevel.MEDIUM,
            sla="<3s", owner="infra-team", eval_hook=None,
            rollback="回退高德 transit（带诚实声明）",
        ),
        SkillSpec(
            name="tourism_qa", version="1.0.0", domain="tourism",
            description="文旅域问答（文化/景点/攻略，知识型兜底走本地库）",
            triggers=["景点", "文化", "攻略", "美食", "非遗"],
            inputs=["query", "history"], outputs=["答案", "出处"],
            dependencies=["milvus", "bge_m3", "reranker"], risk_level=RiskLevel.LOW,
            sla="<3s", owner="rag-team", eval_hook="eval_cases(tourism)", rollback="feature-flag",
        ),
        SkillSpec(
            name="chitchat", version="1.0.0", domain="chitchat",
            description="闲聊/系统交互（问候/感谢/身份）",
            triggers=["你好", "谢谢", "你是谁"],
            inputs=["query"], outputs=["寒暄回复"],
            dependencies=[], risk_level=RiskLevel.LOW, sla="<1s", owner="rag-team",
            eval_hook="eval_cases(chitchat)", rollback="feature-flag",
        ),
    ]
    for s in builtins:
        registry.register(s)


# 模块导入即登记（幂等）
register_builtin_skills()
