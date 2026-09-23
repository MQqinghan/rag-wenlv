# -*- coding: utf-8 -*-
"""H·Data 数据源地图 + 知识有效期/版本管理。

设计（对齐《框架》Data 工程维度，呼应历史踩坑：天气时效数据曾因无有效期管理误导）：
- 每个数据源登记权威地图：name / category / authority / freshness_sla / expiry_seconds / fallback / owner。
- 时效判定：is_fresh(source, fetched_at) 依据 expiry_seconds 计算数据是否过期；过期则提示回退/重新获取。
- 与检索主链路解耦：本模块只描述「数据从哪来、多新才可信」，不介入具体检索实现。
- 全部异常安全。

category: realtime(实时) / static(静态知识库) / rag_corpus(向量语料) / on_demand(按需联网)
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.shared.runtime.logger import logger


class DataCategory(str, Enum):
    REALTIME = "realtime"
    STATIC = "static"
    RAG_CORPUS = "rag_corpus"
    ON_DEMAND = "on_demand"


class DataSourceSpec(BaseModel):
    name: str
    category: DataCategory
    authority: str = ""                 # 权威来源描述（如「和风天气官方 API」）
    freshness_sla: str = ""             # 新鲜度预期（如「分钟级」）
    expiry_seconds: Optional[int] = None  # None=不过期（静态/语料按版本管理）
    versioned: bool = False             # 是否按版本管理（语料导入即新版本）
    fallback: str = ""                  # 失效/不可用时回退源
    owner: str = "data-team"


class DataSourceRegistry:
    def __init__(self):
        self._src: dict[str, DataSourceSpec] = {}

    def register(self, spec: DataSourceSpec) -> DataSourceSpec:
        self._src[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[DataSourceSpec]:
        return self._src.get(name)

    def list_all(self) -> list[DataSourceSpec]:
        return list(self._src.values())

    def is_fresh(self, name: str, fetched_at_ts: float) -> bool:
        """依据 expiry_seconds 判定数据是否仍新鲜。未知源默认新鲜（不误杀）。"""
        spec = self._src.get(name)
        if spec is None or spec.expiry_seconds is None:
            return True
        age = time.time() - float(fetched_at_ts)
        return age <= spec.expiry_seconds

    def staleness_report(self) -> list[dict]:
        """监控用：列出所有有过期策略的源及其 SLA。"""
        return [
            {"name": s.name, "category": s.category.value, "expiry_seconds": s.expiry_seconds,
             "fallback": s.fallback, "versioned": s.versioned}
            for s in self._src.values() if s.expiry_seconds is not None or s.versioned
        ]


registry = DataSourceRegistry()


def register_builtin_sources() -> None:
    builtins = [
        DataSourceSpec(name="weather_qweather", category=DataCategory.REALTIME,
                       authority="和风天气官方 API", freshness_sla="小时级",
                       expiry_seconds=6 * 3600, fallback="amap_weather", owner="infra-team"),
        DataSourceSpec(name="weather_amap", category=DataCategory.REALTIME,
                       authority="高德天气(无降水量)", freshness_sla="小时级",
                       expiry_seconds=6 * 3600, fallback="weather_qweather", owner="infra-team"),
        DataSourceSpec(name="rail_12306", category=DataCategory.REALTIME,
                       authority="12306 官方 MCP", freshness_sla="车次时刻表日级",
                       expiry_seconds=24 * 3600, fallback="amap_transit(带诚实声明)", owner="infra-team"),
        DataSourceSpec(name="amap_poi", category=DataCategory.STATIC,
                       authority="高德 POI", freshness_sla="周级",
                       expiry_seconds=7 * 24 * 3600, fallback="", owner="infra-team"),
        DataSourceSpec(name="tourism_chunks", category=DataCategory.RAG_CORPUS,
                       authority="导入文旅语料(Milvus)", freshness_sla="按导入版本",
                       expiry_seconds=None, versioned=True, owner="rag-team"),
        DataSourceSpec(name="web_search", category=DataCategory.ON_DEMAND,
                       authority="联网搜索", freshness_sla="实时",
                       expiry_seconds=3 * 24 * 3600, fallback="kb", owner="rag-team"),
    ]
    for s in builtins:
        registry.register(s)


register_builtin_sources()
