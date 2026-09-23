# -*- coding: utf-8 -*-
"""地名工具（共享层，供 infra / rag 共用，不引入跨层依赖）。

OBS-4 相关：泛称目的地识别 + 地址反解城市。
"""
from __future__ import annotations

import re

# 泛称目的地词表：单独作为目的地时表示「某城市的市区 / 中心」，必须依附城市上下文，
# 不能独立拿去地理编码（否则会被和风 Geo 误定位到同名异地，如「市区」→ 台湾省台南市）。
# 只收「城市中心类」泛称；景区 / 机场 / 车站 / 酒店等是真实地点类型，不在此列，避免误伤。
_GENERIC_PLACE_TOKENS = frozenset({
    "市区", "市中心", "城区", "城中心", "县城", "主城区",
})

# 只匹配「完整就是这个词」的泛称（去掉首尾空白与标点后），避免误伤「XX景区」等真实目的地。
_TOKEN_STRIP_RE = re.compile(r"[^\u4e00-\u9fa5]")

# 从高德 formatted_address 提取城市名（去「市」后缀）。
# 三段优先级（后者仅在前者不命中时触发）：
#   ① 省 / 自治区之后的「X市」（成都市 / 南宁市）
#   ② 行首直辖市「X市」（北京市）
#   ③ 兜底任意「X市」（极少数无省/市前缀的异常地址）
_CITY_AFTER_PROV_RE = re.compile(r"(?:省|自治区)([\u4e00-\u9fa5]{2,8}市)")
_CITY_AT_START_RE = re.compile(r"^([\u4e00-\u9fa5]{2,4}市)")
_CITY_ANY_RE = re.compile(r"([\u4e00-\u9fa5]{2,8}市)")


def _is_generic_place(dest: str) -> bool:
    """destination 是否为泛称（市区 / 市中心 / 城区 …）。"""
    if not dest:
        return False
    norm = _TOKEN_STRIP_RE.sub("", str(dest))
    return norm in _GENERIC_PLACE_TOKENS


def _city_from_address(addr: str) -> str:
    """从高德 formatted_address（如「四川省成都市武侯区…」）提取城市名（去「市」）。

    直辖市（北京市→北京）、自治区内城市（南宁市→南宁）均兼容；提取失败返回空串。
    """
    if not addr:
        return ""
    m = _CITY_AFTER_PROV_RE.search(addr)
    if not m:
        m = _CITY_AT_START_RE.search(addr)
    if not m:
        m = _CITY_ANY_RE.search(addr)
    return m.group(1)[:-1] if m else ""
