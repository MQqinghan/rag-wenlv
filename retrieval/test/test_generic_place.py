# -*- coding: utf-8 -*-
"""OBS-4 离线单测：泛称识别 + 地址反解城市 + 抽取归一（mock 地理编码，零网络）。

覆盖：
  1. place_utils._is_generic_place / _city_from_address（纯共享层工具）
  2. weather_tool_service._normalize_generic_destination（治本：泛称→城市）
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.runtime.place_utils import _is_generic_place, _city_from_address

_spec = importlib.util.spec_from_file_location(
    "weather_tool_service_under_test",
    ROOT / "app" / "rag" / "tourism_query" / "weather_tool_service.py",
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

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


# ============================================================
# 1. 泛称识别
check("市区 是泛称", _is_generic_place("市区"))
check("市中心 是泛称", _is_generic_place("市中心"))
check("城区 是泛称", _is_generic_place("城区"))
check("成都市区 非泛称(具体)", not _is_generic_place("成都市区"))
check("杭州市 非泛称", not _is_generic_place("杭州市"))
check("景区 非泛称(真实地点类型)", not _is_generic_place("景区"))
check("空串 非泛称", not _is_generic_place(""))
check("带标点 市区。 仍识别为泛称", _is_generic_place("市区。"))

# ============================================================
# 2. 地址反解城市
check("四川省成都市武侯区→成都", _city_from_address("四川省成都市武侯区某某路1号") == "成都")
check("北京市朝阳区→北京", _city_from_address("北京市朝阳区") == "北京")
check("广西壮族自治区南宁市→南宁", _city_from_address("广西壮族自治区南宁市青秀区") == "南宁")
check("空地址→空串", _city_from_address("") == "")

# ============================================================
# 3. 抽取归一（治本）
# 场景1：city 已抽取 → 直接用 city
out1 = m._normalize_generic_destination({"destination": "市区", "city": "成都市", "origin": ""})
check("泛称+city → destination=成都市", out1["destination"] == "成都市")

# 场景2：city 空 + origin 可反解 → 反解城市并回填
m.amap_gateway.geocode = lambda addr: {"formatted_address": "四川省成都市东部新区"}
out2 = m._normalize_generic_destination({"destination": "市区", "city": "", "origin": "成都天府机场"})
check("泛称+origin → 反解城市=成都", out2["city"] == "成都" and out2["destination"] == "成都")

# 场景3：非泛称 → 不改动
out3 = m._normalize_generic_destination({"destination": "西湖", "city": "", "origin": "杭州"})
check("非泛称 不改动", out3["destination"] == "西湖")

# 场景4：泛称但都无解 → 保持原样（绝不盲信泛称）
m.amap_gateway.geocode = lambda addr: None
out4 = m._normalize_generic_destination({"destination": "市区", "city": "", "origin": "某无名地"})
check("泛称无解 → 保持原样", out4["destination"] == "市区")

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
