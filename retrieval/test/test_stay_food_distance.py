# -*- coding: utf-8 -*-
"""② 住宿/餐饮候选「距景点约 X 公里」距离标注 —— 离线单测（零外部请求）。

覆盖：
  1. _format_km：距离中文表述
  2. _resolve_scenic_anchor：参照景点解析（复用 tool_poi / 召回 top1 / 失败）
  3. _annotate_distances：逐条标注（只标注不排序）、开关、测距失败
  4. _poi_segment / build_stay_food_text：渲染
  5. get_stay_food_brief：端到端（全 mock）
  6. 软失败：测距异常不崩、候选不丢
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.tourism_query import stay_food_tool_service as sf

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
# 1. _format_km
# ============================================================
print("== 1. _format_km ==")
check("1500m → 约 1.5 公里", sf._format_km(1500) == "约 1.5 公里")
check("500m → 约 500 米", sf._format_km(500) == "约 500 米")
check("1000m → 约 1.0 公里", sf._format_km(1000) == "约 1.0 公里")
check("0m → 约 0 米", sf._format_km(0) == "约 0 米")

# ============================================================
# 2. _resolve_scenic_anchor
# ============================================================
print("== 2. _resolve_scenic_anchor ==")
with mock.patch.object(sf.amap_gateway, "search_poi", return_value=[]) as sp:
    _name, _loc = sf._resolve_scenic_anchor(
        {"tool_poi": {"ok": True, "anchor": "宽窄巷子", "pois": [{"name": "宽窄巷子", "location": "104.0,30.6"}]}},
        "成都",
    )
    check("复用 tool_poi anchor", (_name, _loc) == ("宽窄巷子", "104.0,30.6"))
    check("复用时不调 search_poi", sp.call_count == 0)

with mock.patch.object(
    sf.amap_gateway, "search_poi", return_value=[{"name": "武侯祠", "location": "104.1,30.6"}]
) as sp:
    _name, _loc = sf._resolve_scenic_anchor({}, "成都")
    check("无 tool_poi → 召回 top1", (_name, _loc) == ("武侯祠", "104.1,30.6"))
    check("召回被调用一次", sp.call_count == 1)

with mock.patch.object(sf.amap_gateway, "search_poi", return_value=[]):
    check("召回为空 → ('','')", sf._resolve_scenic_anchor({}, "成都") == ("", ""))

# ============================================================
# 3. _annotate_distances
# ============================================================
print("== 3. _annotate_distances ==")
with mock.patch.object(sf, "STAY_FOOD_DISTANCE_ENABLE", False):
    check("开关 OFF → 返回空", sf._annotate_distances({}, [{"name": "A", "location": "1,1"}], "成都") == "")

with mock.patch.object(
    sf.amap_gateway, "search_poi", return_value=[{"name": "景点X", "location": "0,0"}]
), mock.patch.object(
    sf.amap_gateway,
    "batch_distance",
    return_value=[
        {"origin": "1,1", "distance_m": 1500, "duration_s": 300},
        {"origin": "2,2", "distance_m": 800, "duration_s": 120},
    ],
):
    pois = [{"name": "A酒店", "location": "1,1"}, {"name": "B酒店", "location": "2,2"}, {"name": "C酒店"}]
    anchor = sf._annotate_distances({}, pois, "成都")
    check("返回参照景点名", anchor == "景点X")
    check("A 标注 1.5 公里", pois[0].get("_distance_text") == "距景点X约 1.5 公里")
    check("B 标注 800 米", pois[1].get("_distance_text") == "距景点X约 800 米")
    check("无坐标候选不标注", "_distance_text" not in pois[2])
    check("顺序未被改变（只标注不排序）", [p["name"] for p in pois] == ["A酒店", "B酒店", "C酒店"])

with mock.patch.object(
    sf.amap_gateway, "search_poi", return_value=[{"name": "景点X", "location": "0,0"}]
), mock.patch.object(sf.amap_gateway, "batch_distance", return_value=[]):
    pois3 = [{"name": "A酒店", "location": "1,1"}]
    check("测距失败 → 返回空且不标注", sf._annotate_distances({}, pois3, "成都") == "" and "_distance_text" not in pois3[0])

with mock.patch.object(sf.amap_gateway, "search_poi", return_value=[]):
    check("无参照景点 → 返回空", sf._annotate_distances({}, [{"name": "A", "location": "1,1"}], "成都") == "")

# ============================================================
# 4. 渲染
# ============================================================
print("== 4. 渲染 ==")
seg = sf._poi_segment(
    {
        "name": "A酒店",
        "type": "住宿服务;宾馆酒店",
        "adname": "青羊区",
        "biz_ext": {"rating": "4.5"},
        "_distance_text": "距宽窄巷子约 1.2 公里",
    }
)
check("segment 含距离标注", "距宽窄巷子约 1.2 公里" in seg)
check("segment 仍含评分", "评分 4.5" in seg)
txt = sf.build_stay_food_text(
    {
        "destination": "成都",
        "fetched_at": "2026-09-10 21:00",
        "hotels": [{"name": "A酒店", "_distance_text": "距宽窄巷子约 1.2 公里"}],
        "foods": [],
    }
)
check("简报含距离标注", "距宽窄巷子约 1.2 公里" in txt)
check("简报含目的地", "目的地 成都" in txt)

# ============================================================
# 5. get_stay_food_brief 端到端（全 mock）
# ============================================================
print("== 5. get_stay_food_brief ==")
with mock.patch.object(sf, "AMAP_API_KEY", "fake"), mock.patch.object(
    sf, "extract_destination_info", return_value={"destination": "成都", "city": "成都"}
), mock.patch.object(
    sf.amap_gateway,
    "search_poi",
    side_effect=lambda kw, **k: [{"name": f"{kw}1", "location": "1,1", "biz_ext": {}}],
), mock.patch.object(
    sf.amap_gateway,
    "batch_distance",
    return_value=[{"origin": "1,1", "distance_m": 2000, "duration_s": 400}],
):
    r = sf.get_stay_food_brief({})
    check("ok=True", r["ok"] is True)
    check("result 含 anchor", bool(r.get("anchor")))
    check("text 含「距…公里」标注", "距" in r["text"] and "公里" in r["text"])
    check("酒店候选保留", len(r["hotels"]) > 0)

# ============================================================
# 6. 软失败
# ============================================================
print("== 6. 软失败 ==")
with mock.patch.object(sf, "AMAP_API_KEY", "fake"), mock.patch.object(
    sf, "extract_destination_info", return_value={"destination": "成都", "city": "成都"}
), mock.patch.object(
    sf.amap_gateway,
    "search_poi",
    side_effect=lambda kw, **k: [{"name": f"{kw}1", "location": "1,1", "biz_ext": {}}],
), mock.patch.object(sf.amap_gateway, "batch_distance", side_effect=RuntimeError("boom")):
    r2 = sf.get_stay_food_brief({})
    check("测距异常 → 不崩且候选保留", r2["ok"] is True and len(r2["hotels"]) > 0)
    check("测距异常 → anchor 空", r2.get("anchor") == "")

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
