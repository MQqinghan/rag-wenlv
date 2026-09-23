# -*- coding: utf-8 -*-
"""离线单测：对话内行程规划引擎桥接（第二步）。

覆盖：开关读取 / 天数与日期解析 / 偏好规整 / TripRequest 默认补全 /
图片收集 / 地图构造 / 桥接 fail-open（抽值失败一律返回 None，回落原引擎）。
全部离线：不调 LLM、不联网、不连 Redis/Mongo。

运行：python test/test_trip_plan_bridge.py
"""
import os
import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(label: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[OK]   {label}")
    else:
        FAIL += 1
        print(f"[FAIL] {label}")


from app.rag.tourism_query import trip_plan_bridge as bridge  # noqa: E402


# ============================================================
# 1. 开关
# ============================================================
os.environ["PLAN_ENGINE_TRIP_PLAN"] = "false"
check("开关显式 OFF", bridge.trip_plan_engine_enabled() is False)
os.environ["PLAN_ENGINE_TRIP_PLAN"] = "true"
check("开关 ON", bridge.trip_plan_engine_enabled() is True)
os.environ.pop("PLAN_ENGINE_TRIP_PLAN", None)


# ============================================================
# 2. 天数解析
# ============================================================
check("parse_days 整数", bridge.parse_days(3) == 3)
check("parse_days 字符串数字", bridge.parse_days("5") == 5)
check("parse_days 带单位", bridge.parse_days("3天") == 3)
check("parse_days 中文数字", bridge.parse_days("两天一夜") == 2)
check("parse_days 空串", bridge.parse_days("") == 0)
check("parse_days None", bridge.parse_days(None) == 0)
check("parse_days 布尔不算数", bridge.parse_days(True) == 0)
check("parse_days 零", bridge.parse_days(0) == 0)
check("parse_days 越界 99", bridge.parse_days(99) == 0)
check("parse_days 越界 31", bridge.parse_days(31) == 0)


# ============================================================
# 3. 日期解析
# ============================================================
check("日期 ISO", bridge.normalize_date("2026-09-20") == date(2026, 9, 20))
check("日期 斜杠", bridge.normalize_date("2026/09/20") == date(2026, 9, 20))
check("日期 点号", bridge.normalize_date("2026.09.20") == date(2026, 9, 20))
check("日期 中文", bridge.normalize_date("2026年9月20日") == date(2026, 9, 20))
check("日期 空串", bridge.normalize_date("") is None)
check("日期 无法解析", bridge.normalize_date("下周三") is None)
check("日期 月日按当年补", bridge.normalize_date("10-01").month == 10)


# ============================================================
# 4. 偏好规整
# ============================================================
check("偏好 数组", bridge.normalize_preferences(["自然风光", "美食"]) == ["自然风光", "美食"])
check("偏好 顿号串", bridge.normalize_preferences("自然风光、美食") == ["自然风光", "美食"])
check("偏好 逗号串", bridge.normalize_preferences("历史, 美食") == ["历史", "美食"])
check("偏好 空串", bridge.normalize_preferences("") == [])
check("偏好 None", bridge.normalize_preferences(None) == [])
check("偏好 数组去空白", bridge.normalize_preferences([" 美食 ", ""]) == ["美食"])


# ============================================================
# 5. TripRequest 构造（默认补全）
# ============================================================
req = bridge.build_trip_request({"city": "杭州", "days": 3})
check("构造成功", req is not None)
check("城市透传", req.city == "杭州")
check("天数透传", req.travel_days == 3)
check(
    "end = start + days - 1",
    (date.fromisoformat(req.end_date) - date.fromisoformat(req.start_date)).days == 2,
)
check("默认交通", req.transportation == "公共交通")
check("默认住宿", req.accommodation == "经济型酒店")
check("origin 允许为空", req.origin == "")
check("free_text 默认空", req.free_text_input == "")

check("缺城市 → None", bridge.build_trip_request({"days": 3}) is None)
check("空 dict → None", bridge.build_trip_request({}) is None)
check("None → None", bridge.build_trip_request(None) is None)
check("城市空串 → None", bridge.build_trip_request({"city": "   "}) is None)

req2 = bridge.build_trip_request(
    {"city": "成都", "days": "两天一夜", "origin": "北京", "preferences": "美食、历史"}
)
check("中文天数解析", req2.travel_days == 2)
check("出发地透传", req2.origin == "北京")
check("偏好规整入请求", req2.preferences == ["美食", "历史"])

check("缺天数默认 3", bridge.build_trip_request({"city": "西安"}).travel_days == 3)
check("越界天数回落默认 3", bridge.build_trip_request({"city": "西安", "days": 99}).travel_days == 3)

req5 = bridge.build_trip_request({"city": "丽江", "start_date": "2026-10-01", "days": 4})
check("指定出发日透传", req5.start_date == "2026-10-01")
check("指定出发日 end 正确", req5.end_date == "2026-10-04")

req6 = bridge.build_trip_request({"city": "丽江", "start_date": "乱码", "days": 2})
check("非法日期走默认（start<end）", req6.start_date < req6.end_date)


# ============================================================
# 6. 图片收集
# ============================================================
plan_imgs = {
    "days": [
        {"attractions": [{"image_url": "u1", "photos": ["u2", "u1"]}]},
        {"attractions": [{"photos": ["u3"]}, {"image_url": ""}]},
    ]
}
check("图片去重", bridge.collect_image_urls(plan_imgs) == ["u1", "u2", "u3"])
check("空行程无图片", bridge.collect_image_urls({}) == [])
check(
    "图片限流 6",
    len(bridge.collect_image_urls(
        {"days": [{"attractions": [{"photos": [f"u{i}" for i in range(20)]}]}]}
    )) == 6,
)


# ============================================================
# 7. 地图构造
# ============================================================
plan_map = {
    "city": "杭州",
    "days": [
        {
            "day_index": 0,
            "attractions": [
                {"name": "西湖", "location": {"longitude": 120.1, "latitude": 30.2}, "address": "杭州市西湖区"},
                {"name": "无坐标景点", "location": None},
            ],
        },
        {"day_index": 1, "attractions": [{"name": "灵隐寺", "location": {"longitude": 120.09, "latitude": 30.24}}]},
    ],
}
m = bridge.build_render_map(plan_map)
check("地图 ok", m["ok"] is True)
check("地图城市", m["city"] == "杭州")
check("地图名称", m["name"] == "杭州行程")
check("地图天数", len(m["days"]) == 2)
check("无坐标点位被跳过", len(m["days"][0]["attractions"]) == 1)
check("点位名称", m["days"][0]["attractions"][0]["name"] == "西湖")
check("点位经度", m["days"][0]["attractions"][0]["lon"] == 120.1)
check("点位 kind", m["days"][0]["attractions"][0]["kind"] == "poi")
check("地图标签", m["days"][1]["label"] == "第2天")
check("空行程 ok=False", bridge.build_render_map({})["ok"] is False)
check(
    "无坐标行程 ok=False",
    bridge.build_render_map({"city": "杭州", "days": [{"day_index": 0, "attractions": []}]})["ok"] is False,
)


# ============================================================
# 8. 桥接纯逻辑 fail-open（失败必须回落原引擎）
# ============================================================
check(
    "准备：空问句 → None",
    bridge.prepare_trip_plan_request({"session_id": "", "original_query": ""}) is None,
)
check(
    "渲染：无行程文本 → None",
    bridge.finalize_trip_plan_answer({"session_id": ""}, {"trip_plan": {}}) is None,
)

_orig_extract = bridge.extract_trip_plan_values
try:
    bridge.extract_trip_plan_values = lambda q, h="": {}
    check(
        "准备：抽值空 → None（回落原引擎）",
        bridge.prepare_trip_plan_request(
            {"session_id": "", "original_query": "帮我规划一下杭州"}
        ) is None,
    )
finally:
    bridge.extract_trip_plan_values = _orig_extract


# ============================================================
# 9. process 层协调器（run_trip_plan_engine）：三条路径
# ============================================================
from app.process.unified_query.agent import trip_plan_engine as engine  # noqa: E402


def _bump(d, k):
    d[k] = d.get(k, 0) + 1
    return d


_dummy_req = bridge.build_trip_request({"city": "杭州", "days": 3})
_orig_prepare = engine.bridge.prepare_trip_plan_request
_orig_finalize = engine.bridge.finalize_trip_plan_answer
_orig_invoke = engine.invoke_structured

try:
    # a) 未凑齐目的地 → None 且不调用子图
    engine.bridge.prepare_trip_plan_request = lambda state: None
    calls = {"invoke": 0}
    engine.invoke_structured = lambda req, session_id=None: (_bump(calls, "invoke"), {"rendered_text": "X"})[1]
    check(
        "引擎：无目的地 → None（不调子图）",
        engine.run_trip_plan_engine({"session_id": ""}) is None and calls["invoke"] == 0,
    )

    # b) 子图异常 → None（回落原引擎）
    engine.bridge.prepare_trip_plan_request = lambda state: _dummy_req

    def _invoke_boom(req, session_id=None):
        raise RuntimeError("boom")

    engine.invoke_structured = _invoke_boom
    check(
        "引擎：子图异常 → None（回落原引擎）",
        engine.run_trip_plan_engine({"session_id": ""}) is None,
    )

    # c) 成功路径 → 交由 finalize 渲染并透传
    engine.invoke_structured = lambda req, session_id=None: {"rendered_text": "行程", "trip_plan": {"city": "杭州"}}
    engine.bridge.finalize_trip_plan_answer = lambda state, final: {"answer": final["rendered_text"]}
    check(
        "引擎：成功路径透传",
        engine.run_trip_plan_engine({"session_id": ""}) == {"answer": "行程"},
    )
finally:
    engine.bridge.prepare_trip_plan_request = _orig_prepare
    engine.bridge.finalize_trip_plan_answer = _orig_finalize
    engine.invoke_structured = _orig_invoke


print("=" * 48)
print(f"单测结果：{PASS} 通过 / {FAIL} 失败")
if FAIL:
    sys.exit(1)
print("全部通过")
