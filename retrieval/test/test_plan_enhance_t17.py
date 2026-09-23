"""
问题1~5 改造单测（2026-09-10 主人测试反馈）。

**全程离线**：monkeypatch 掉目的地解析与高德网关，不发任何网络请求。
覆盖：
    1. plan_extract 新槽位归一化（_norm_transport）与 extract_destination_info 返回新字段
    2. 交通方式确定性推荐器：阈值规则 / 用户指定优先 / 排除项过滤 / 正则兜底 / 飞机标注"估算"
    3. 花销估算器：真实数据优先 + 缺失项标记"预估" + 门票不估算 + 合计只累加有金额项
    4. 行程地图：按天分组 + 候选池命中 + 地理编码补点 + 无坐标不渲染
    5. 意图路由：跨城+安排信号；tr-008 咨询句仍不算行程
    6. 状态声明：新增字段已进 UnifiedQueryGraphState（否则 LangGraph 静默丢弃）
    7. Prompt：itinerary_out 新增三个区块可渲染

运行：
    ./.venv/Scripts/python.exe test/test_plan_enhance_t17.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.process.unified_query.agent.state import UnifiedQueryGraphState  # noqa: E402
from app.rag.common import intent_route_service as route_svc  # noqa: E402
from app.rag.tourism_query import budget_service as budget_svc  # noqa: E402
from app.rag.tourism_query import itinerary_map_service as map_svc  # noqa: E402
from app.rag.tourism_query import transport_advice_service as transport_svc  # noqa: E402
from app.rag.tourism_query import weather_tool_service as weather_svc  # noqa: E402
from app.rag.tourism_query.route_tool_service import plan_route  # noqa: E402
from app.shared.runtime.load_prompt import load_prompt  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, info: str = "") -> None:
    results.append((name, cond, info))


def _patch(mod, fake):
    """替换模块级 extract_destination_info，返回 restore 函数。"""
    original = mod.extract_destination_info
    mod.extract_destination_info = fake
    return lambda: setattr(mod, "extract_destination_info", original)


EMPTY_INFO = {
    "origin": "", "destination": "", "city": "", "latitude": None, "longitude": None,
    "travel_days": 0, "transport_preference": "", "transport_avoid": "",
}


def main() -> None:  # noqa: C901
    # ---------- 1. _norm_transport ----------
    check("_norm_transport 空值 → 空串", weather_svc._norm_transport("空") == "")
    check("_norm_transport 近义写法「航班」→ 飞机", weather_svc._norm_transport("航班") == "飞机")
    check("_norm_transport 近义写法「动车」→ 高铁", weather_svc._norm_transport("动车") == "高铁")
    check("_norm_transport 无法识别 → 空串（宁缺勿错）", weather_svc._norm_transport("走路") == "")

    # ---------- 2. 交通方式推荐器 ----------
    # 2.1 缺出发地/目的地 → 不推荐
    r = _patch(transport_svc, lambda state: dict(EMPTY_INFO))
    check("推荐器：无跨城两地 → ok=False", transport_svc.decide_transport({})["ok"] is False)
    r()

    # 2.2 长途铁路（19h）→ 推荐飞机，飞机标注估算
    info = {**EMPTY_INFO, "origin": "深圳", "destination": "三亚", "city": "三亚市"}
    r = _patch(transport_svc, lambda state: dict(info))
    st = {
        "original_query": "我下周从深圳出发去三亚玩 5 天，行程怎么安排比较好？",
        "tool_route": {"ok": True, "distance_km": 980.0, "duration_seconds": 43200, "fuel_cost": 611.0,
                       "transit_duration_seconds": 75420, "transit_summary": "火车"},
        "tool_rail": {"ok": True, "trains": [{"code": "Z8006", "duration": "19:08", "prices": {"硬座": "267.5"}}]},
    }
    t = transport_svc.decide_transport(st)
    check("推荐器：铁路 19h → 推荐飞机", t["ok"] and t["recommended"] == "飞机", str(t))
    check("推荐器：飞机方案标注为估算", any(o["mode"] == "飞机" and o["estimated"] for o in t["options"]), str(t["options"]))
    check("推荐器：文本含「推荐交通方式」", "推荐交通方式" in t["text"], t["text"])
    r()

    # 2.3 近程（150km）→ 自驾
    info2 = {**EMPTY_INFO, "origin": "成都", "destination": "都江堰", "city": "成都市"}
    r = _patch(transport_svc, lambda state: dict(info2))
    t2 = transport_svc.decide_transport({
        "original_query": "从成都去都江堰怎么玩两天？",
        "tool_route": {"ok": True, "distance_km": 150.0, "duration_seconds": 7200, "fuel_cost": 94.0},
        "tool_rail": {"ok": False},
    })
    check("推荐器：近程 150km → 自驾", t2["recommended"] == "自驾", str(t2))
    r()

    # 2.4 用户明确指定飞机 → 指定优先，且不列铁路
    info3 = {**EMPTY_INFO, "origin": "深圳", "destination": "三亚", "transport_preference": "飞机"}
    r = _patch(transport_svc, lambda state: dict(info3))
    t3 = transport_svc.decide_transport({
        "original_query": "坐飞机",
        "tool_route": {"ok": True, "distance_km": 980.0, "duration_seconds": 43200, "fuel_cost": 611.0},
        "tool_rail": {"ok": True, "trains": [{"code": "Z8006", "duration": "19:08", "prices": {}}]},
    })
    check("推荐器：用户指定飞机 → 推荐飞机且标注「用户指定」",
          t3["recommended"] == "飞机" and "用户指定" in t3["text"], t3["text"])
    r()

    # 2.5 LLM 槽位缺失时正则兜底：问题3 的真实问句
    info4 = {**EMPTY_INFO, "origin": "深圳", "destination": "三亚"}
    r = _patch(transport_svc, lambda state: dict(info4))
    t4 = transport_svc.decide_transport({
        "original_query": "深圳去三亚坐火车，你怎么想的？当然是坐飞机去啊",
        "tool_route": {"ok": True, "distance_km": 980.0, "duration_seconds": 43200, "fuel_cost": 611.0},
        "tool_rail": {"ok": True, "trains": [{"code": "Z8006", "duration": "19:08", "prices": {}}]},
    })
    check("推荐器：正则兜底识别「当然是坐飞机」→ 飞机", t4["specified"] == "飞机", str(t4))
    r()

    # 2.6 排除项过滤：avoid 自驾 → 不得出现在推荐与备选中
    info5 = {**EMPTY_INFO, "origin": "成都", "destination": "九寨沟", "transport_avoid": "自驾"}
    r = _patch(transport_svc, lambda state: dict(info5))
    t5 = transport_svc.decide_transport({
        "original_query": "不想开车去九寨沟",
        "tool_route": {"ok": True, "distance_km": 430.0, "duration_seconds": 30600, "fuel_cost": 268.0},
        "tool_rail": {"ok": True, "trains": [{"code": "C6102", "duration": "02:30", "prices": {"二等座": "110"}}]},
    })
    check("推荐器：排除自驾后不推荐自驾", t5["recommended"] != "自驾", str(t5))
    r()

    # ---------- 3. 花销估算 ----------
    info6 = {**EMPTY_INFO, "origin": "深圳", "destination": "三亚", "city": "三亚市", "travel_days": 5}
    r = _patch(budget_svc, lambda state: dict(info6))
    st_b = {
        "original_query": "深圳去三亚玩5天",
        "tool_rail": {"ok": True, "trains": [{"code": "Z8006", "duration": "19:08",
                                             "prices": {"硬座": "267.5", "硬卧": "478.5"}}]},
        "tool_route": {"ok": True, "distance_km": 980.0, "fuel_cost": 611.0},
        "tool_stay_food": {
            "ok": True,
            "hotels": [{"name": "海景酒店", "biz_ext": {"cost": "300"}}, {"name": "民宿A", "biz_ext": {"cost": "500"}}],
            "foods": [{"name": "海鲜餐厅", "biz_ext": {"cost": "50"}}],
        },
    }
    b = budget_svc.estimate_budget(st_b, transport={"recommended": "飞机"})
    items = {i["name"]: i for i in b["items"]}
    check("估算：5 天 → 4 晚", b["days"] == 5 and b["nights"] == 4, str(b["days"]))
    check("估算：飞机交通项不给假数字（写以航司为准）", "航司" in items["交通"]["text"], items["交通"]["text"])
    check("估算：住宿用高德人均中位数 400 × 4 晚 = 1600",
          items["住宿"].get("amount") == 1600.0 and items["住宿"]["estimated"] is False, str(items["住宿"]))
    check("估算：餐饮 50×3×5 = 750", items["餐饮"].get("amount") == 750.0, str(items["餐饮"]))
    check("估算：门票不估算（写以景区为准）", "以景区实际收费为准" in items["门票"]["text"] and "amount" not in items["门票"])
    check("估算：合计=1600+750=2350", b["total"] == 2350.0, str(b["total"]))
    check("估算：文本含来源标注", "花销估算" in b["text"] and "来源" in b["text"])

    # 无高德数据 → 走默认单价并标注"预估"
    b2 = budget_svc.estimate_budget(
        {"original_query": "深圳去三亚玩5天", "tool_route": {"ok": False}, "tool_rail": {"ok": False},
         "tool_stay_food": {"ok": False}},
        transport={"recommended": "自驾"},
    )
    i2 = {i["name"]: i for i in b2["items"]}
    check("估算：无高德住宿数据 → 标记「预估」", i2["住宿"]["estimated"] is True and "预估" in i2["住宿"]["text"])
    check("估算：无高德餐饮数据 → 标记「预估」", i2["餐饮"]["estimated"] is True and "预估" in i2["餐饮"]["text"])
    r()

    # ---------- 4. 行程地图 ----------
    info7 = {**EMPTY_INFO, "origin": "深圳", "destination": "杭州", "city": "杭州市"}
    r = _patch(map_svc, lambda state: dict(info7))
    orig_geo = map_svc.amap_gateway.geocode
    map_svc.amap_gateway.geocode = lambda name, city=None: {"longitude": 120.1, "latitude": 30.2, "formatted_address": "杭州市"}
    st_m = {
        "tool_poi": {"ok": True, "pois": [
            {"name": "西湖", "location": "120.14,30.25", "address": "西湖区"},
            {"name": "灵隐寺", "location": "120.10,30.24", "address": "西湖区"},
        ]},
        "tool_stay_food": {"ok": True, "hotels": []},
    }
    text = "第1天（2026-09-11）：抵达\n上午：游览西湖\n下午：参观灵隐寺\n第2天（2026-09-12）：周边\n上午：前往西溪湿地"
    m = map_svc.build_itinerary_map(st_m, text)
    names = [a["name"] for d in m["days"] for a in d["attractions"]]
    check("地图：按天分组为 2 天", m["ok"] and len(m["days"]) == 2, str(m["days"]))
    check("地图：候选池命中西湖/灵隐寺", "西湖" in names and "灵隐寺" in names, str(names))
    check("地图：候选池未命中者走地理编码补点", "西溪湿地" in names, str(names))
    check("地图：命中项坐标取自高德 POI",
          any(a["name"] == "西湖" and a["lon"] == 120.14 for d in m["days"] for a in d["attractions"]))
    # 概览行里的日期区间不得被误当成"天"切分（否则会切出一堆只有概览文字的空天）
    text2 = ("行程概览\n出行日期：2026-09-11→2026-09-15\n"
             "第1天（2026-09-11）：抵达\n上午：游览西湖\n第2天（2026-09-12）：周边\n上午：参观灵隐寺")
    m2 = map_svc.build_itinerary_map(st_m, text2)
    check("地图：概览日期区间不产生额外分组", m2["ok"] and len(m2["days"]) == 2, str(m2["days"]))
    check("地图：无坐标景点不渲染（空 POI → ok=False）",
          map_svc.build_itinerary_map({"tool_poi": {"ok": False}, "tool_stay_food": {}}, "第1天：随便走走")["ok"] is False)
    # 地名可信度过滤（2026-09-10 补充）：从句片段送去地理编码会落到错误坐标
    # （实测"周边的轮渡码头"被高德编到十几公里外的翔安区），故一律丢弃——错误标记比不标更糟。
    check("地图：过滤从句片段「周边的轮渡码头」", not map_svc._is_place_like("周边的轮渡码头"))
    check("地图：过滤从句片段「鼓浪屿的轮渡信息」", not map_svc._is_place_like("鼓浪屿的轮渡信息"))
    check("地图：过滤动宾短语「环岛路骑行或散步」", not map_svc._is_place_like("环岛路骑行或散步"))
    check("地图：过滤章节名「交通与市内行程」", not map_svc._is_place_like("交通与市内行程"))
    check("地图：超长描述片段不作为地名", not map_svc._is_place_like("这是一个非常长的描述性片段文字"))
    check("地图：正常地名保留（西溪湿地/曾厝垵/鼓浪屿风景名胜区/中山路步行街）",
          map_svc._is_place_like("西溪湿地") and map_svc._is_place_like("曾厝垵")
          and map_svc._is_place_like("鼓浪屿风景名胜区") and map_svc._is_place_like("中山路步行街"))
    # 从句边界截断：保住真实地名前缀（比整段丢弃覆盖率更高），且不误伤功能字起头的地名
    check("地图：截断「鼓浪屿需乘轮渡」→ 鼓浪屿", map_svc._truncate_at_clause("鼓浪屿需乘轮渡") == "鼓浪屿")
    check("地图：截断「环岛路骑行或散步」→ 环岛路", map_svc._truncate_at_clause("环岛路骑行或散步") == "环岛路")
    check("地图：截断「鼓浪屿的轮渡信息」→ 鼓浪屿", map_svc._truncate_at_clause("鼓浪屿的轮渡信息") == "鼓浪屿")
    check("地图：截断「周边的轮渡码头」→ 周边（随后被尾缀清洗丢弃）",
          map_svc._truncate_at_clause("周边的轮渡码头") == "周边")
    check("地图：前 2 字保护不误伤「可可托海」", map_svc._truncate_at_clause("可可托海") == "可可托海")
    check("地图：从句片段抽取后得到干净地名（前往鼓浪屿需乘轮渡 → 鼓浪屿）",
          map_svc._extract_place_names("上午：前往鼓浪屿需乘轮渡") == ["鼓浪屿"],
          str(map_svc._extract_place_names("上午：前往鼓浪屿需乘轮渡")))
    geocoded: list[str] = []
    map_svc.amap_gateway.geocode = lambda name, city=None: (
        geocoded.append(name) or {"longitude": 120.1, "latitude": 30.2, "formatted_address": "杭州市"}
    )
    text3 = "第1天：抵达\n上午：游览西湖\n下午：前往周边的轮渡码头\n晚上：漫步环岛路骑行或散步"
    m3 = map_svc.build_itinerary_map(st_m, text3)
    check("地图：从句片段不触发地理编码",
          "周边的轮渡码头" not in geocoded and "环岛路骑行或散步" not in geocoded, str(geocoded))
    check("地图：过滤后仍保留真实点位（西湖）",
          m3["ok"] and any(a["name"] == "西湖" for d in m3["days"] for a in d["attractions"]), str(m3))
    map_svc.amap_gateway.geocode = orig_geo
    r()

    # ---------- 5. 意图路由信号（问题4） ----------
    check("路由：tr-008 单地名咨询句不算行程",
          route_svc._route_plan_signal_hit("第一次去杭州，景点应该怎么安排？") is False)
    check("路由：跨城+安排 → 判为行程",
          route_svc._route_plan_signal_hit("从深圳去杭州，景点应该怎么安排？") is True)
    check("路由：tr-003 转场咨询句不算行程",
          route_svc._route_plan_signal_hit("从昆明到大理再到丽江，城市之间怎么转场最方便？") is False)

    # ---------- 6. 状态声明 ----------
    annotations = getattr(UnifiedQueryGraphState, "__annotations__", {})
    for field in ("tool_stay_food", "tool_transport", "tool_budget", "itinerary_map", "map_data"):
        check(f"UnifiedQueryGraphState 已声明 {field}", field in annotations)

    # ---------- 7. Prompt 渲染 ----------
    prompt = load_prompt(
        "tourism/itinerary_out",
        current_date="2026-09-10 周四", weather="w", route="r", rail="l", poi="p",
        stay_food="s", transport="t", budget="b", context="c", history="h",
        item_names="i", question="q", rewritten="rw", consult_instruction="",
    )
    for block in ("【交通方式建议】", "【花销估算参考】", "【住宿餐饮参考】"):
        check(f"itinerary_out 渲染含 {block}", block in prompt)

    # ---------- 8. 路线工具结构化字段 ----------
    empty = plan_route({})  # 无 Key/无目的地 → 直接空结构（不发网络请求）
    check("路线工具空结构含新增结构化字段",
          all(k in empty for k in ("distance_km", "fuel_cost", "transit_duration_seconds")), str(empty))

    # ---------------------------------------------------------------- 汇总
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, info in results:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"     {info}" if info and not ok else ""))
    print(f"\n{passed}/{len(results)} 通过")
    print("全部通过 ✅" if passed == len(results) else "存在失败 ❌")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
