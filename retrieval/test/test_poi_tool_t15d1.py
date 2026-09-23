"""
T15-D1 单测：景点 POI 工具（poi_tool_service）+ 高德网关 POI/测距扩展。

**全程离线**：monkeypatch 掉 amap_gateway 与目的地解析，不发任何网络请求。
覆盖：
    1. 闸门：总开关关闭 / 未配 Key / 未解析目的地 → ok=False
    2. 多关键词召回：合并去重 + "市区（行政区以『区』结尾）优先"排序
    3. 字段透传：biz_ext（rating / cost / open_time）+ adname；缺失即不写（反编造）
    4. 测距：方向为"多起点 + 单终点"（高德实测约束），结果按坐标对齐回景点名
    5. 降级：高德空结果 / 抛异常 → ok=False 空简报，不抛不阻断
    6. 网关：_normalize_biz_ext 丢弃高德的空列表 []；batch_distance 参数方向与对齐
    7. state 声明：tool_poi 已在 UnifiedQueryGraphState 中声明（否则 LangGraph 静默丢弃）

运行：
    ./.venv/Scripts/python.exe test/test_poi_tool_t15d1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infra.amap_gateway import AmapGateway  # noqa: E402
from app.process.unified_query.agent.state import UnifiedQueryGraphState  # noqa: E402
from app.rag.tourism_query import poi_tool_service as svc  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, info: str = "") -> None:
    results.append((name, cond, info))


# ---------------------------------------------------------------- 测试替身
def _poi(name: str, adname: str, location: str, business=None, type_text: str = "风景名胜;国家级景点"):
    return {
        "id": f"id_{name}", "name": name, "type": type_text, "typecode": "110202",
        "address": f"{adname}{name}路1号", "location": location, "tel": None,
        "pname": "四川省", "cityname": "成都市", "adname": adname,
        "biz_ext": business or {},
    }


class FakeGateway:
    """假高德网关：记录调用参数，返回预设结果。"""

    def __init__(self, by_keyword=None, raise_on_search=False, distance_rows=None):
        self.by_keyword = by_keyword or {}
        self.raise_on_search = raise_on_search
        self.distance_rows = distance_rows
        self.search_calls: list[tuple] = []
        self.distance_calls: list[tuple] = []

    def search_poi(self, keywords, city=None, citylimit=True, offset=10, extensions="base"):
        self.search_calls.append((keywords, city, offset, extensions))
        if self.raise_on_search:
            raise RuntimeError("模拟高德异常")
        return list(self.by_keyword.get(keywords, []))

    def batch_distance(self, origins, destination, distance_type=1):
        self.distance_calls.append((list(origins), destination, distance_type))
        if self.distance_rows is None:
            return []
        return list(self.distance_rows)

    def last_outcome(self):
        return "ok"


def _patch(monkeypatch_gateway, destination="成都", city="成都市"):
    """替换服务层依赖（网关 + 目的地解析 + Key），返回原值以便还原。"""
    originals = (svc.amap_gateway, svc.extract_destination_info, svc.AMAP_API_KEY, svc.POI_TOOL_ENABLE)
    svc.amap_gateway = monkeypatch_gateway
    svc.extract_destination_info = lambda state: {"destination": destination, "city": city}
    svc.AMAP_API_KEY = "test-key"
    svc.POI_TOOL_ENABLE = True
    return originals


def _restore(originals):
    svc.amap_gateway, svc.extract_destination_info, svc.AMAP_API_KEY, svc.POI_TOOL_ENABLE = originals


def main() -> None:
    state = {"original_query": "下周一从深圳去成都玩三天", "current_date": "2026-09-10 周四"}

    # ---------- 1. 闸门 ----------
    gw = FakeGateway()
    orig = _patch(gw)
    svc.POI_TOOL_ENABLE = False
    check("总开关关闭 → ok=False 且不调用网关", svc.get_poi_brief(state)["ok"] is False and gw.search_calls == [])

    svc.POI_TOOL_ENABLE = True
    svc.AMAP_API_KEY = ""
    check("未配置 AMAP_API_KEY → ok=False", svc.get_poi_brief(state)["ok"] is False)
    svc.AMAP_API_KEY = "test-key"

    svc.extract_destination_info = lambda state: {"destination": "", "city": ""}
    check("未解析到目的地 → ok=False", svc.get_poi_brief(state)["ok"] is False)
    svc.extract_destination_info = lambda state: {"destination": "成都", "city": "成都市"}

    # ---------- 2. 多关键词合并 + 市区优先排序 ----------
    by_kw = {
        "旅游景区": [_poi("都江堰景区", "都江堰市", "103.61,31.00"), _poi("宽窄巷子景区", "青羊区", "104.05,30.66")],
        "热门景点": [_poi("宽窄巷子景区", "青羊区", "104.05,30.66"), _poi("大熊猫基地", "成华区", "104.14,30.73")],
        "名胜古迹": [_poi("文殊院", "青羊区", "104.07,30.67")],
    }
    gw2 = FakeGateway(by_keyword=by_kw)
    _restore(orig)
    orig = _patch(gw2)
    res = svc.get_poi_brief(state)
    names = res["names"]
    check("多关键词合并去重（宽窄巷子只出现一次）", names.count("宽窄巷子景区") == 1, str(names))
    check("三词全部被检索", {c[0] for c in gw2.search_calls} == {"旅游景区", "热门景点", "名胜古迹"})
    check("市区（行政区以『区』结尾）排在远郊（都江堰市）之前", names.index("都江堰景区") > names.index("大熊猫基地"), str(names))
    check("extensions=all 已传（取评分/营业时间）", all(c[3] == "all" for c in gw2.search_calls))

    # ---------- 3. 字段透传 + 缺失即不写 ----------
    biz_poi = _poi("武侯祠", "武侯区", "104.04,30.64", {"rating": "4.8", "cost": "50", "opentime": "08:30-18:30"})
    plain_poi = _poi("人民公园", "青羊区", "104.05,30.66", {"rating": "4.8"})
    gw3 = FakeGateway(by_keyword={"旅游景区": [biz_poi, plain_poi]})
    _restore(orig)
    orig = _patch(gw3)
    res3 = svc.get_poi_brief(state)
    text3 = res3["text"]
    check("透传评分/人均/开放时间", "评分 4.8" in text3 and "人均 50 元" in text3 and "开放时间 08:30-18:30" in text3)
    # 逐行判定：人民公园只有 rating，其条目行不应出现"人均/开放时间"（抬头文案不算）
    pm_line = next((ln for ln in text3.splitlines() if "人民公园" in ln), "")
    wu_line = next((ln for ln in text3.splitlines() if "武侯祠" in ln), "")
    check("缺 cost/opentime 的景点不写该字段（反编造）",
          "人均" not in pm_line and "开放时间" not in pm_line and "人均" in wu_line,
          f"pm_line={pm_line!r}")
    check("透传行政区", "武侯区" in text3 and "青羊区" in text3)
    check("简报带数据时间与『以景区实际执行为准』", "数据时间" in text3 and "以景区实际执行为准" in text3)

    # ---------- 4. 测距方向与对齐 ----------
    a, b, c = "104.14,30.73", "104.05,30.66", "104.04,30.64"
    pois = [_poi("大熊猫基地", "成华区", a), _poi("宽窄巷子景区", "青羊区", b), _poi("武侯祠", "武侯区", c)]
    rows = [
        {"origin": b, "distance_m": 20400.0, "duration_s": 3420},
        {"origin": c, "distance_m": 22000.0, "duration_s": 3720},
    ]
    gw4 = FakeGateway(by_keyword={"旅游景区": pois}, distance_rows=rows)
    _restore(orig)
    orig = _patch(gw4)
    res4 = svc.get_poi_brief(state)
    check("测距方向：多起点(其余景点) + 单终点(参照景点)",
          len(gw4.distance_calls) == 1 and gw4.distance_calls[0][0] == [b, c] and gw4.distance_calls[0][1] == a,
          str(gw4.distance_calls))
    check("距离条数与其余景点一致", len(res4["distances"]) == 2, str(res4["distances"]))
    check("距离按坐标对齐回景点名",
          {d["to"] for d in res4["distances"]} == {"宽窄巷子景区", "武侯祠"}
          and all(d["from"] == "大熊猫基地" for d in res4["distances"]))
    check("简报距离行含公里与参照说明", "距 大熊猫基地" in res4["text"] and "约 20.4 公里" in res4["text"])

    # 单景点时不测距（无参照意义）
    gw5 = FakeGateway(by_keyword={"旅游景区": [pois[0]]})
    _restore(orig)
    orig = _patch(gw5)
    res5 = svc.get_poi_brief(state)
    check("仅 1 个景点 → 不调用测距", gw5.distance_calls == [] and res5["distances"] == [] and res5["ok"])

    # ---------- 5. 降级 ----------
    gw6 = FakeGateway(by_keyword={})
    _restore(orig)
    orig = _patch(gw6)
    r6 = svc.get_poi_brief(state)
    check("高德无结果 → ok=False 空简报", r6["ok"] is False and r6["text"] == "" and r6["pois"] == [])

    gw7 = FakeGateway(raise_on_search=True)
    _restore(orig)
    orig = _patch(gw7)
    try:
        r7 = svc.get_poi_brief(state)
        check("网关抛异常 → ok=False 且不向上抛", r7["ok"] is False)
    except Exception as exc:  # noqa: BLE001
        check("网关抛异常 → ok=False 且不向上抛", False, f"异常外泄: {exc}")

    # ---------- 6. 网关层 ----------
    norm = AmapGateway._normalize_biz_ext({"rating": "4.8", "cost": [], "open_time": "08:00-18:00", "opentime2": "长文案"})
    check("cost=[] 被丢弃（高德空数据形状）", "cost" not in norm and norm.get("rating") == "4.8")
    check("open_time 优先于 opentime2", norm.get("opentime") == "08:00-18:00", str(norm))
    check("非 dict 输入返回空", AmapGateway._normalize_biz_ext(None) == {} and AmapGateway._normalize_biz_ext([]) == {})

    captured: dict = {}

    class CapGateway(AmapGateway):
        def _http_get(self, url, params):  # noqa: D102
            captured["url"] = url
            captured["params"] = params
            return {"status": "1", "results": [
                {"origin_id": "1", "dest_id": "1", "distance": "62581", "duration": "4588"},
                {"origin_id": "2", "dest_id": "1", "distance": "60708", "duration": "4211"},
            ]}

    cap = CapGateway()
    dist = cap.batch_distance(["104.041,30.647", "104.055,30.669"], "103.610529,31.003363", distance_type=1)
    check("batch_distance 端点/lng,lat 方向正确",
          captured["url"].endswith("/v3/distance")
          and captured["params"]["origins"] == "104.041,30.647|104.055,30.669"
          and captured["params"]["destination"] == "103.610529,31.003363"
          and captured["params"]["type"] == "1",
          str(captured))
    check("测距结果按 origin_id 对回坐标且单位换算正确",
          dist[0]["origin"] == "104.041,30.647" and dist[0]["distance_m"] == 62581.0 and dist[0]["duration_s"] == 4588 and len(dist) == 2,
          str(dist))

    class ErrGateway(AmapGateway):
        def _http_get(self, url, params):  # noqa: D102
            return {"status": "0", "info": "INVALID_PARAMS"}

    check("测距业务失败 → 返回 []（不抛）", ErrGateway().batch_distance(["a,b"], "c,d") == [])

    _restore(orig)

    # ---------- 7. state 声明 ----------
    annotations = getattr(UnifiedQueryGraphState, "__annotations__", {})
    check("UnifiedQueryGraphState 已声明 tool_poi", "tool_poi" in annotations)

    # ---------------------------------------------------------------- 汇总
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, info in results:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"     {info}" if info and not ok else ""))
    print(f"\n{passed}/{len(results)} 通过")
    print("全部通过 ✅" if passed == len(results) else "存在失败 ❌")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
