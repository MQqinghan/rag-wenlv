"""
T14 铁路网关 + 工具 + 白名单剥离 单测（零 LLM、零联网、零 MCP 服务）。

覆盖：
    1. 字段口径坑：票价端点车次号取 train_code（其 train_no 是内部哈希）
    2. query_trains 正常路径（含 codes 白名单来源）
    3. 票价端点失败 → 退化到 query-tickets（有车次无票价，price_available=False）
    4. 两端均失败 / 参数缺失 / 开关关闭 → ok=False 且 error 留痕（不冒充"无车次"）
    5. 缓存"同一对象引用"防污染：改写返回值不影响缓存条目
    6. infer_travel_date 五档推断（显式年月日 / 月日 / 相对词 / 周几 / 兜底明天）
    7. build_rail_text：含真实票价与「以 12306 官方为准」；无票价时零数字
    8. _strip_flight_train_numbers 白名单：真实车次保留、编造车次剥离
    9. _rail_keep_codes：ok=False / 缺字段时为空集
   10. state 已声明 tool_rail（LangGraph 未声明键会被静默丢弃）

运行：
    ./.venv/Scripts/python.exe test/test_rail_gateway_t14.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infra import rail_gateway as rg  # noqa: E402
from app.process.unified_query.agent.state import UnifiedQueryGraphState  # noqa: E402
from app.rag.tourism_query import rail_tool_service as rts  # noqa: E402
from app.rag.tourism_query.itinerary_service import (  # noqa: E402
    _rail_keep_codes,
    _strip_flight_train_numbers,
)

_PRICE_ROW = {
    "train_no": "6i000G294203",  # 内部哈希：绝不是车次号！
    "train_code": "G2942",
    "from_station": "深圳北",
    "to_station": "成都东",
    "start_time": "07:03",
    "arrive_time": "14:46",
    "duration": "07:43",
    "train_class_name": "高速",
    "prices": {"二等座": "813.5", "一等座": "1281.5", "商务座": "2717.5"},
}
_TICKET_ROW = {
    "train_no": "G2942",  # 余票端点车次号在 train_no
    "from_station": "深圳北",
    "to_station": "成都东",
    "start_time": "07:03",
    "arrive_time": "14:46",
    "duration": "07:43",
    "seats": {"second_class": "有"},
}


def _patch_mcp(handler):
    """替换 rail_gateway._call_mcp，返回记录调用的闭包。"""
    calls: list[tuple[str, dict]] = []

    def fake(tool: str, arguments: dict) -> dict:
        calls.append((tool, arguments))
        return handler(tool, arguments)

    rg._call_mcp = fake
    return calls


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    # 1. 归一化：车次号必须取 train_code，不能取 train_no（内部哈希）
    row = rg._normalize_price_row(_PRICE_ROW)
    check("归一化取 train_code 而非 train_no", row["code"] == "G2942", f"code={row['code']!r}")
    check("归一化保留票价", row["prices"].get("二等座") == "813.5", str(row["prices"]))
    check("余票端点取 train_no", rg._normalize_ticket_row(_TICKET_ROW)["code"] == "G2942", "")

    # 2. 正常路径
    _patch_mcp(lambda tool, args: {"success": True, "data": [_PRICE_ROW]} if tool == "query-ticket-price" else {"success": True, "trains": [_TICKET_ROW]})
    ok = rg.query_trains("深圳", "成都", "2026-09-11")
    check("正常路径 ok=True", ok["ok"] is True, str(ok.get("error")))
    check("正常路径 codes 正确", ok["codes"] == ["G2942"], str(ok["codes"]))
    check("正常路径 price_available=True", ok["price_available"] is True, "")
    check("正常路径带出票来源", "12306" in ok["source"], ok["source"])

    # 5. 缓存对象防污染：改写上次返回值后重新查询，票价不应被污染
    ok["trains"][0]["prices"]["二等座"] = "0"
    ok_again = rg.query_trains("深圳", "成都", "2026-09-11")
    check("缓存对象防污染(深拷贝)", ok_again["trains"][0]["prices"].get("二等座") == "813.5",
          str(ok_again["trains"][0]["prices"]))

    # 3. 票价端点失败 → 退化到余票端点（有车次无票价）
    calls = _patch_mcp(
        lambda tool, args: {"success": False, "error": "boom"}
        if tool == "query-ticket-price"
        else {"success": True, "trains": [_TICKET_ROW]}
    )
    deg = rg.query_trains("广州", "成都", "2026-09-12")
    check("票价失败退化为余票端点", deg["ok"] is True and deg["codes"] == ["G2942"], str(deg.get("error")))
    check("退化后 price_available=False", deg["price_available"] is False, "")
    check("退化确实调了两个端点", [c[0] for c in calls] == ["query-ticket-price", "query-tickets"], str([c[0] for c in calls]))

    # 4. 两端均失败 / 参数缺失 / 开关关闭
    _patch_mcp(lambda tool, args: {"success": False, "error": "车站名称无效"})
    fail = rg.query_trains("不存在", "成都", "2026-09-13")
    check("两端失败 ok=False", fail["ok"] is False, "")
    check("失败保留错误原因(不冒充无车次)", "车站名称无效" in fail["error"], fail["error"])

    _patch_mcp(lambda tool, args: {"success": True, "data": [_PRICE_ROW]})
    check("缺出发地 → ok=False", rg.query_trains("", "成都", "2026-09-14")["error"] == "missing_params", "")
    check("缺日期 → ok=False", rg.query_trains("深圳", "成都", "")["error"] == "missing_params", "")

    _orig_enable = rg.RAIL_TOOL_ENABLE
    rg.RAIL_TOOL_ENABLE = False
    disabled = rg.query_trains("深圳", "成都", "2026-09-11")
    rg.RAIL_TOOL_ENABLE = _orig_enable
    check("开关关闭 → error=disabled", disabled["error"] == "disabled", disabled["error"])

    # 6. 日期推断（以 2026-09-10 周四 为今天）
    base = {"current_date": "2026-09-10 周四"}
    cases = [
        ("明天去成都怎么走", "2026-09-11", False),
        ("大后天出发", "2026-09-13", False),
        ("9月20日出发", "2026-09-20", False),
        ("2026-10-01出发去成都", "2026-10-01", False),
        ("下周一出发", "2026-09-14", False),
        ("周六出发", "2026-09-12", False),
        ("去成都玩几天", "2026-09-11", True),
    ]
    for question, expect_date, expect_assumed in cases:
        got, assumed = rts.infer_travel_date({**base, "original_query": question})
        check(f"日期推断[{question}]→{expect_date}", got == expect_date and assumed is expect_assumed,
              f"got={got} assumed={assumed}")

    # 7. 简报文本
    result = rg.query_trains("深圳", "成都", "2026-09-11")
    text = rts.build_rail_text(result, date_assumed=False)
    check("简报含真实车次", "G2942" in text, "")
    check("简报含真实票价", "813.5" in text, "")
    check("简报含官方口径限定语", "12306 官方为准" in text or "12306官方为准" in text, "")
    no_price = rts.build_rail_text(
        {"date": "2026-09-11", "source": rg.SOURCE_TAG, "price_available": False,
         "trains": [rg._normalize_ticket_row(_TICKET_ROW)]},
        date_assumed=True,
    )
    check("无票价简报零数字票价(反编造)", "813.5" not in no_price and "元" not in no_price, no_price.replace("\n", " | "))
    check("兜底日期简报显式声明", "未从问题中识别到明确出行日期" in no_price, "")

    # 8. 白名单剥离
    with_code = "第一天乘坐G2942次高铁前往成都，返程可乘G9999。"
    kept = _strip_flight_train_numbers(with_code, keep_codes={"G2942"})
    check("白名单保留真实车次", "G2942" in kept, kept)
    check("白名单仍剥离编造车次", "G9999" not in kept, kept)
    check("保留车次时不清掉搭配词", "乘坐G2942次高铁" in kept, kept)
    check("无白名单时照旧全剥离",
          "G2942" not in _strip_flight_train_numbers(with_code), "")

    # 9. _rail_keep_codes
    check("keep_codes: ok=False → 空集", _rail_keep_codes({"tool_rail": {"ok": False, "codes": ["G1"]}}) == set(), "")
    check("keep_codes: 正常取值", _rail_keep_codes({"tool_rail": {"ok": True, "codes": ["G2942", "Z332"]}}) == {"G2942", "Z332"}, "")
    check("keep_codes: 缺字段不报错", _rail_keep_codes({}) == set(), "")

    # 10. state 声明（未声明的键会被 LangGraph 静默丢弃）
    check("UnifiedQueryGraphState 已声明 tool_rail", "tool_rail" in UnifiedQueryGraphState.__annotations__, "")

    # 输出
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if not ok and detail else ""))
    print(f"\n{passed}/{len(results)} 通过")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
