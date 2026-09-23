# -*- coding: utf-8 -*-
"""
A1 景点提及确定性回填单测。

运行方式：
  python test/test_attraction_mention_service.py

覆盖：
1) 候选词过滤 is_valid_term（长度/数字/纯ASCII/行政区含省市组合/方位泛称/句式碎片/通用描述词/城市别称）
   与 _normalize_term（后缀归一）；
2) 词典构建 build_attraction_lexicon（主题字段豁免次数 / 标题词需正文复现≥2次 / extra_terms 同受过滤）；
3) 提及匹配 match_mentions（长词优先、子串去重、按出现顺序、去重、空文本）；
4) 类型→关系字段映射 relation_key_for，并与 content_schema.EXTRA_MODELS 对表防漂移；
5) 回填计划 plan_relation_backfill（仅承载类型、已有值不覆盖=幂等、不修改入参）；
6) 景点级线索收割 harvest_attraction_terms。

设计约束：本模块为 rag 层纯逻辑，零外部服务、零 LLM，故本单测可离线全绿。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.tourism_import import attraction_mention_service as ams  # noqa: E402


def _case(results, name, fn) -> None:
    """执行一个断言函数并登记结果（异常即判失败，不中断其余用例）。"""
    try:
        info = fn() or "ok"
        results.append((name, True, str(info)))
    except AssertionError as e:
        results.append((name, False, f"断言失败: {e}"))
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"异常: {e!r}"))


# ---- 1) 候选词过滤 ----
def _case_filter():
    reject = [
        "", "西", "abcdefg",              # 空 / 太短 / 太长
        "2024年", "3号线",                 # 含数字
        "abc",                             # 纯 ASCII（无中文）
        "杭州", "四川", "北京", "四川成都", "甘肃敦煌",  # 行政区 / 省+市组合
        "东线", "商圈", "市区",             # 方位·片区泛称
        "怎么去", "值得吗", "适合谁",        # 句式碎片
        "元数据", "攻略", "住宿", "打卡", "古城", "海边",  # 结构词 / 通用描述词
        "千塔之城", "永恒之城", "音乐之都", "光之城", "北方威尼斯",  # 描述性城市别称
    ]
    for t in reject:
        assert not ams.is_valid_term(t), f"应被拒绝却通过: {t!r}"
    accept = ["西湖", "灵隐", "鼓浪屿", "曾厝垵", "武陵源", "蜈支洲岛",
              "宽窄巷子", "张掖丹霞地貌"]
    for t in accept:
        assert ams.is_valid_term(t), f"应通过却被拒: {t!r}"
    # 行政区内嵌同名景点：is_valid_term 会挡，但由 _looks_admin 走补充词表通道放行
    assert not ams.is_valid_term("九寨沟")
    assert ams._looks_admin("九寨沟")
    return f"拒绝{len(reject)}/通过{len(accept)}"


def _case_normalize():
    assert ams._normalize_term("龙井方向") == "龙井"
    assert ams._normalize_term("灵隐周边") == "灵隐"
    assert ams._normalize_term("  西湖  ") == "西湖"
    assert ams._normalize_term("｜鼓浪屿｜") == "鼓浪屿"
    return "4/4"


# ---- 2) 候选抽取 & 词典构建 ----
def _case_extract():
    text = "- 主题：西湖、灵隐\n## 灵隐\n## 注意事项\n"
    topics, headings = ams.extract_candidates(text)
    assert topics == {"西湖", "灵隐"}, topics
    assert headings == {"灵隐"}, headings  # 「注意事项」含结构词「注意」→ 被过滤
    return str(sorted(topics))


def _case_lexicon():
    doc = ("- 主题：西湖、灵隐\n"
           "西湖很大，灵隐寺香火旺盛，梵净山也值得去。\n"
           "## 西湖\n## 梵净山\n## 永恒之城\n"
           "西湖与灵隐相隔不远。\n")
    lex = ams.build_attraction_lexicon([("杭州景点推荐", doc)])
    # 主题字段豁免次数（灵隐/西湖）；梵净山靠正文复现≥2次入典；
    # 「永恒之城」只出现 1 次且属城市别称 → 不入典
    assert lex == {"西湖", "灵隐", "梵净山"}, lex

    # 标题词出现 1 次 → 不入典
    assert ams.build_attraction_lexicon([("t", "## 只出现一次\n正文没有提到它。\n")]) == set()

    # extra_terms 同受过滤：九寨沟靠 _looks_admin 放行，含通用词的沙溪古镇被拦
    lex2 = ams.build_attraction_lexicon(
        [("x", "- 主题：西湖\n西湖很好，西湖很美。\n")],
        extra_terms=["九寨沟", "沙溪古镇", "四川阿坝", "3号", "abc"],
    )
    assert "九寨沟" in lex2 and "四川阿坝" in lex2 and "西湖" in lex2
    assert "沙溪古镇" not in lex2, "含通用词「古镇」的补充词不应入典"
    assert "3号" not in lex2 and "abc" not in lex2
    return f"主词典{len(lex)}词"


# ---- 3) 提及匹配 ----
def _case_match():
    # 长词优先 + 子串去重
    assert ams.match_mentions("故宫博物院很大，泰山的日出很美",
                              {"故宫", "故宫博物院", "泰山"}) == ["故宫博物院", "泰山"]
    # 按出现顺序
    assert ams.match_mentions("灵隐之后去西湖", {"西湖", "灵隐"}) == ["灵隐", "西湖"]
    # 去重
    assert ams.match_mentions("西湖 西湖 西湖", {"西湖"}) == ["西湖"]
    # 空文本 / 无命中
    assert ams.match_mentions("", {"西湖"}) == []
    assert ams.match_mentions("无关文本", {"西湖"}) == []
    return "6/6"


# ---- 4) 类型映射 + schema 防漂移 ----
def _case_relation_key():
    from app.rag.tourism_import.content_schema import EXTRA_MODELS, TourismContentType

    expect = {
        "酒店信息": "nearby_attractions",
        "文化知识介绍": "related_attractions",
        "游记攻略": "attractions",
        "线路推荐": "attractions",
    }
    for ct, key in expect.items():
        assert ams.relation_key_for(ct) == key, (ct, ams.relation_key_for(ct))
        model = EXTRA_MODELS[TourismContentType(ct)]
        fields = getattr(model, "model_fields", None) or getattr(model, "__fields__", {})
        assert key in fields, f"content_schema 已无 {ct}.{key}，本模块映射需同步"
    for ct in ("景点信息", "景区介绍", "美食推荐", "交通指南",
               "推荐运营资料", "游客评论摘要", "常见问答", "运营表格", ""):
        assert ams.relation_key_for(ct) is None, ct
    assert ams.SOURCE_CONTENT_TYPES == ("景点信息", "景区介绍")
    return "4类映射 + schema 对表一致"


# ---- 5) 回填计划（幂等） ----
def _case_plan():
    lex = {"西湖", "灵隐"}
    rows = [
        {"chunk_id": "c1", "file_title": "杭州住宿推荐", "content_type": "酒店信息",
         "title": "湖滨住宿", "content": "住在湖滨，步行可到西湖和灵隐。", "extra_meta": {}},
        {"chunk_id": "c2", "file_title": "杭州住宿推荐", "content_type": "酒店信息",
         "title": "西湖边", "content": "西湖边上的酒店。",
         "extra_meta": {"nearby_attractions": ["西湖"]}},
        {"chunk_id": "c3", "file_title": "杭州景点推荐", "content_type": "景点信息",
         "title": "西湖十景", "content": "西湖十景。", "extra_meta": {}},
        {"chunk_id": "c4", "file_title": "杭州美食", "content_type": "美食推荐",
         "title": "西湖醋鱼", "content": "西湖醋鱼。", "extra_meta": {}},
    ]
    plan = ams.plan_relation_backfill(rows, lex)
    assert [p["chunk_id"] for p in plan] == ["c1"], plan
    assert plan[0]["key"] == "nearby_attractions"
    assert plan[0]["values"] == ["西湖", "灵隐"], plan[0]["values"]
    assert plan[0]["extra_meta"] == {"nearby_attractions": ["西湖", "灵隐"]}
    assert rows[0]["extra_meta"] == {}, "入参 extra_meta 不应被就地修改"

    # 幂等：按计划回写后再算一次，应为空（已有值不覆盖）
    merged = []
    for r in rows:
        m = dict(r)
        if r["chunk_id"] == "c1":
            m["extra_meta"] = dict(plan[0]["extra_meta"])
        merged.append(m)
    assert ams.plan_relation_backfill(merged, lex) == []

    # 承载类型但正文无命中 → 不入计划
    assert ams.plan_relation_backfill(
        [{"chunk_id": "c9", "content_type": "游记攻略", "content": "没有景点名。",
          "extra_meta": {}}], lex) == []
    return "幂等 2 次一致"


# ---- 6) 景点级线索收割 ----
def _case_harvest():
    rows = [
        {"content_type": "景点信息", "item_name": "九寨沟", "file_title": "九寨沟",
         "extra_meta": {}},
        {"content_type": "景点信息", "item_name": "沙溪古镇", "file_title": "沙溪古镇景点介绍",
         "extra_meta": {}},
        {"content_type": "景点信息", "item_name": "x", "file_title": "x",
         "extra_meta": {"attraction_name": "峨眉山"}},
        {"content_type": "酒店信息", "item_name": "西湖", "file_title": "杭州住宿推荐",
         "extra_meta": {}},
    ]
    terms = ams.harvest_attraction_terms(rows)
    assert terms == {"沙溪古镇", "峨眉山"}, terms
    return str(sorted(terms))


def main() -> None:
    results: list[tuple[str, bool, str]] = []
    _case(results, "1 候选词过滤（7类拒绝规则 + 行政区放行通道）", _case_filter)
    _case(results, "2 后缀归一 _normalize_term", _case_normalize)
    _case(results, "3 候选抽取（主题/标题分流）", _case_extract)
    _case(results, "4 词典构建（次数门槛 + 补充词过滤）", _case_lexicon)
    _case(results, "5 提及匹配（长词优先/顺序/去重）", _case_match)
    _case(results, "6 类型→关系字段映射 + schema 对表", _case_relation_key)
    _case(results, "7 回填计划（仅承载类型/幂等/不改入参）", _case_plan)
    _case(results, "8 景点级线索收割", _case_harvest)

    print("== attraction_mention_service（A1）单测结果 ==")
    ok = True
    for name, passed, info in results:
        print(f"  {'PASS' if passed else 'FAIL'} {name}：{info}")
        ok = ok and passed
    print(f"A1 单测结果：{sum(1 for r in results if r[1])}/{len(results)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
