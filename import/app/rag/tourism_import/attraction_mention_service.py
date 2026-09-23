# -*- coding: utf-8 -*-
"""
景点提及确定性回填服务（A1）。

背景（2026-09-18 实测）：LLM 抽取的 extra_meta 关系字段（nearby_attractions /
related_attractions / attractions）在真库的填充率是 **0** —— flash 级模型
（glm-4-flash-250414）会照抄抽取 prompt 输出骨架里的空对象 `"extra": {{}}`。
故改为**确定性回填**：不调 LLM、零幻觉、可复现、可单测。

做法：用库内景点类文档构建景点词典，再对「住宿/游记/文化/线路」切片的正文做
关键词匹配，命中即写入对应关系字段。

词典构建的两级来源（精度优先，2026-09-18 量测后定稿）：
1. **元数据「主题：」字段** —— 上游模板产出的受控小词表，视为可信（豁免出现次数要求）；
2. **Markdown `###` 标题** —— 噪声大，要求该词在**本文正文里出现 ≥2 次**才入典。

候选词另需通过 `is_valid_term`：档掉通用描述词 / 结构体裁词 / 句式碎片 / 行政区名
（含「省+市」组合）/ 方位泛称后缀（线·区·圈）/ 描述性城市别称（*之城 · *之都 ·
*威尼斯）/ 纯 ASCII 词。

真库量测（2026-09-18，387 chunk）：词典 43 词，可回填 49/208 承载切片（23.6%），
其中**酒店信息 19/19 = 100%**、游记攻略 42.4%、线路推荐 33.3%、文化知识介绍 2.6%。
明细见 `output/relation_backfill_analysis.txt`（只读量测脚本 `scripts/analyze_relation_backfill.py`）。

局限（必须知悉，勿夸大效果）：
- 只覆盖「正文里明确写出景点名」的切片，不做语义推断；
- 与行政区同名的景点（九寨沟/峨眉山/敦煌等）被行政区过滤挡掉 → 由调用方把
  `extra_meta.attraction_name` 与主体库景点级 `item_name` 作为**补充词表**并入；
- 词典是**需要维护的资产**：语料新增城市/新景点名后，如未命中需检查过滤表。

依赖方向：本模块属 rag 层纯逻辑，**禁止 import app.process.***。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from app.shared.utils.city_utils import CITY_LIST

# 词典来源类型：只有这些类型承载「景点名」
SOURCE_CONTENT_TYPES: tuple[str, ...] = ("景点信息", "景区介绍")

# 内容类型 → 该类型承载关系字段的键名（不在表内 = 不回填）
# 依据：app/rag/tourism_import/content_schema.py 各 Extra 模型的关系字段
RELATION_KEY_BY_CONTENT_TYPE: dict[str, str] = {
    "酒店信息": "nearby_attractions",
    "文化知识介绍": "related_attractions",
    "游记攻略": "attractions",
    "线路推荐": "attractions",
}

# ---- 候选词过滤 ----
# 1) 体裁/结构词：多为小节标题，不是景点名
_STRUCT_WORDS: tuple[str, ...] = (
    "景点", "景区", "推荐", "结构", "概览", "逻辑", "建议", "时长", "人群",
    "决策", "优先级", "差异", "展开", "安排", "核心", "层", "天数", "游客",
    "旅行", "旅游", "攻略", "指南", "注意", "费用", "交通", "住宿", "美食",
    "线路", "行程", "玩法", "体验", "价值", "误区", "参考", "总结", "简介",
    "介绍", "特点", "文化", "风俗", "礼仪", "信息", "清单", "问答", "评论",
    "提示", "必备", "须知", "技巧", "预算",
)

# 2) 通用描述词：是「体验/场景/泛称」，不是可关联的景点节点
_GENERIC_WORDS: tuple[str, ...] = (
    "漫游", "慢游", "观光", "度假", "休闲", "海湾", "离岛", "本岛", "海岛",
    "古城", "古镇", "老城", "街区", "湖景", "海景", "夜景", "风景", "景观",
    "雪山", "山地", "山水", "拍照", "摄影", "散步", "打卡", "出片",
    "民宿", "酒店", "特产", "自由行", "串联", "门户", "到访", "首次",
    "节奏", "氛围", "元数据", "结语", "引言", "前言", "序言", "目录",
    "附录", "亮点", "特色", "榜单", "排行", "名单", "说明", "背景", "表格",
    "海边", "市区", "湾区", "轮渡", "亲子", "情侣", "古城", "古典", "厚重",
    "好玩", "出片", "小众", "宝藏", "秘境", "景色", "看海", "慢游",
)

# 3) 句式碎片标记：命中即丢（多为小节标题里的半句话）
_SENTENCE_MARKERS: tuple[str, ...] = (
    "的", "怎么", "为什么", "如何", "适合", "值得", "值不值", "要不要",
    "是否", "一定", "必须", "不能", "可以", "需要", "最后", "解决", "越",
    "最", "还是", "不要", "分别", "相关", "常见", "以上", "喜欢", "想要",
    "适合谁", "值得吗",
)

# 3.5) 描述性城市别称后缀：「千塔之城」（布拉格）「永恒之城」（罗马）「音乐之都」
#      （维也纳）「光之城」（巴黎）「北方威尼斯」——是修辞，不是可作为关联锚点的地名
_METAPHOR_SUFFIXES: tuple[str, ...] = ("之城", "之都", "威尼斯", "之珠", "之滨")

# 4) 行政区名（与省市同名 → 作为「景点关联锚点」无意义，会匹配到任意文档）
_PROVINCES: tuple[str, ...] = (
    "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建",
    "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "四川",
    "贵州", "云南", "陕西", "甘肃", "青海", "内蒙古", "宁夏", "新疆", "西藏",
    "台湾", "香港", "澳门",
)

_ADMIN_NAMES: frozenset[str] = frozenset(CITY_LIST) | frozenset({
    "北京", "上海", "天津", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "广西", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "内蒙古", "宁夏", "新疆", "西藏", "台湾", "香港", "澳门",
    "中国", "全国", "全区", "全省", "南方", "北方", "西部", "东部", "欧洲",
    "亚洲", "非洲", "美洲", "全球", "世界", "国内", "国外",
})


def _looks_admin(term: str) -> bool:
    """是否行政区名（含「省+市」组合，如 四川成都 / 甘肃敦煌 / 陕西西安）。"""
    if term in _ADMIN_NAMES:
        return True
    for p in _PROVINCES:
        if term.startswith(p) and term[len(p):] in _ADMIN_NAMES:
            return True
    return False

# 候选词切分符（把「河坊街和湖滨」切成两个）
_SPLIT_RE = re.compile(r"[、，,／/｜|（）()\[\]【】\s]+|和|与|及")

# 中文字符检测（候选词必须含中文）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 归一化后缀（「龙井方向」→「龙井」、「灵隐周边」→「灵隐」）
_SUFFIXES: tuple[str, ...] = ("方向", "片区", "周边", "区域", "一带", "附近", "所在")

_HEADING_RE = re.compile(r"^\s{0,3}#{2,6}\s*(.+?)\s*$", re.MULTILINE)
_TOPIC_RE = re.compile(r"^\s*[-*]?\s*主题\s*[：:]\s*(.+?)\s*$", re.MULTILINE)

_TERM_MIN_LEN = 2
_TERM_MAX_LEN = 6

# 早期版本遗留的精确噪声词（保留以兼容既有行为）
_NOISE_EXACT_LEGACY: frozenset[str] = frozenset({
    "周边", "附近", "全区", "全城", "市内",
})


def _normalize_term(term: str) -> str:
    """去掉常见结构后缀与包裹符号，返回归一后的候选词（可能为空串）。"""
    t = term.strip().strip("｜|·・-—:：\"'“”‘’")
    for suf in _SUFFIXES:
        if len(t) > len(suf) + 1 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t.strip()


def is_valid_term(term: str) -> bool:
    """候选词是否可作为景点词典条目（纯确定性规则，无外部依赖）。"""
    if not term:
        return False
    if not (_TERM_MIN_LEN <= len(term) <= _TERM_MAX_LEN):
        return False
    if any(ch.isdigit() for ch in term):
        return False
    # 必须含中文：挡住纯 ASCII / 拼音碎片（如 "abc"）
    if not _CJK_RE.search(term):
        return False
    if term in _NOISE_EXACT_LEGACY or _looks_admin(term):
        return False
    # 描述性城市别称（千塔之城 / 永恒之城 / 北方威尼斯 …）
    if term.endswith(_METAPHOR_SUFFIXES):
        return False
    # 方位/片区泛称（东线、西线、市区、湾区…）
    if term.endswith(("线", "区", "圈")):
        return False
    if any(m in term for m in _SENTENCE_MARKERS):
        return False
    if any(w in term for w in _STRUCT_WORDS) or any(w in term for w in _GENERIC_WORDS):
        return False
    return True


def split_raw_segments(text: str) -> tuple[list[str], list[str]]:
    """把正文拆成两类原始片段：(主题字段片段, Markdown 标题片段)。"""
    topics = [m.group(1) for m in _TOPIC_RE.finditer(text or "")]
    headings = [m.group(1) for m in _HEADING_RE.finditer(text or "")]
    return topics, headings


def extract_candidates(text: str) -> tuple[set[str], set[str]]:
    """从一份景点类文档正文抽候选词，返回 (主题候选, 标题候选)。"""
    topics_raw, headings_raw = split_raw_segments(text)

    def _clean(segs: list[str]) -> set[str]:
        out: set[str] = set()
        for seg in segs:
            for part in _SPLIT_RE.split(seg):
                term = _normalize_term(part)
                if is_valid_term(term):
                    out.add(term)
        return out

    return _clean(topics_raw), _clean(headings_raw)


def build_attraction_lexicon(
    sources: Iterable[tuple[str, str]],
    *,
    min_occurrences: int = 2,
    extra_terms: Iterable[str] = (),
) -> set[str]:
    """用景点类文档（(file_title, text) 序列）构建景点词典。

    入典条件（满足其一）：
      - 来自元数据「主题：」字段且通过过滤（受控词表，视为可信）；
      - 来自 Markdown 标题，且该词在**本文正文**里出现次数 ≥ `min_occurrences`
        （用于挡掉「永恒之城」「千塔之城」这类只出现一次的修饰性小标题）。

    Args:
        sources: 每项为 (file_title, 正文) 的景点类文档。
        min_occurrences: 标题来源候选词在本文正文中的最小出现次数。
        extra_terms: 补充词表（如库内已有的 `attraction_name`、主体库景点级主体名）。

    Returns:
        set[str]: 归一后的景点词集合。
    """
    kept: set[str] = set()
    for _title, text in sources:
        text = text or ""
        topic_terms, heading_terms = extract_candidates(text)
        kept |= topic_terms
        for term in heading_terms:
            if term in topic_terms:
                continue
            if text.count(term) >= max(1, min_occurrences):
                kept.add(term)
    for term in extra_terms:
        t = _normalize_term(str(term or ""))
        if is_valid_term(t) or _looks_admin(t):
            kept.add(t)
    return kept


def match_mentions(text: str, lexicon: Iterable[str]) -> list[str]:
    """在正文里做词典匹配，返回按出现顺序去重的命中词。

    - 长词优先，避免「西湖」把「西湖北线」的命中吞掉；
    - 若某命中词是另一命中词的子串，只保留更长的那个。
    """
    if not text:
        return []
    lex = set(lexicon)
    hits: list[tuple[int, str]] = []
    for term in lex:
        idx = text.find(term)
        if idx >= 0:
            hits.append((idx, term))
    if not hits:
        return []
    kept: list[tuple[int, str]] = []
    for idx, term in sorted(hits, key=lambda x: (-len(x[1]), x[0])):
        if any(term != other and term in other for _, other in hits):
            continue
        kept.append((idx, term))
    kept.sort(key=lambda x: x[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _idx, term in kept:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered


def relation_key_for(content_type: str) -> str | None:
    """返回该内容类型应回填的关系字段名；不适用则 None。"""
    return RELATION_KEY_BY_CONTENT_TYPE.get((content_type or "").strip())


def harvest_attraction_terms(rows: Iterable[dict]) -> set[str]:
    """从已有 chunk 行里收割「景点级」线索，作为词典补充词表。

    两类：
      1. `extra_meta["attraction_name"]`（LLM 抽到过景点名的少数切片）；
      2. `item_name` 与 `file_title` **不同**的景点类主体（说明主体落到了景点级，
         例如「九寨沟」「沙溪古镇」，而不是文件级兜底名）。
    """
    terms: Counter[str] = Counter()
    for r in rows:
        if (r.get("content_type") or "") not in SOURCE_CONTENT_TYPES:
            continue
        extra = r.get("extra_meta") or {}
        if isinstance(extra, dict) and extra.get("attraction_name"):
            terms[str(extra["attraction_name"]).strip()] += 1
        item = (r.get("item_name") or "").strip()
        if item and item != (r.get("file_title") or "").strip():
            terms[item] += 1
    return {t for t in terms if t}


def plan_relation_backfill(rows: Iterable[dict], lexicon: Iterable[str]) -> list[dict]:
    """计算回填计划（纯函数，不修改入参、不碰数据库）。

    只针对承载关系字段的内容类型，且**仅在该键当前为空时**填（幂等、不覆盖 LLM 结果）。

    Args:
        rows: 每项需含 chunk_id / content_type / content（正文）；extra_meta 可选。
        lexicon: 景点词典。

    Returns:
        list[dict]: [{"chunk_id", "content_type", "key", "values", "extra_meta"}]，
        `extra_meta` 为合并后的新值（便于直接回写）。
    """
    lex = set(lexicon)
    plan: list[dict] = []
    for row in rows:
        ct = (row.get("content_type") or "").strip()
        key = relation_key_for(ct)
        if not key:
            continue
        extra = row.get("extra_meta") or {}
        if not isinstance(extra, dict):
            extra = {}
        if extra.get(key):
            continue  # 已有值 → 不覆盖
        text = " ".join(filter(None, [
            row.get("title") or "",
            row.get("parent_title") or "",
            row.get("content") or "",
        ]))
        values = match_mentions(text, lex)
        if not values:
            continue
        merged = dict(extra)
        merged[key] = values
        plan.append({
            "chunk_id": row.get("chunk_id"),
            "file_title": row.get("file_title"),
            "content_type": ct,
            "key": key,
            "values": values,
            "extra_meta": merged,
        })
    return plan
