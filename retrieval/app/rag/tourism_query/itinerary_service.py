"""

行程拼装服务（规划类问题专用答案生成）。

与普通答案输出的差异：使用 tourism/itinerary_out 专用模板，要求输出结构化行程

（概览/每日安排/花销预估/装备注意），并放宽 max_tokens 以容纳更长内容。

数据来源：rerank 后的 KB 检索结果 + 工具简报（tool_weather/tool_route，含代码层油费估算）。

"""

import re

import time



from app.infra.llm import llm_provider
from app.infra.llm_harness import answer_llm_client, reliable_invoke

from app.infra.persistence.history_repository import history_repository

from app.shared.runtime.date_utils import current_date_text

from app.rag.common.history_text_utils import build_history_text

from app.rag.tourism_query.answer_output_service import extract_image_urls

from app.rag.tourism_query.budget_service import estimate_budget

from app.rag.tourism_query.itinerary_map_service import build_itinerary_map

from app.rag.tourism_query.transport_advice_service import decide_transport

from app.rag.common.intent_route_service import CONSULT_PATTERN

from app.rag.common.place_name_utils import expand_place_tokens

from app.rag.tourism_query.weather_tool_service import extract_destination_info
from app.rag.common.memory_harness import record_plan_memory

from app.shared.runtime.load_prompt import load_prompt

from app.shared.runtime.logger import logger, step_log

from app.shared.utils.sse_utils import SSEEvent, is_cancelled, push_to_session

from app.shared.utils.task_utils import set_task_result

from app.shared.utils.text_utils import BlankLineCompressor, normalize_answer_text



# 引用标记 regex（与答案输出服务共用逻辑）：流式 delta 级别剥离【第N块】和【N】

_CITATION_PATTERN = re.compile(r"(?:\s*【(?:第\d+块|\d+)】)+")



# 行程规划内容更长，放宽生成长度

ITINERARY_MAX_TOKENS: int = 2500



# E1 接驳模式（2026-09-11 主人拍板方案二）：枢纽→市区问句走轻量生成，内容更短

TRANSFER_MAX_TOKENS: int = 1200



# E1 接驳类问句判定：问句同时出现「交通枢纽」与「到市区」意图，且无返程/多天信号。

# 命中 → generate_itinerary 走 build_transfer_prompt（交通段+当日动线），

# 不套"行程概览/每日安排/花销预估"多日模板（主人实测 E1：机场到市区被套完整行程模板）。

_TRANSFER_HUB_PATTERN = re.compile(

    r"(?:机场|高铁站?|火车站|车站|客运站|码头|北站|南站|东站|西站)[^。？?]{0,12}(?:怎么|如何|怎样)(?:到|去|前往|回)"

    r"|(?:机场|高铁站?|火车站|车站|客运站|码头|北站|南站|东站|西站)\s*(?:到|去|前往)\s*(?:市区|市中心|城区|市里)"

    r"|(?:我在|到了?|抵达)[^。？?]{0,14}(?:机场|高铁站?|火车站|车站|客运站|码头|北站|南站|东站|西站)"

)

_TRANSFER_DOWNTOWN_PATTERN = re.compile(r"到市区|去市区|到市中心|到城区|进市区|市区里|市内怎么|市内交通")

_TRANSFER_MULTI_DAY_PATTERN = re.compile(

    r"回程|返程|回去|(?:\d+|[一二两三四五六七八九十]+)\s*(?:天|晚)|(?:几|哪)[天晚]|天行程|日游"

)



def _is_transfer_question(question: str) -> bool:

    """判定是否为「枢纽→市区」接驳类问句（E1 方案二：子图内接驳模式分流）。"""

    if not question:

        return False

    if not _TRANSFER_HUB_PATTERN.search(question):

        return False

    if not _TRANSFER_DOWNTOWN_PATTERN.search(question):

        return False

    if _TRANSFER_MULTI_DAY_PATTERN.search(question):

        return False

    return True



# 链式道路罗列（"沿A高速→B高速→C高速"，≥2 个箭头）：路线工具不提供道路明细，罗列必为编造

_ROAD_CHAIN_PATTERN = re.compile(

    r"[，,;；]?\s*(?:沿|经)?[\u4e00-\u9fa5A-Za-z0-9]{2,10}(?:高速|国道|省道|大道|路)"

    r"(?:\s*→\s*[\u4e00-\u9fa5A-Za-z0-9]{2,10}(?:高速|国道|省道|大道|路)){2,}行驶?|"

    r"[，,;；]?\s*(?:沿|经)?[\u4e00-\u9fa5A-Za-z0-9]{2,10}(?:高速|国道|省道|大道|路)"

    r"(?:、[\u4e00-\u9fa5A-Za-z0-9]{2,10}(?:高速|国道|省道|大道|路)){2,}"

)

_ROAD_CHAIN_REPLACEMENT = "沿主要高速及国道行驶"



# itinerary 结构化标题：紧凑化时仅在这些标题前保留空行

_ITINERARY_HEADING_PATTERN = re.compile(

    r"^(#{1,6}\s|\*\*|\d+[\.、]\s*)?(行程概览|每日安排|花销预估|装备与注意事项|注意事项|Day\s*\d|第[一二三四五六七八九十]+天)"

)





def _compact_itinerary(text: str) -> str:

    """

    行程输出紧凑化：删除标题行以外的空行（列表项/段落之间紧贴），仅结构化标题前保留一个空行。

    模型对"段落间最多空一行"遵循度差，代码级保证排版紧凑。

    """

    lines = text.split("\n")

    result: list[str] = []

    for i, line in enumerate(lines):

        if line.strip() == "" and result:

            # 判断空行后的下一个非空行：是结构化标题则保留空行，否则删除

            nxt = next((l for l in lines[i + 1:] if l.strip()), "")

            if _ITINERARY_HEADING_PATTERN.match(nxt.strip()):

                result.append(line)

            continue

        result.append(line)

    # 收尾：行尾空白 + 首尾空白

    cleaned = "\n".join(line.rstrip() for line in result).strip()

    return cleaned





def _strip_road_chains(text: str) -> str:

    """剥离链式道路罗列（LLM 编造的驾车路径明细），替换为泛化描述。"""

    return _ROAD_CHAIN_PATTERN.sub(_ROAD_CHAIN_REPLACEMENT, text)





# 编造的航班号/车次号（"ZH9901""HU7750""G1234""3U8881"）：路线工具只提供驾车路线，

# 不提供任何航班/列车班次数据，回答里出现具体班次号必为模型幻觉（plan-003 实测编造

# ZH9901/HU7750；itinerary prompt 规则 5 已明令禁止但遵循度不足 → 代码级剥离，

# 与 _strip_road_chains 同思路）。

# 码位前两位须含至少一个大写字母（兼容 3U/9C 等数字开头的航司码），且尾号 ≥3 位，

# 避免误伤"G15高速/G60走廊"等公路编号与 6 位纯数字（价格/电话）。

_FLIGHT_TRAIN_CODE_PATTERN = re.compile(

    r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9]|[0-9][A-Z])\d{3,4}(?![A-Za-z0-9])"

)

# 剥离班次号后的搭配碎片清理："乘坐ZH9901次航班"→"乘坐次航班"→"乘坐航班"；"，HU7750次"→"，"

_CODE_STRIPPED_COLLOCATION_PATTERN = re.compile(

    r"(乘坐|搭乘|乘|搭)\s*次(航班|高铁|动车|火车|列车|大巴|班车)"

    r"|(?<![一-龥0-9A-Za-z])次(?=(?:航班|高铁|动车|火车|列车))"

)





def _strip_flight_train_numbers(text: str, keep_codes: set[str] | None = None) -> str:

    """

    剥离 LLM 编造的航班号/车次号，并清理剥离后残留的搭配碎片与多余空格。



    T14（12306 接入）后改为**白名单保留**：`keep_codes` 传入【铁路参考】中真实存在的

    车次号（来自自部署 12306-MCP），这些车次予以保留；其余仍一律剥离。

    这是 T14 的核心收益——真实车次得以出现在行程里，编造的照样被删。

    注意：保留的车次号两侧的"乘坐…次高铁"碎片不会被误清（码位夹在"乘坐"与"次"之间，

    现有 regex 的 `\\s*` 与数字负向后顾均不匹配），无需额外保护。

    """

    keep = {str(c).strip().upper() for c in (keep_codes or set()) if str(c).strip()}



    def _replace(match: "re.Match[str]") -> str:

        code = match.group(0)

        return code if code.upper() in keep else ""



    text = _FLIGHT_TRAIN_CODE_PATTERN.sub(_replace, text)

    text = _CODE_STRIPPED_COLLOCATION_PATTERN.sub(

        lambda m: (m.group(1) or "") + (m.group(2) or ""), text

    )

    return re.sub(r"[ \t]{2,}", " ", text)





def _rail_keep_codes(state: dict) -> set[str]:

    """取【铁路参考】中真实存在的车次号（供班次号白名单保留使用）。"""

    rail = state.get("tool_rail") or {}

    if not rail.get("ok"):

        return set()

    return {str(code) for code in (rail.get("codes") or []) if code}





# 规划分支无素材拒答闸门（neg-006 根治）：马尔代夫库内无任何资料，模型仍生成

# "贝努埃州 扬代夫"式幻觉行程——KB 素材为空、或解析出的目的地在全部 KB chunk 中

# 零出现时，继续生成必然编造，必须前置拦截停写行程。

_ITINERARY_REFUSAL_FALLBACK = (

    "抱歉，知识库中暂无该目的地的行程规划资料，无法为您生成具体行程，"

    "建议您换一个已收录的目的地再试试。"

)





def _destination_material_hit(

    kb_docs: list[dict],

    tokens: list[str],

    min_text_count: int = 2,

) -> bool:

    """

    目的地素材命中判定（从严）：

    - 命中任一 KB chunk 的"标题"即算有素材；

    - 或正文同一 chunk 内出现 ≥min_text_count 次（多次展开描述才像真实素材）。

    只看"正文出现 1 次"不够：实测 neg-006"马尔代夫"仅因海南攻略里一句

    "中国的马尔代夫"式修辞被误放行，随后照样编造"贝努埃州 扬代夫"行程。



    """

    for doc in kb_docs:

        title = doc.get("title") or ""

        text = doc.get("text") or ""

        for token in tokens:

            if token in title or text.count(token) >= min_text_count:

                return True

    return False





def resolve_plan_material(state: dict) -> tuple[str, str]:

    """

    规划分支素材判定：一次算出「是否拒答」与「素材模式」两个结论。



    判据（满足其一即拒答）：

    1. rerank 后无任何本地资料（全是 web 或为空）——行程只能凭工具简报+幻觉硬编；

    2. 解析出的目的地（及其空格拆分片段）与所属城市，在文旅 chunk 中"标题零命中且

       正文均少于 2 次出现"——库内确实没有素材。

    解析不出目的地时保守放行（闸门宁可漏放，不可误杀正常规划）。



    Returns:

        tuple[str, str]: (拒答文案, 素材模式)。拒答文案非空时直接输出并停写行程；

        素材模式取值为 "kb"（文旅库有素材，正常生成完整行程）、

        "none"（无素材，与拒答文案同时出现）。

    """

    kb_docs = [d for d in (state.get("reranked_docs") or []) if d.get("type") != "web"]

    if not kb_docs:

        return _ITINERARY_REFUSAL_FALLBACK, "none"

    try:

        info = extract_destination_info(state)

    except Exception as e:  # noqa: BLE001

        logger.warning(f"规划闸门:目的地解析异常,放行不拦截,错误信息:{str(e)}")

        return "", "kb"

    destination = (info.get("destination") or "").strip()

    city = (info.get("city") or "").strip()

    if not destination:

        return "", "kb"

    # 形态归一化：LLM 解析"成都/成都市"随机，只按字面判会让闸门在"成都市"上误杀

    # （实测 plan-001：KB 正文写"成都"、解析出"成都市" → 标题正文均零命中 → 拒答）。

    # 闸门原则是"宁可漏放不可误杀"，故原样 + 去行政区划后缀的变体一起参与判定。

    tokens = expand_place_tokens(destination, city)



    # 文旅库有该目的地素材 → 正常完整行程

    if _destination_material_hit(kb_docs, tokens):

        return "", "kb"

    return (

        f"抱歉，知识库中暂无「{destination}」相关的行程规划资料，无法为您生成具体行程，"

        "建议您换一个已收录的目的地。",

        "none",

    )





@step_log("build_itinerary_prompt")

def build_itinerary_prompt(state: dict, material_mode: str = "kb") -> str:

    """构建行程规划 Prompt：KB 上下文 + 工具简报（天气/路线/油费）+ 日期 + 历史。"""

    reranked_docs = state.get("reranked_docs") or []

    if not reranked_docs:

        logger.warning("行程拼装:reranked_docs 为空,仅依据工具简报生成")

        reranked_docs = [{"title": "无", "score": 0, "type": "milvus", "text": "未找到相关资料"}]

    context_limit = 5

    top_docs = reranked_docs[:context_limit]

    context_chunk_list = []

    for number, chunk in enumerate(top_docs, start=1):

        if chunk.get("type") == "web":

            source = "网络搜索"

        else:

            source = "向量查询"

        text = (chunk.get("text") or "")[:800]

        context_chunk_list.append(

            f"第{number}块: 标题:{chunk.get('title')} 来源:{source}\n内容:{text}"

        )

    context_chunk_str = "\n\n".join(context_chunk_list)



    weather = state.get("tool_weather") or {}

    weather_text = weather.get("text", "") if weather.get("ok") else "（本次无实时天气数据）"

    route = state.get("tool_route") or {}

    route_text = route.get("text", "") if route.get("ok") else "（本次无路线规划数据）"

    # T14：铁路简报（真实车次/历时/各席别票价），跨城行程的交通与路费以此为准

    rail = state.get("tool_rail") or {}

    rail_text = rail.get("text", "") if rail.get("ok") else "（本次无铁路车次数据）"

    # T15-D1：景点 POI 简报（高德结构化：地址/评分/人均/开放时间 + 景点间距离）

    poi = state.get("tool_poi") or {}

    poi_text = poi.get("text", "") if poi.get("ok") else "（本次无景点POI数据）"

    # 问题2：住宿/餐饮 POI 简报（高德结构化：位置/评分/人均/营业时间）

    stay_food = state.get("tool_stay_food") or {}

    stay_food_text = (

        stay_food.get("text", "") if stay_food.get("ok") else "（本次无住宿/餐饮POI数据）"

    )

    history_text = build_history_text(state.get("history", []))

    item_names = state.get("item_names") or []

    item_name_str = "本次关联主体:" + ",".join(item_names) if item_names else "未确认具体主体"

    # 行程拼装必须用"原始问句"（保留"可以吗/行不行"等征询语气），改写句常被归一化成无语气陈述

    original_question = state.get("original_query") or state.get("rewritten_query") or ""

    rewritten_question = state.get("rewritten_query") or state.get("original_query") or ""

    # 征询语气问题（"我想坐飞机去杭州可以吗"）：先答可行性结论再给方案，且针对原问句里的交通方式作答

    consult_instruction = (

        "用户问题是征询语气的疑问句（含“可以吗/能不能/行不行”等），要求在【行程概览】之前先单独输出一段"

        "【结论】：直接回答可以还是不可以（结合【实时天气参考】【路线参考】判断，一句话说明理由；天气/路线"

        "缺失时基于常识给出结论并提示出行前核实），再按下面要求输出完整行程规划。交通方式以用户问题中提出的"

        "方式为准（若用户明确提到坐飞机，概览与每日安排就围绕飞机展开，先说明是否可行再给安排）。"

        if CONSULT_PATTERN.search(original_question)

        else ""

    )





    # 交通方式建议（问题1/3）：代码层确定性推荐（阈值规则 + 距离换算），不让 LLM 自由裁量

    transport = decide_transport(state)

    transport_text = transport.get("text") or "（本次无法给出交通方式建议，请按【路线参考】与【铁路参考】自行判断）"

    # 反向锚定治理（问题3）：已定飞机方案时，【铁路参考】里的真实车次会把模型拽回火车

    # （线上实测："当然是坐飞机去啊"仍输出"公共交通（火车）20h57m"）。此处直接把铁路区块

    # 换成禁止性说明；tool_rail 本身保留在 state（车次白名单仍需用于剥离编造班次号）。

    if transport.get("recommended") == "飞机":

        rail_text = (

            "（本次采用飞机方案，请勿在行程中出现火车/高铁/动车班次、铁路历时或铁路票价；"

            "飞机只写「乘飞机」，不得编造航班号与起降时刻）"

        )

    # 花销估算（问题2）：代码层确定性计算，逐项标注来源（12306/高德真实值 或 标注"预估"）

    budget = estimate_budget(state, transport)

    budget_text = budget.get("text") or "（本次无法给出花销估算，请按【参考内容】与原口径处理）"

    state["tool_transport"] = transport

    state["tool_budget"] = budget



    return load_prompt(

        "tourism/itinerary_out",

        # 日期必须兜底现算：current_date 缺失时模型会对"下周一"等相对时间自行做日历算术，

        # 实测(plan-001)会用训练记忆里的旧年份编造日期（如"2024年3月25日"）

        current_date=state.get("current_date") or current_date_text(),

        weather=weather_text,

        route=route_text,

        rail=rail_text,

        poi=poi_text,

        stay_food=stay_food_text,

        transport=transport_text,

        budget=budget_text,

        context=context_chunk_str,

        history=history_text,

        item_names=item_name_str,

        question=original_question,

        # 延续类追问（"我想坐飞机去"）原始问句无目的地，改写句才是补全后的完整问题，

        # 交给 prompt 做消解；实测缺失时模型会凭空编造目的地（深圳→成都）。

        rewritten=rewritten_question,

        consult_instruction=consult_instruction,



    )





def build_transfer_prompt(state: dict) -> str:

    """

    E1 接驳模式 Prompt（2026-09-11 主人拍板方案二）：枢纽→市区交通段 + 当日市内动线。

    与 build_itinerary_prompt 的差异：轻量化——只保留天气/驾车路线/POI 三类工具简报，

    不带交通方式建议/铁路/花销估算（接驳指南不需要多日跨城决策数据）。

    当日动线仍用「第1天」标题格式，保证 build_itinerary_map 能复用解析出地图点位。

    """

    reranked_docs = state.get("reranked_docs") or []

    if not reranked_docs:

        logger.warning("接驳模式:reranked_docs 为空,仅依据工具简报生成")

        reranked_docs = [{"title": "无", "score": 0, "type": "milvus", "text": "未找到相关资料"}]

    top_docs = reranked_docs[:5]

    context_chunk_list = []

    for number, chunk in enumerate(top_docs, start=1):

        if chunk.get("type") == "web":

            source = "网络搜索"



        else:

            source = "向量查询"

        text = (chunk.get("text") or "")[:800]

        context_chunk_list.append(

            f"第{number}块: 标题:{chunk.get('title')} 来源:{source}\n内容:{text}"

        )

    context_chunk_str = "\n\n".join(context_chunk_list)

    weather = state.get("tool_weather") or {}

    weather_text = weather.get("text", "") if weather.get("ok") else "（本次无实时天气数据）"

    route = state.get("tool_route") or {}

    route_text = route.get("text", "") if route.get("ok") else "（本次无路线规划数据）"

    poi = state.get("tool_poi") or {}

    poi_text = poi.get("text", "") if poi.get("ok") else "（本次无景点POI数据）"

    history_text = build_history_text(state.get("history", []))

    item_names = state.get("item_names") or []

    item_name_str = "本次关联主体:" + ",".join(item_names) if item_names else "未确认具体主体（依据检索内容作答）"

    original_question = state.get("original_query") or state.get("rewritten_query") or ""

    rewritten_question = state.get("rewritten_query") or state.get("original_query") or ""

    return load_prompt(

        "tourism/itinerary_transfer",

        # 日期必须兜底现算：缺失时模型会对相对时间自行做日历算术（与 itinerary_out 同口径）

        current_date=state.get("current_date") or current_date_text(),

        weather=weather_text,

        route=route_text,

        poi=poi_text,

        context=context_chunk_str,

        history=history_text,

        item_names=item_name_str,

        question=original_question,

        rewritten=rewritten_question,

    )



@step_log("generate_itinerary")

def generate_itinerary(state: dict) -> dict:

    """

    行程拼装主入口：构建 Prompt → 调用模型生成结构化行程 → 落历史。

    支持流式/普通两种模式，与普通答案输出的输出协议一致（SSE delta + task_result）。



    Returns:

        dict: {"answer","image_urls"} 节点更新。

    """

    # P1 闸门：库内无该目的地素材时前置拒答，不调用模型（杜绝无出处行程幻觉）



    refusal_text, material_mode = resolve_plan_material(state)

    if refusal_text:

        logger.info(f"规划闸门命中,拒答停写行程: {refusal_text[:50]}")

        session_id = state.get("session_id")

        set_task_result(session_id, "answer", refusal_text)

        if state.get("is_stream", False):

            push_to_session(session_id, SSEEvent.DELTA, {"delta": refusal_text})

        state["answer"] = refusal_text

        state["image_urls"] = []

        history_repository.save_message(

            session_id=session_id,

            role="assistant",

            text=refusal_text,

            rewritten_query=state.get("rewritten_query") or state.get("original_query"),

            item_names=state.get("item_names", []),

            image_urls=[],

            domain=state.get("domain", ""),

            is_plan=bool(state.get("is_plan", False)),

        )

        return {"answer": refusal_text, "image_urls": []}



    # E1 接驳分流（2026-09-11 主人拍板方案二）：枢纽→市区问句走轻量接驳模式

    transfer_mode = _is_transfer_question(

        state.get("original_query") or state.get("rewritten_query") or ""

    )

    if transfer_mode:

        logger.info("E1 接驳模式命中：枢纽→市区问句走轻量接驳指南（不套多日行程模板）")

    prompt = build_transfer_prompt(state) if transfer_mode else build_itinerary_prompt(state)

    session_id = state.get("session_id")

    is_stream = state.get("is_stream", False)

    lm_client = answer_llm_client(

        state,

        TRANSFER_MAX_TOKENS if transfer_mode else ITINERARY_MAX_TOKENS,

        tier_default="longctx",

    )



    final_result = ""

    if is_stream:

        blank_comp = BlankLineCompressor()

        for chunk in lm_client.stream(prompt):

            if is_cancelled(session_id):

                logger.info(f"用户停止生成,中断行程规划输出,session_id={session_id}")

                break

            # per-delta 轻清理：剥离引用标记 + 压缩连续空行（模型常输出 \n\n\n\n 大片空行）

            delta_content = blank_comp.feed(_CITATION_PATTERN.sub("", chunk.content))

            final_result += delta_content

            if delta_content:

                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})

    else:

        response = reliable_invoke(lm_client, prompt, state)

        final_result = response.content



    final_result = normalize_answer_text(final_result)

    # 代码级根治：剥离 LLM 编造的链式道路罗列 + 班次号 + 紧凑化排版

    # （模型对 prompt 排版/禁编造约束遵循度差，统一代码级保证）

    final_result = _compact_itinerary(

        _strip_road_chains(

            _strip_flight_train_numbers(final_result, keep_codes=_rail_keep_codes(state))

        )

    )

    set_task_result(session_id, "answer", final_result)

    state["answer"] = final_result

    state["image_urls"] = extract_image_urls(state.get("reranked_docs") or [])

    # 问题5：把行程正文还原成「按天景点 + 坐标」，供前端内嵌高德地图（无坐标则空结构，不渲染）

    map_data = build_itinerary_map(state, final_result)

    state["itinerary_map"] = map_data

    state["map_data"] = map_data

    record_plan_memory(state)  # F：行程成功生成后回写长期记忆（开关默认 OFF）

    history_repository.save_message(

        session_id=session_id,

        role="assistant",

        text=final_result,

        rewritten_query=state.get("rewritten_query") or state.get("original_query"),

        item_names=state.get("item_names", []),

        image_urls=state.get("image_urls", []),

        domain=state.get("domain", ""),

        is_plan=bool(state.get("is_plan", False)),

        map_data=map_data,

    )

    return {"answer": final_result, "image_urls": state["image_urls"], "map_data": map_data}

