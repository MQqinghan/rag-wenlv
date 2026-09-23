"""
文旅意图路由服务模块：判断用户问题归属（文旅/闲聊），规则优先 + LLM 兜底。
输出 state["domain"]，供统一查询图做分支路由。

三层匹配优先级：
1. 关键词规则匹配：文旅关键词 > 闲聊关键词（文旅优先）
2. 规划/天气/交通/纠正等结构性信号：规则显式命中归文旅并置标记
3. LLM 兜底：只在前两层都未命中时才走 LLM
"""
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm import llm_provider
from app.infra.vectorstore import milvus_gateway
from app.shared.config.common import env_bool
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.logger import logger, step_log

# 意图域枚举（与统一图路由键保持一致）
DOMAIN_TOURISM = "tourism"
DOMAIN_CHITCHAT = "chitchat"

# ---- 规则关键词（零 LLM 成本命中大部分场景）----
# 文旅关键词：涉及旅游、景点、文化等
_TOURISM_KEYWORDS = (
    "旅游", "旅行", "景点", "景区", "攻略", "游记", "文化", "历史",
    "民俗", "非遗", "博物馆", "古镇", "门票", "美食", "酒店", "打卡",
    "风土人情", "习俗", "泼水节", "北欧", "傣族", "路线", "行程",
    "省份", "城市", "地区", "国家", "一日游", "两日游",
)
# 闲聊关键词
_CHITCHAT_KEYWORDS = (
    "你好", "您好", "hi", "hello", "hey", "谢谢", "感谢", "再见",
    "拜拜", "你是谁", "你能做什么", "天气", "几点", "在吗",
)
# 追问句式：明显承接上文的短问句（"那门票多少钱""还有别的吗""它有什么特点呢"）。
# 这类问句自身领域信号极弱，靠关键词和 LLM 都容易误判，但上下文已明确给出答案。
_FOLLOWUP_MAX_LEN = 30
_FOLLOWUP_PATTERN = re.compile(
    r"^(那|它|他|她|这个|那个|这些|那些|还有|再|另外|其他|此外|那么|其次|然后)"
    r"|呢[?？]?$"
)
# 知识型问句句式：命中说明用户在问知识，即使与文旅无关也应走检索（检索优先策略）
# 误判历史："什么是猫娘"曾被 LLM 归为闲聊（走联网搜索，本地库没参与），知识型问题宁可错检索不错闲聊
# 注意：只保留纯疑问词句式；"介绍一下/讲讲/说说"后面常接实体（如"介绍一下成都"），统一交给 LLM 层判定
_KNOWLEDGE_QUERY_PATTERN = re.compile(
    r"什么是|是什么|为什么|什么意思|如何|怎么|怎么样|"
    r"是谁|有哪些|解释|含义|由来|起源|区别"
)
# 域外产品词（2026-09-04 P2）：手机/数码/3C 等明显超出文旅知识域的产品类问句
# （"华为P60拍照怎么样"曾命中"怎么样"知识型句式误入文旅域检索，库内必无资料）。
# 必须放在知识型句式兜底之前拦截 → 归闲聊走联网搜索作答。
# 注意"拍照/续航/参数"等泛词不收（"拍照技巧"可能是旅游摄影），只收强产品向词。
# 第三轮扩游戏品类（neg-005 根治）："原神这个游戏好玩吗"曾走文旅检索被模板包装成
# "文旅攻略"，具体游戏评价超出知识域，应归闲聊联网作答；具体游戏名只收头部大作，
# 避免生僻词误伤；"崩坏"单独出现可能是普通词汇，只收"崩坏三/星穹铁道"等专名。
_OFF_DOMAIN_PRODUCT_PATTERN = re.compile(
    r"手机|平板|笔记本|台式机|电脑配置|显卡|处理器|芯片|内存条|硬盘|"
    r"相机|镜头|耳机|音箱|无人机|智能手表|手环|充电宝|路由器|"
    r"手机测评|数码测评|发布会|新机|旗舰机|"
    r"原神|王者荣耀|和平精英|英雄联盟|蛋仔派对|第五人格|明日方舟|"
    r"崩坏三|星穹铁道|我的世界|光遇|"
    r"手游|端游|网游|打游戏|电子游戏|游戏(版本|好玩|测评|攻略|榜单|更新)"
)
# 规划类问句句式：用户要的是"结合出行条件给出可执行安排"，后续由规划分支（API 工具调用）承接。
# 这类问题常不含文旅关键词（如"我明天想去九寨沟，帮我规划一下"），
# 走 LLM 兜底易被误判闲聊，因此在规则层显式命中并归文旅、同时置 is_plan=True。
# 规划类问句（is_plan=True）＝ 用户要「多天行程生成」，须具备明确的行程结构诉求。
# 注意（T7 修正）：此处**不再包含「值得去吗」「值得去玩吗」** —— 它们表达的是
# 「这个地方值不值得去」的咨询型疑问，属攻略问答而非多天行程生成。此前误入本正则
# 会把景点咨询（如「这个景点怎么去？值得去吗？」）当成行程生成，
# 触发目的地抽取与行程编造。
# 这类"出行咨询"信号按文旅/规划意图承接，不进入行程生成。
_PLAN_QUERY_PATTERN = re.compile(
    r"规划|帮我安排|安排一下|行程安排|路线安排|旅游计划|旅行计划|出行计划|"
    r"几日游|一日游|两日游|三日游|四日游|五日游|\d+\s*天\s*\d*\s*晚|"
    r"怎么玩"
)
# ⑤ is_plan 放宽（方案 C 守卫）：**仅用于 LLM 输出之后的确定性收敛**，不参与正则短路。
# 背景（详见 docs/is_plan放宽评估.md）：v2 LLM 曾因 prompt 末尾示例 JSON 的
# `"is_plan": false` 强锚定而恒判 false；修正示例后识别恢复（plan-002~005 转 true），
# 但随之把「第一次去杭州，景点应该怎么安排？」这类**咨询句**也误判成行程。
# 该守卫给出可枚举的判据：LLM 判 is_plan=True 但问句**一个行程定义词都不含** → 降回 false。
# 取舍：**刻意不收「怎么安排」以外的泛请求词**——"怎么安排" 在咨询句里同样出现
# （tr-008 判例），直接收进来守卫即失效；天数/时长（阿拉伯数字 + 中文数词）是区分
# "行程生成"与"知识咨询"的最强信号，与项目"LLM 宽召回 + 代码严收敛"的一贯做法一致。
# 2026-09-10 问题4：守卫放宽为 `_route_plan_signal_hit`（见下），
# "怎么安排" + 跨城两地 这一窄组合也被认作行程信号。
_PLAN_SIGNAL_PATTERN = re.compile(
    r"行程|规划|路线|自由行|"
    r"旅游计划|旅行计划|出行计划|"
    r"几日游|一日游|两日游|三日游|四日游|五日游|六日游|七日游|"
    r"\d+\s*天|\d+\s*晚|"
    r"[一二两三四五六七八九十半几数]{1,3}\s*天|[一二两三四五六七八九十半]{1,3}\s*晚|"
    r"怎么玩|接下来"
)
# 「安排请求」弱信号词（2026-09-10 问题4 新增）：单独出现**不足以**判定行程——
# tr-008「第一次去杭州，景点应该怎么安排？」是纯咨询句，与真行程同形；
# 故仅在同时出现「跨城两地」时才作为行程信号（见 _route_plan_signal_hit）。
_ARRANGE_REQUEST_PATTERN = re.compile(r"怎么安排|如何安排|怎样安排|怎么规划|如何规划|安排几天|安排一下")
# 跨城两地（"从深圳去杭州"）：行程意图的结构性证据（单地名"去杭州"不算）
_CROSSCITY_PATTERN = re.compile(
    r"从\s*[\u4e00-\u9fa5]{2,10}\s*(?:出发)?\s*(?:去|到|前往)\s*[\u4e00-\u9fa5]{2,10}"
)
# 规划问句信号已在 _CROSSCITY_PATTERN / _ARRANGE_REQUEST_PATTERN 中覆盖。
def _route_plan_signal_hit(query: str) -> bool:
    """
    is_plan 守卫的最终判据：问句是否具备「行程意图」的结构性证据。

    原判据只有 `_PLAN_SIGNAL_PATTERN`（行程/规划/天数/怎么玩…）。问题4 的
    「第一次从深圳去杭州，景点应该怎么安排？」不含上述任何行程定义词，
    会被误降为 false；但它是**真行程**（跨城两地 + 安排请求），靠窄规则补回。
    因此补充一条可枚举的窄规则：安排请求 **且** 跨城两地。
    tr-008 这类单地名咨询句不满足，仍按咨询处理（不误伤基线）。
    """
    if _PLAN_SIGNAL_PATTERN.search(query):
        return True
    return bool(_ARRANGE_REQUEST_PATTERN.search(query)) and bool(
        _CROSSCITY_PATTERN.search(query)
    )
# 交通方式变更类追问（"驾车太累了，换成公共交通""改坐高铁"）：
# 明显承接上一轮行程规划，但可能不以"那/它"等追问词开头（如"驾车太累了"），
# 需要单独识别——继承上轮 domain 的同时置 is_plan=True，继续走规划分支。
_TRANSPORT_CHANGE_PATTERN = re.compile(
    r"换(成|坐|乘|用)|改(坐|乘|用)|不想(开车|驾车|自驾|坐车)|不(想|要)自驾|"
    r"(开车|驾车|自驾)(太|好|真)(累|麻烦|辛苦|远|久|慢)|"
    r"公共交通|坐(高铁|动车|火车|飞机|大巴)|乘(高铁|动车|火车|飞机|大巴)"
)
# 天气/出行条件类问句（"下周一天气如何""有台风吗""三亚会下雨吗"）：
# 需要实时天气数据才能回答，继承上轮 domain 的同时置 is_plan=True——
# 文旅图的实时天气工具挂在规划分支上，借道它拿到天气简报；生成端由 prompt 控制"只问天气时输出简短回答"。
_WEATHER_QUERY_PATTERN = re.compile(
    r"天气|下雨|台风|暴雨|降雨|气温|温度|热不热|冷不冷|穿什么|防晒|适合出行|适合旅游|适合玩"
)
# 当前时间/日期类问句（"今天是几月几号""现在几点""今天星期几""明天是几号"）：
# 需要确定性的当前时间才能回答，绝不能走 LLM 训练记忆（实测答成"8月28日"）。
# 命中后 domain=chitchat + is_datetime_query=True，由 node_datetime_answer 用系统时钟直答。
# 时间"询问词"：几号/几点/星期几/周几/礼拜几 + 时间/日期类词。
_DATETIME_QUERY_PATTERN = re.compile(
    r"(几月几号|几号|多少号|几点钟?|星期几|周几|礼拜几|什么日期|日期|时间)"
)
# 时间问句的"活动语境"排除词：出现这些词说明时间点挂在出行/营业等活动上
# （"几点出发""几点开门""几点下班"），不是纯时间询问，不应被直答节点拦截（应随上文走规划/检索）。
_DATETIME_ACTIVITY_PATTERN = re.compile(
    r"出发|到达|抵达|开门|关门|营业|闭园|闭馆|开始|结束|集合|发车|起飞|降落|"
    r"检票|演出|上班|下班|上课|下课|关门"
)
# 时间问句的"时间锚点"：只拦锚定"现在/今天/明天..."的相对时间询问，
# 保证 node_datetime_answer 一定能用系统时钟确定性换算；无锚点的"几号/几点"（如"几点出发"）放行。
_DATETIME_ANCHOR_PATTERN = re.compile(
    r"今天|今日|现在|目前|当前|此刻|明天|明日|后天|大后天|昨天|昨日|前天|今晚|明晚"
)
# 节日名（含公历固定节日与农历节日）：带"几月几号"等日期问词时也属于确定性可答范围
# （公历节日直接算本年日期，农历节日只答农历日期并提示以日历为准，杜绝 LLM 编造公历日期）
_FESTIVAL_NAME_PATTERN = re.compile(
    r"元旦|春节|元宵节?|情人节|妇女节|植树节|清明节?|劳动节|青年节|儿童节|建党节|建军节|"
    r"七夕节?|中元节|教师节|国庆节|中秋节|重阳节|腊八节|圣诞节|除夕"
)


# 指代不明咨询问句（cl-002）：以"那边/那里/这边/这里/当地/此地"开头的短咨询句。
# 只有追问继承（上一层）未命中才会判定——无可继承历史时指代没有先行词，
# 不能按字面词走闲聊闲扯（实测"那边天气怎么样"被"天气"闲聊词命中后编造"阳光明媚"），
# 应确定性反问补充地点。限定句首：避免"丽江那边天气怎么样"这类地名在前的正常问句误伤。
_DEIXIS_CONSULT_PATTERN = re.compile(r"^(那边|那里|这边|这里|当地|此地)")


def _is_pure_datetime_query(query: str) -> bool:
    """是否为可用系统时钟确定性回答的纯时间/日期询问。

    判断条件：含时间询问词 + 不含出行/营业等"活动语境" + 带今天/明天等时间锚点
    （纯时钟问句"现在几点"可无日期锚点，单独放行；"中秋节是几月几号"等带节日的
    日期问句有专门映射表兜底，也放行）。
    """
    if not _DATETIME_QUERY_PATTERN.search(query):
        return False
    if _DATETIME_ACTIVITY_PATTERN.search(query):
        return False
    if _DATETIME_ANCHOR_PATTERN.search(query):
        return True
    # 无日期锚点但纯问"几点/时间"（"现在几点""几点了""什么时间了"）
    if re.search(r"(现在|目前|此刻)?几点钟?|几点了|现在时间|什么时间|时间是多少", query):
        return True
    # 带节日名的日期问句（"中秋节是几月几号""国庆节是几号"）
    if _FESTIVAL_NAME_PATTERN.search(query):
        return True
    return False


# 行程微调类延续追问："再安排一天""多玩一天""第三天改成灵隐寺""太赶了""预算砍半"。
# 这类句子是上轮行程规划的直接延续，应继承 is_plan；而"那门票多少钱""还有别的吗"这类
# 泛追问只是接着问知识，置 is_plan 会把它们错变成完整行程生成，故单独识别、不混为一类。
_ITINERARY_TWEAK_PATTERN = re.compile(
    r"(再|又|多|追加|增加|加|缩减|减少|少|改|换成|调|去掉|删掉).{0,4}(一|两|二|三|四|五|\d+)?\s*(天|晚|日|夜)"
    r"|第[一二三四五六七八九十\d]+[天日]"
    r"|(太赶|太满|太紧|轻松点|紧凑点|预算|少花|砍半|省钱)"
)
# 纠正类追问（"我要去杭州，不是成都""不是成都""我说的是杭州""搞错了"）：
# 用户对上一轮的实体/结论做否定与更正。此类句子自身常常不带规划词，
# 不继承就会退化成普通检索问答（实测"我要去杭州，不是成都"被 LLM 兜底判 tourism 后
# 丢失 is_plan，行程变成 4 天普通问答、日期与天气全丢），因此与追问词并列作为继承信号。
# 只收"否定/更正"强信号，避开"是不是/要不要/有没有"等疑问句式（"故宫是不是要门票"不得命中）。
_CORRECTION_PATTERN = re.compile(
    r"(我要去的是|我说的是|去的是|说的是|应该是|其实是)"
    r"|，\s*不是|，\s*错了|，\s*不对"
    r"|搞错|弄错|说错了|写错了|看错了"
)
# 困惑/澄清类输入（"？""？？""什么意思""你说什么""没懂""不对吧"）：
# 用户对上一轮回答表示不理解/质疑。此时绝不能当作新问题走检索/改写
# （实测"？"被 LLM 误判 tourism 后把历史行程再生成一遍）。
# 命中后 domain=chitchat + is_confusion=True，澄清逻辑结合上一条助手回答与当前时间纠正。
_CONFUSION_PATTERN = re.compile(
    r"^[?？!！~～。.、\s]{1,6}$|"  # 纯符号/极短（？）
    r"^(什么意思|你说什么|啥意思|没听清|没听懂|没明白|不理解|不对吧|错了吧|为什么不对|"
    r"再说一遍|能再说一次|什么鬼|怎么回事|在说什么|啥|嗯？|啊？)[?？!！]?$"
)
# 征询句式（"可以吗/行不行/能不能"）：答案需要先给可行性结论再展开，保留在改写与行程拼装端。
# 供 itinerary_service 等消费（行程拼装时对征询句先给结论）。
CONSULT_PATTERN = re.compile(r"(可以吗|可以么|行不行|行吗|能不能|可不可以|合适吗|可行吗|是否可行|靠谱吗|允许吗)")

@step_log("match_by_rule")
def match_by_rule(query: str) -> str:
    """
    基于关键词的快速意图匹配。
    优先级：闲聊白名单 > 文旅 > 知识型句式兜底（检索优先）。
    知识型句式放最后，避免"这本书怎么样"这类问题被"怎么样"误抢。
    """
    lowered = query.lower()
    for keyword in _CHITCHAT_KEYWORDS:
        if keyword in lowered:
            return DOMAIN_CHITCHAT
    tourism_hit = any(kw in lowered for kw in _TOURISM_KEYWORDS)
    # 文旅关键词优先（更具体）
    if tourism_hit:
        return DOMAIN_TOURISM
    # 域外产品词拦截（P2）：手机/数码等 3C 产品问句先于知识型句式兜底，
    # 归闲聊走联网搜索作答（本地库必无此类资料，误入文旅域只会产出无出处回答）
    if _OFF_DOMAIN_PRODUCT_PATTERN.search(query):
        logger.info(f"域外产品词命中，拦截归闲聊走联网: [{query[:30]}]")
        return DOMAIN_CHITCHAT
    # 知识型句式兜底：宁可错检索，不可错闲聊（闲聊走联网搜索，本地库完全不参与）
    if _KNOWLEDGE_QUERY_PATTERN.search(query):
        logger.info(f"知识型句式命中，检索优先归文旅: [{query[:30]}]")
        return DOMAIN_TOURISM
    # 规划类句式：归文旅（规划本质是出行安排），is_plan 由 classify_intent 统一写入
    if _PLAN_QUERY_PATTERN.search(query):
        logger.info(f"规划类句式命中，归文旅: [{query[:30]}]")
        return DOMAIN_TOURISM
    return ""


def _call_intent_model(query: str, history_text: str) -> str:
    """执行一次 LLM 意图分类，返回三分类之一（内部函数，供缓存包装调用）。"""
    client = llm_provider.chat(json_mode=True)
    prompt = load_prompt(
        "tourism/intent_route",
        history_text=history_text,
        query=query,
    )
    messages = [
        SystemMessage(content="你是智能问答系统的意图识别器，只能输出合法 JSON。"),
        HumanMessage(content=prompt),
    ]
    result = (client | JsonOutputParser()).invoke(messages)
    domain = result.get("domain", "")
    if domain in (DOMAIN_TOURISM, DOMAIN_CHITCHAT):
        # 双保险：LLM 把知识型问句判成闲聊时推翻（闲聊走联网搜索，本地库不参与，宁可检索）
        if domain == DOMAIN_CHITCHAT and _KNOWLEDGE_QUERY_PATTERN.search(query):
            logger.warning(f"LLM 判闲聊但命中知识型句式，推翻为 tourism: [{query[:30]}]")
            return DOMAIN_TOURISM
        return domain
    logger.warning(f"意图识别返回非法域[{domain}]，默认走文旅检索")
    return DOMAIN_TOURISM


@step_log("classify_by_llm")
def classify_by_llm(query: str, history_text: str = "") -> str:
    """
    LLM 兜底意图识别，返回三分类之一。
    相同(问题, 历史)的分类结果高度稳定，走缓存可省掉一次模型往返（约 0.8~1.5s）。
    """
    try:
        return cached_invoke(
            namespace="intent_route",
            cache_parts=(query, history_text),
            producer=lambda: _call_intent_model(query, history_text),
            cache_label=query[:20],
            semantic=True,
        )
    except Exception as e:
        logger.warning(f"LLM 意图识别失败，默认走文旅检索：{e}")
        return DOMAIN_TOURISM


@step_log("inherit_domain_if_followup")
def inherit_domain_if_followup(query: str, session_id: str) -> tuple[str, bool]:
    """
    追问继承：本轮是简短追问/纠正且上一轮已有明确业务域时，直接继承上一轮域与规划标记。

    追问（"那门票多少钱""还有别的吗""我要去杭州，不是成都"）自身几乎不带领域信号，
    走关键词会乱匹配、走 LLM 要付 0.8~1.5s 往返，而上下文已经给出了确定答案。

    Args:
        query: 用户当前问题。
        session_id: 会话 ID，用于读取上一轮落库的 domain / is_plan。

    Returns:
        tuple[str, bool]: (继承到的域, 上轮是否为规划类)；无法继承时返回 ("", False)。
    """
    if not session_id or not query or len(query) > _FOLLOWUP_MAX_LEN:
        return "", False
    # 常规追问词 / 交通变更类 / 天气类 / 纠正类问句都视为承接上文
    if not (_FOLLOWUP_PATTERN.search(query) or _TRANSPORT_CHANGE_PATTERN.search(query)
            or _WEATHER_QUERY_PATTERN.search(query) or _CORRECTION_PATTERN.search(query)):
        return "", False

    try:
        from app.infra.persistence import history_repository

        recent_messages = history_repository.list_recent(session_id, limit=3)
    except Exception as e:
        logger.warning(f"读取历史 domain 失败，放弃追问继承：{e}")
        return "", False

    # list_recent 返回时间正序（旧→新），从末尾往回取最近一轮的有效标记
    for message in reversed(recent_messages):
        last_domain = message.get("domain") or ""
        # 只继承检索型域：闲聊后用户很可能转入正题，继承 chitchat 反而会挡掉检索
        if last_domain == DOMAIN_TOURISM:
            last_is_plan = bool(message.get("is_plan"))
            logger.info(
                f"追问继承命中：[{query[:20]}] 继承上轮域 {last_domain}，is_plan={last_is_plan}"
            )
            return last_domain, last_is_plan
    return "", False


@step_log("classify_intent")
def classify_intent(state: dict) -> str:
    """
    意图路由服务总入口：
    规划类标记 → 追问继承 → 关键词规则 → LLM 兜底（带缓存）。
    命中规划类句式时写 state['is_plan'] = True，供后续规划分支（API 工具调用）消费。
    """
    query = state.get("original_query") or state.get("rewritten_query") or ""
    if not query:
        logger.warning("original_query 为空，默认走文旅检索")
        return DOMAIN_TOURISM

    # 第零层：规划类标记（零成本，仅标记不改域；放在最前，追问短句"那帮我规划一下"也能覆盖）
    if _PLAN_QUERY_PATTERN.search(query):
        state["is_plan"] = True
        logger.info(f"规划类问句标记 is_plan=True: [{query[:30]}]")

    # 第 0.5 层：纯时间/日期询问 → 确定性直答节点（系统时钟，零 LLM，杜绝幻觉日期）。
    # 必须放在"追问继承"之前：实测"今天几号呢"这类带"呢"尾的句子会被追问正则误判为文旅追问。
    if _is_pure_datetime_query(query):
        state["is_datetime_query"] = True
        state["is_confusion"] = False
        logger.info(f"纯时间/日期询问命中，走确定性直答: [{query[:30]}]")
        return DOMAIN_CHITCHAT

    # 第 0.6 层：困惑/澄清类输入（"？""没懂""不对吧"）→ 走澄清节点。
    # 放在所有检索判断之前：绝不能当作新问题走检索/改写/行程重生成。
    if _CONFUSION_PATTERN.match(query):
        state["is_confusion"] = True
        state["is_datetime_query"] = False
        logger.info(f"困惑/澄清输入命中，走澄清节点: [{query[:30]}]")
        return DOMAIN_CHITCHAT


    # 第二层：追问继承（零成本）——承接上文的短问句/纠正句直接复用上轮业务域与规划标记
    followup_domain, last_is_plan = inherit_domain_if_followup(query, state.get("session_id", ""))
    if followup_domain:
        if followup_domain == DOMAIN_TOURISM:
            # 交通变更类追问（"换成公共交通"）＝上轮行程规划的延续，继续走规划分支
            if _TRANSPORT_CHANGE_PATTERN.search(query):
                state["is_plan"] = True
                logger.info(f"交通变更追问，继承规划意图 is_plan=True: [{query[:30]}]")
            # 天气/出行条件类问句 → 借道规划分支拿实时天气数据（生成端 prompt 控制简短回答）
            elif _WEATHER_QUERY_PATTERN.search(query):
                state["is_plan"] = True
                logger.info(f"天气问句追问，借道天气工具 is_plan=True: [{query[:30]}]")
            # 上轮是规划类且本轮是"纠正/微调"型延续（"我要去杭州，不是成都""再安排一天"）：
            # 这些句式无法被单一正则穷举，靠上轮 is_plan + 延续信号共同确认，
            # 实测漏继承会退化成普通检索问答，行程形态、天数与日期全丢。
            elif last_is_plan and (_CORRECTION_PATTERN.search(query)
                                   or _ITINERARY_TWEAK_PATTERN.search(query)):
                state["is_plan"] = True
                logger.info(f"延续上轮规划（纠正/微调类），继承 is_plan=True: [{query[:30]}]")
        return followup_domain

    # 第 2.5 层：指代不明咨询问句 → 确定性澄清反问（不联网、不闲扯、不走检索）。
    # 只有追问继承未命中才会走到这里：有可继承历史时"那边"有先行词，已在上层正常继承。
    if _DEIXIS_CONSULT_PATTERN.match(query):
        state["is_unresolved_deixis"] = True
        state["is_confusion"] = False
        state["is_datetime_query"] = False
        logger.info(f"指代不明咨询句命中，走澄清反问: [{query[:30]}]")
        return DOMAIN_CHITCHAT

    # 第三层：关键词规则匹配
    rule_domain = match_by_rule(query)
    if rule_domain:
        logger.info(f"意图规则命中: [{query[:30]}] -> {rule_domain}")
        # 域外产品词命中且归闲聊（neg-002 兜底）：标记给闲聊答案追加"以官方参数为准"限定语
        if rule_domain == DOMAIN_CHITCHAT and _OFF_DOMAIN_PRODUCT_PATTERN.search(query):
            state["is_off_domain_product"] = True
            logger.info(f"域外产品词命中，闲聊答案将追加官方参数提示: [{query[:30]}]")
        return rule_domain

    # 第四层：LLM 兜底（同输入走缓存，命中时零延迟）
    logger.info(f"规则未命中，使用 LLM 兜底判断: [{query[:30]}]")
    history_text = state.get("history_text", "")
    domain = classify_by_llm(query, history_text=history_text)
    logger.info(f"LLM 兜底结果: [{query[:30]}] -> {domain}")
    return domain


# ============================================================
# B1 v2 意图路由（LLM 主导 + 正则短路/降级）—— 2026-09-09
# ------------------------------------------------------------
# 开关：INTENT_ROUTER_LLM_FIRST=true 时启用 LLM 主导；默认 false 走既有 v1 链路
# （护栏优先：v1 的确定性规则零回归风险，v2 成熟后再切默认）。
# 层次（v2）：
#   确定性护栏层（datetime/confusion/追问继承/deixis）→ 无论开关都优先于 LLM；
#   LLM v2 单次输出（改写+域判定+数据源三档+工具+is_plan+need_clarify）；
#   失败/非法 → 降级 match_by_rule → 旧 classify_by_llm → tourism 兜底。
# 说明：与 v1 共享同套护栏正则，本文件 v1 分支保持原样以保基线。
# ============================================================

#: v2 开关（LLM 主导意图路由）。已拍板切换默认 true：route_info 是 B2/B3 域判定的唯一数据源。
#: 经 T5 全量验收（Acc 1.0 + 路由域与 baseline 差异 0/40），env 可回退 false 作旁路。
INTENT_ROUTER_LLM_FIRST = env_bool("INTENT_ROUTER_LLM_FIRST", default=True)
#: v2 合法枚举
V2_DOMAINS = (DOMAIN_TOURISM, DOMAIN_CHITCHAT)
V2_SOURCE_POLICIES = ("kb", "web", "kb_then_web")
V2_TOOLS = {"none", "weather", "route", "poi"}


def parse_route_v2_result(result) -> dict | None:
    """
    校验并归一 LLM v2 单次输出；任意外形非法返回 None（调用方走降级，不抛异常）。
    """
    if not isinstance(result, dict):
        return None
    domains = result.get("domains")
    if not isinstance(domains, list) or not domains:
        return None
    norm_domains = [d for d in domains if isinstance(d, str) and d in V2_DOMAINS]
    if not norm_domains:
        return None
    policy = result.get("source_policy")
    if policy not in V2_SOURCE_POLICIES:
        policy = "kb"
    tools_raw = result.get("tools") or []
    tools = [t for t in tools_raw if isinstance(t, str) and t in V2_TOOLS] or ["none"]
    return {
        "rewritten_query": str(result.get("rewritten_query") or "").strip(),
        "domains": norm_domains[:2],  # 主域在首位（B2 消费完整列表）
        "source_policy": policy,
        "tools": tools,
        "is_plan": bool(result.get("is_plan")),
        "need_clarify": bool(result.get("need_clarify")),
        "reason": str(result.get("reason") or "")[:200],
    }


def _call_route_v2_model(query: str, history_text: str) -> dict:
    """执行一次 LLM v2 路由调用（json_mode 输出 schema v2）。"""
    client = llm_provider.chat(json_mode=True)
    prompt = load_prompt("tourism/intent_route_v2", history_text=history_text, query=query)
    messages = [
        SystemMessage(content="你是智能问答系统的统一路由器，只能输出合法 JSON。"),
        HumanMessage(content=prompt),
    ]
    return (client | JsonOutputParser()).invoke(messages)


#: v2 路由 prompt 版本标记——**参与缓存键**。缓存键原本只含「问题 + 历史」，不含 prompt 内容，
#: 因此**每次改动 `intent_route_v2.prompt` 都必须递增本标记**，否则旧 prompt 的缓存结果会被
#: 继续命中（本次方案 C 若不递增，修复会被旧缓存完全掩盖）。
_ROUTE_PROMPT_REV = "r3"  # r2 = 2026-09-10 方案C；r3 = 2026-09-10 问题4


def classify_route_v2(query: str, history_text: str = "") -> dict | None:
    """
    LLM v2 路由（带语义缓存：namespace=intent_route_v2，同(问题,历史)免重复往返）。
    失败/异常返回 None，不抛。
    """
    try:
        raw = cached_invoke(
            namespace="intent_route_v2",
            cache_parts=(query, history_text, _ROUTE_PROMPT_REV),
            producer=lambda: _call_route_v2_model(query, history_text),
            cache_label=query[:20],
            semantic=True,
        )
        return parse_route_v2_result(raw)
    except Exception as e:
        logger.warning(f"LLM v2 意图路由失败，走降级链路：{e}")
        return None


def _apply_route_v2(state: dict, route: dict) -> str:
    """把 v2 结果落到 state（route_info 供 B2 域判定/B3 衔接消费），返回主域。"""
    is_plan = bool(route.get("is_plan"))
    tools = list(route.get("tools") or ["none"])
    domains = list(route.get("domains") or [])
    query = state.get("original_query") or state.get("rewritten_query") or ""
    # 方案 C 守卫（见 docs/is_plan放宽评估.md）：LLM 判 is_plan=True，但问句不含任何
    # 「行程定义词」→ 视为把咨询类误当行程（示例锚定修正后 LLM 判定转宽松的副作用），
    # 确定性降回 False；tools 里的 route 一并去掉，保证"工具决策"与"行程判定"自洽。
    # 2026-09-10：判据放宽为 `_route_plan_signal_hit`（+ 跨城两地的窄组合）。
    if is_plan and not _route_plan_signal_hit(query):
        logger.info(
            f"is_plan 守卫：LLM 判 true 但问句无行程定义词，降回 false（并去 tools.route）: [{query[:30]}]"
        )
        is_plan = False
        tools = [t for t in tools if t != "route"] or ["none"]
    state["route_info"] = {
        "rewritten_query": route.get("rewritten_query", ""),
        "domains": domains,
        "source_policy": route.get("source_policy", "kb"),
        "tools": tools,
        "is_plan": is_plan,
        "need_clarify": route.get("need_clarify", False),
    }
    if is_plan:
        state["is_plan"] = True
    if route.get("need_clarify"):
        state["need_clarify"] = True
    return domains[0] if domains else DOMAIN_TOURISM


@step_log("classify_intent_v2")
def classify_intent_v2(state: dict) -> str:
    """
    v2 意图路由主决策入口：确定性护栏层 → LLM v2 主导 → 规则 → 旧 LLM 兜底。
    与 classify_intent（v1）返回同一 domain 口径，node 层无需区分。
    """
    query = state.get("original_query") or state.get("rewritten_query") or ""
    if not query:
        logger.warning("original_query 为空，默认走文旅检索")
        return DOMAIN_TOURISM

    # ---- 确定性护栏层（与 v1 同序同步维护；零成本、高安全，不交给 LLM）----
    if _PLAN_QUERY_PATTERN.search(query):
        state["is_plan"] = True
        logger.info(f"v2 规划类问句标记 is_plan=True: [{query[:30]}]")
    if _is_pure_datetime_query(query):
        state["is_datetime_query"] = True
        state["is_confusion"] = False
        logger.info(f"v2 纯时间/日期询问命中，走确定性直答: [{query[:30]}]")
        return DOMAIN_CHITCHAT
    if _CONFUSION_PATTERN.match(query):
        state["is_confusion"] = True
        state["is_datetime_query"] = False
        logger.info(f"v2 困惑/澄清输入命中，走澄清节点: [{query[:30]}]")
        return DOMAIN_CHITCHAT
    followup_domain, last_is_plan = inherit_domain_if_followup(query, state.get("session_id", ""))
    if followup_domain:
        if followup_domain == DOMAIN_TOURISM:
            if _TRANSPORT_CHANGE_PATTERN.search(query):
                state["is_plan"] = True
            elif _WEATHER_QUERY_PATTERN.search(query):
                state["is_plan"] = True
            elif last_is_plan and (_CORRECTION_PATTERN.search(query)
                                   or _ITINERARY_TWEAK_PATTERN.search(query)):
                state["is_plan"] = True
        logger.info(f"v2 追问继承命中: [{query[:20]}] -> {followup_domain}")
        return followup_domain
    if _DEIXIS_CONSULT_PATTERN.match(query):
        state["is_unresolved_deixis"] = True
        state["is_confusion"] = False
        state["is_datetime_query"] = False
        logger.info(f"v2 指代不明咨询句命中，走澄清反问: [{query[:30]}]")
        return DOMAIN_CHITCHAT

    # ---- 强信号关键词短路（v1 已验证的高置信信号，零 LLM；v2 只把「规则力所不及」
    #      的语义判断交给 LLM：文化知识 vs 闲聊 vs 规划 vs 域外）----
    lowered = query.lower()
    if _OFF_DOMAIN_PRODUCT_PATTERN.search(query):
        state["is_off_domain_product"] = True
        logger.info(f"v2 域外产品词短路 -> chitchat: [{query[:30]}]")
        return DOMAIN_CHITCHAT
    if any(kw in lowered for kw in _CHITCHAT_KEYWORDS):
        logger.info(f"v2 闲聊白名单短路 -> chitchat: [{query[:30]}]")
        return DOMAIN_CHITCHAT

    # ---- LLM v2 主导 ----
    route = classify_route_v2(query, state.get("history_text", ""))
    if route:
        domain = _apply_route_v2(state, route)
        info = state["route_info"]
        logger.info(
            f"v2 LLM 路由: [{query[:30]}] -> domain={domain} domains={route['domains']} "
            f"policy={route['source_policy']} tools={info['tools']} "
            f"is_plan={info['is_plan']}（LLM 原始 is_plan={route['is_plan']}）"
        )
        # 双保险（沿用 v1）：知识型句式判成闲聊/纯联网时推翻为 tourism（本地库必须参与）
        if domain == DOMAIN_CHITCHAT and _KNOWLEDGE_QUERY_PATTERN.search(query):
            logger.warning(f"v2 判闲聊但命中知识型句式，推翻为 tourism: [{query[:30]}]")
            state["route_info"]["domains"] = [DOMAIN_TOURISM]
            return DOMAIN_TOURISM
        # 域外产品词归闲聊时（v1 同款）：闲聊答案追加"以官方为准"限定语
        if domain == DOMAIN_CHITCHAT and _OFF_DOMAIN_PRODUCT_PATTERN.search(query):
            state["is_off_domain_product"] = True
            logger.info(f"v2 域外产品词命中，闲聊答案将追加官方参数提示: [{query[:30]}]")
        return domain

    # ---- 降级：关键词规则 → 旧 LLM 兜底（tourism/chitchat） ----
    rule_domain = match_by_rule(query)
    if rule_domain:
        logger.info(f"v2 降级规则命中: [{query[:30]}] -> {rule_domain}")
        if rule_domain == DOMAIN_CHITCHAT and _OFF_DOMAIN_PRODUCT_PATTERN.search(query):
            state["is_off_domain_product"] = True
        return rule_domain
    domain = classify_by_llm(query, state.get("history_text", ""))
    logger.info(f"v2 降级旧 LLM 兜底: [{query[:30]}] -> {domain}")
    return domain


def classify_intent_dispatch(state: dict) -> str:
    """
    意图路由统一入口（node_intent_route 调用）：
    INTENT_ROUTER_LLM_FIRST=true → classify_intent_v2（LLM 主导）；
    默认 false → classify_intent（v1 规则链路，行为与历史完全一致）。
    """
    if INTENT_ROUTER_LLM_FIRST:
        return classify_intent_v2(state)
    return classify_intent(state)
