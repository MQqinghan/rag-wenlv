"""
闲聊回答服务模块：处理被路由为 chitchat 的日常对话（问候/感谢/自我介绍等）。
只消费联网搜索结果，不访问本地知识库；支持流式/非流式输出与历史落库。
"""
import re

from app.infra.llm import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.rag.common.history_text_utils import build_history_text
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.sse_utils import SSEEvent, is_cancelled, push_to_session
from app.shared.utils.task_utils import set_task_result
from app.shared.utils.text_utils import BlankLineCompressor, normalize_answer_text

# 引用标记 regex：流式 delta 级别剥离【第N块】和【N】
_CITATION_PATTERN = re.compile(r"(?:\s*【(?:第\d+块|\d+)】)+")

# 闲聊场景最多取前 N 条网页结果做参考，避免 prompt 过长
CHITCHAT_WEB_TOP: int = 5
# 闲聊场景注入的最近对话条数（给 LLM 提供话题上下文，也用于日期问句判断是否在延续上文的出行安排）
CHITCHAT_HISTORY_LIMIT: int = 6

# 指代不明咨询句（cl-002）的确定性澄清反问：指代无先行词时不能闲聊闲扯
# （实测"那边天气怎么样"被"天气"闲聊词命中后编造"阳光明媚"），直接反问补充地点。
_DEIXIS_CLARIFY_REPLY = (
    "您说的「那边」我还不清楚具体指哪里——请补充一下具体地点（城市或景区名），我再帮您查询。"
)

# 时效类问句（neg-004）：闲聊域回答时间类问题必须带"可能变动/以官方为准"限定，
# 严禁无出处的确定性日期断言；代码级追加限定语（不依赖模型遵循，与班次号剥离同思路）。
_TIME_SENSITIVE_PATTERN = re.compile(
    r"什么时候|何时|几号|几月|几点|哪一天|哪年|发布会|发售|上映|开播|上线|开售|截止|开幕|闭幕"
)
_TIME_SENSITIVE_SUFFIX = (
    "\n\n（提示：以上时间类信息可能随官方安排变动，请以官方渠道最新发布为准。）"
)

# 未来事件类问句（neg-004 根治）：发布会/上映/发售等未定档时间 LLM 极易编造具体日期
# （实测追加限定语后仍先断言"定在 9 月 10 日"，judge 判 fabricated），此类问题直接给
# 确定性"无法确认"答复：不调 LLM、零编造（与 cl-002 澄清反问同款零 LLM 直答模式）。
_FUTURE_EVENT_PATTERN = re.compile(
    r"发布会|发布时间|发售|上映|开播|上线|开售|截止|开幕|闭幕|什么时候开|哪天开"
)
_TIME_UNCONFIRMED_REPLY = (
    "这类具体活动时间我无法给出确切答案——相关安排可能随官方发布随时变动，"
    "请以官方渠道最新消息为准。"
)

# 域外产品类问句（neg-002 兜底）：闲聊联网作答易给出无出处具体参数断言（如"4800万像素
# 主摄"），统一追加"以官方发布为准"限定语（代码级追加，不依赖模型遵循）。
_OFF_DOMAIN_SUFFIX = (
    "\n\n（提示：以上为联网信息整理，具体参数与配置请以官方发布为准。）"
)


@step_log("generate_chitchat_answer")
def generate_chitchat_answer(state: dict) -> dict:
    """
    闲聊回答服务主入口：
    1. 组装上下文（取前 N 条联网结果）
    2. 渲染 chitchat prompt
    3. 流式/非流式生成回答
    4. 保存历史记录
    """
    question = state.get("original_query") or state.get("rewritten_query") or ""
    if not question:
        raise ValueError("生成闲聊回答需要 original_query/rewritten_query")

    # 指代不明咨询句（cl-002）：确定性澄清反问，不联网、不调模型
    if state.get("is_unresolved_deixis"):
        return _reply_deixis_clarify(state)

    # 未来事件类问句（neg-004）：确定性"无法确认"答复，不调 LLM、零编造
    if _FUTURE_EVENT_PATTERN.search(question):
        return _reply_time_unconfirmed(state)

    web_docs = state.get("web_search_docs", []) or []
    top_docs = web_docs[:CHITCHAT_WEB_TOP]

    context_str = build_web_context(top_docs)
    # 最近对话注入：给模型话题上下文；日期问句须据【当前日期】作答，禁止凭训练记忆猜
    history_text = build_history_text(
        history_repository.list_recent(state["session_id"], limit=CHITCHAT_HISTORY_LIMIT)
    )
    prompt = load_prompt(
        "common/chitchat_answer",
        question=question,
        context=context_str,
        current_date=state.get("current_date") or "未知",
        history=history_text or "（无）",
    )

    # 生成回答（流式逐字推送，用户点停止时中断；非流式一次性调用）
    answer = final_answer(state, prompt)

    # 时效类问句（neg-004）：代码级追加"可能变动/以官方为准"限定，杜绝无出处确定性日期断言
    if _TIME_SENSITIVE_PATTERN.search(question):
        answer = answer.rstrip() + _TIME_SENSITIVE_SUFFIX
        if state.get("is_stream", False):
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": _TIME_SENSITIVE_SUFFIX})
        set_task_result(state["session_id"], "answer", answer)

    # 域外产品类问句（neg-002 兜底）：追加"以官方发布为准"限定语
    if state.get("is_off_domain_product"):
        answer = answer.rstrip() + _OFF_DOMAIN_SUFFIX
        if state.get("is_stream", False):
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": _OFF_DOMAIN_SUFFIX})
        set_task_result(state["session_id"], "answer", answer)

    state["answer"] = answer
    state["image_urls"] = []
    # 历史落库（role=assistant）
    history_repository.save_message(
        session_id=state["session_id"],
        role="assistant",
        text=answer,
        rewritten_query=question,
        item_names=[],
        image_urls=[],
    )
    return state


def _reply_time_unconfirmed(state: dict) -> dict:
    """
    未来事件类问句（neg-004）的确定性"无法确认"答复：不调 LLM、零编造。
    发布会/上映等未定档时间是 LLM 编造高发区，追加限定语兜不住（模型先断言后提示）。
    """
    session_id = state["session_id"]
    set_task_result(session_id, "answer", _TIME_UNCONFIRMED_REPLY)
    if state.get("is_stream", False):
        push_to_session(session_id, SSEEvent.DELTA, {"delta": _TIME_UNCONFIRMED_REPLY})
    state["answer"] = _TIME_UNCONFIRMED_REPLY
    state["image_urls"] = []
    history_repository.save_message(
        session_id=session_id,
        role="assistant",
        text=_TIME_UNCONFIRMED_REPLY,
        rewritten_query=state.get("original_query") or state.get("rewritten_query") or "",
        item_names=[],
        image_urls=[],
        domain=state.get("domain", "chitchat"),
    )
    return state


def _reply_deixis_clarify(state: dict) -> dict:
    """
    指代不明咨询句（cl-002）的确定性澄清反问：不联网、不调模型，
    直接反问补充地点（与 node_datetime_answer 同款"零 LLM 直答"模式）。
    """
    session_id = state["session_id"]
    set_task_result(session_id, "answer", _DEIXIS_CLARIFY_REPLY)
    if state.get("is_stream", False):
        push_to_session(session_id, SSEEvent.DELTA, {"delta": _DEIXIS_CLARIFY_REPLY})
    state["answer"] = _DEIXIS_CLARIFY_REPLY
    state["image_urls"] = []
    history_repository.save_message(
        session_id=session_id,
        role="assistant",
        text=_DEIXIS_CLARIFY_REPLY,
        rewritten_query=state.get("original_query") or state.get("rewritten_query") or "",
        item_names=[],
        image_urls=[],
        domain=state.get("domain", "chitchat"),
    )
    return state


@step_log("build_web_context")
def build_web_context(web_docs: list[dict]) -> str:
    """将联网搜索结果拼为参考文本；为空时返回占位说明。"""
    if not web_docs:
        return "（无相关联网信息）"
    parts = []
    for number, doc in enumerate(web_docs, start=1):
        title = doc.get("title") or "无标题"
        snippet = doc.get("snippet") or doc.get("content") or ""
        url = doc.get("url") or ""
        parts.append(f"第{number}条: 标题:{title}\n内容:{snippet}\n来源:{url}")
    return "\n\n".join(parts)


@step_log("final_answer")
def final_answer(state: dict, prompt: str) -> str:
    """调用大模型生成闲聊回答，支持流式与普通两种模式。"""
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    lm_client = llm_provider.chat().bind(max_tokens=1500)  # 限制生成长度，防止过度输出拖慢响应
    final_result = ""
    if is_stream:
        blank_comp = BlankLineCompressor()
        for chunk in lm_client.stream(prompt):
            if is_cancelled(session_id):
                logger.info(f"用户停止生成，中断闲聊回答输出，session_id={session_id}")
                break
            delta_content = chunk.content
            # per-delta 轻清理：剥离引用标记 + 压缩连续空行（模型常输出 \n\n\n\n 大片空行）
            delta_content = blank_comp.feed(_CITATION_PATTERN.sub("", delta_content))
            final_result += delta_content
            if delta_content:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        response = lm_client.invoke(prompt)
        final_result = response.content
    final_result = normalize_answer_text(final_result)
    set_task_result(session_id, "answer", final_result)
    return final_result
