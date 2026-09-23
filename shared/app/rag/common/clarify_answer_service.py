"""
澄清回答服务模块：node_clarify 专用。

处理被路由为困惑/澄清类输入（"？""没明白""不对吧"）的消息。
核心思路：用户对上一条回答没看懂时，绝不能当作新问题走检索/改写（实测"？"被误判为文旅后
把历史行程重生成一遍），而是结合「助手上一条回答 + 当前日期」用大白话重新解释；
若上一条回答存在时间/日期表述错误（模型训练记忆幻觉），先纠正再重述。
"""
from app.infra.llm import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.rag.common.history_text_utils import build_history_text
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.sse_utils import SSEEvent, is_cancelled, push_to_session
from app.shared.utils.task_utils import set_task_result
from app.shared.utils.text_utils import BlankLineCompressor, normalize_answer_text

# 澄清场景最多回溯的历史条数（找"助手上一条回答"用）
CLARIFY_HISTORY_LIMIT: int = 10

# 澄清回答比闲聊略长些，但也限制长度防止过度输出
CLARIFY_MAX_TOKENS: int = 600


@step_log("generate_clarify_answer")
def generate_clarify_answer(state: dict) -> dict:
    """
    澄清回答主入口：读取上一条助手回答 → 渲染澄清 Prompt → 流式/非流式生成 → 落历史。

    Returns:
        dict: 更新 answer/image_urls 后的最新状态。
    """
    question = state.get("original_query") or state.get("rewritten_query") or ""
    if not question:
        raise ValueError("生成澄清回答需要 original_query/rewritten_query")
    session_id = state["session_id"]

    # 读取最近历史，取时间上最近的一条助手回答作为被质疑对象
    # list_recent 返回时间正序（旧→新），因此从末尾往回找最新的一条助手消息
    history_messages = history_repository.list_recent(session_id, limit=CLARIFY_HISTORY_LIMIT)
    last_answer = next(
        (m.get("text") or "" for m in reversed(history_messages) if m.get("role") == "assistant"),
        "",
    ).strip()
    if not last_answer:
        last_answer = "（当前会话还没有助手回答，用户似乎发来了困惑输入）"
        logger.info(f"澄清节点未找到上一条助手回答，session_id={session_id}")

    history_text = build_history_text(history_messages) or "（无）"
    prompt = load_prompt(
        "common/clarify_answer",
        question=question,
        last_answer=last_answer,
        current_date=state.get("current_date") or "未知",
        history=history_text,
    )

    # 生成回答（流式逐字推送 / 非流式一次性调用）
    is_stream = state.get("is_stream", False)
    lm_client = llm_provider.chat().bind(max_tokens=CLARIFY_MAX_TOKENS)
    final_result = ""
    if is_stream:
        blank_comp = BlankLineCompressor()
        for chunk in lm_client.stream(prompt):
            if is_cancelled(session_id):
                logger.info(f"用户停止生成，中断澄清回答输出，session_id={session_id}")
                break
            delta_content = blank_comp.feed(chunk.content or "")
            final_result += delta_content
            if delta_content:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        response = lm_client.invoke(prompt)
        final_result = response.content

    final_result = normalize_answer_text(final_result)
    set_task_result(session_id, "answer", final_result)
    state["answer"] = final_result
    state["image_urls"] = []
    history_repository.save_message(
        session_id=session_id,
        role="assistant",
        text=final_result,
        rewritten_query=question,
        item_names=[],
        image_urls=[],
        domain=state.get("domain", "chitchat"),
    )
    return state
