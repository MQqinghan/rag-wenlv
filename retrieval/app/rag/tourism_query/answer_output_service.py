"""
文旅答案输出服务模块，负责全链路最终的答案生成、图片提取与历史落库。
使用 tourism/answer_out 模板；item_names 语义为景点名/文化主题。
"""
import json
import re
import time

from app.infra.llm import llm_provider
from app.infra.llm_harness import answer_llm_client, reliable_invoke
from app.rag.common.memory_harness import maybe_compress_history, memory_block
from app.infra.persistence.history_repository import history_repository
from app.rag.common.history_text_utils import build_history_text
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.sse_utils import SSEEvent, is_cancelled, push_to_session
from app.shared.utils.task_utils import set_task_result
from app.shared.utils.text_utils import BlankLineCompressor, normalize_answer_text

# 引用标记 regex（与 text_utils 共用逻辑）：流式 delta 级别剥离【第N块】和【N】
_CITATION_PATTERN = re.compile(r"(?:\s*【(?:第\d+块|\d+)】)+")

# 代办/预订类请求（cl-003）：系统无任何预订交易能力，答案必须先声明能力边界再给参考建议。
# 代码级前置注入（prompt 规则遵循度不足的既有教训），避免模型直接以"能订"口吻给建议。
# 动词只收强交易向的"订/预订/预定/购买/买下"，不收"推荐/安排/看看"等泛词防误伤。
_BOOKING_INTENT_PATTERN = re.compile(r"(帮|替|给)我?[^。？！?!]{0,15}(订|预订|预定|购买|买下)|代订|代办")
_BOOKING_DISCLAIMER = (
    "先说明：我无法代为预订酒店、机票或门票，以下信息仅供您参考，实际预订请通过官方或正规平台办理。\n\n"
)


@step_log("generate_answer")
def generate_answer(state: dict) -> dict:
    """
    文旅答案输出节点主入口：
    1. 尝试复用已有答案
    2. 校验生成参数
    3. 构建 Prompt（tourism/answer_out）
    4. 调用模型生成答案
    5. 提取图片
    6. 保存历史记录
    """
    if not try_return_existing_answer(state):
        reranked_docs, item_names, rewritten_query, history = validate_generation_inputs(state)
        # 规划类问题才有天气/路线简报；ok=False 或缺失时传空串占位
        weather = state.get("tool_weather") or {}
        weather_text = weather.get("text", "") if weather.get("ok") else ""
        route = state.get("tool_route") or {}
        route_text = route.get("text", "") if route.get("ok") else ""
        extra_note = ""
        prompt = build_answer_prompt(
            reranked_docs, rewritten_query, item_names, history,
            current_date=state.get("current_date", ""),
            tool_weather=weather_text,
            tool_route=route_text,
            memory_text=memory_block(state),  # F：长期记忆注入（开关默认 OFF）
            extra_subject_note=extra_note,
        )
        final_answer(state, prompt)
        state["image_urls"] = extract_image_urls(reranked_docs)

    save_assistant_message(state)
    return state


@step_log("try_return_existing_answer")
def try_return_existing_answer(state: dict) -> bool:
    """复用已有答案：流式逐字推送；非流式直接写任务结果。"""
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    if not answer:
        return False
    if is_stream:
        for ch in answer:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.1)
    set_task_result(session_id, "answer", answer)
    return True


@step_log("validate_generation_inputs")
def validate_generation_inputs(state: dict) -> tuple[list[dict], list[str], str, list[dict]]:
    """答案生成前的必填参数校验，reranked_docs 为空时生成兜底回答。"""
    history = maybe_compress_history(state, state.get("history", []))  # F：长对话压缩（开关默认 OFF）
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    if not rewritten_query:
        raise ValueError("生成答案需要 rewritten_query/original_query")
    if not reranked_docs:
        logger.warning("reranked_docs 为空，生成兜底回答")
        reranked_docs = [{"title": "无", "score": 0, "type": "milvus", "text": "未找到相关资料"}]
    return reranked_docs, item_names, rewritten_query, history


@step_log("build_answer_prompt")
def build_answer_prompt(
    reranked_docs: list[dict],
    rewritten_query: str,
    item_names: list[str],
    history: list[dict],
    current_date: str = "",
    tool_weather: str = "",
    tool_route: str = "",
    memory_text: str = "",
    extra_subject_note: str = "",
) -> str:
    """构建文旅答案生成的 Prompt，整合参考内容、来源、历史、主体、当前日期、实时天气与路线。"""
    # 限制送入 LLM 的上下文长度：取 rerank 前 5 块（提速，reank 已排序，靠后的块相关性低）
    top_docs = reranked_docs[:5]
    context_chunk_list = []
    for number, chunk in enumerate(top_docs, start=1):
        # 来源归属：网页/本地文旅 两类区分，供答案"分块标注来源"引用
        _t = chunk.get("type")
        if _t == "web":
            source = "网络搜索"
        else:
            source = "向量查询"
        # 截断每块文本到 800 字（防单块过长拖慢 LLM）
        text = (chunk.get("text") or "")[:800]
        # 内容类型（文旅字段，网页块为空）
        content_type = chunk.get("content_type") or ""
        # 类型专属字段（文旅 extra_meta，含星级/价格/招牌菜等结构化信息；转 JSON 供 LLM 精确引用）
        extra_meta = chunk.get("extra_meta") or {}
        extra_line = ""
        if isinstance(extra_meta, dict) and extra_meta:
            extra_line = f"\n专属字段:{json.dumps(extra_meta, ensure_ascii=False)}"
        context_chunk_list.append(
            f"第{number}块: 标题:{chunk.get('title')} 匹配度得分:{chunk.get('score')} 来源:{source}"
            + (f" 类型:{content_type}" if content_type else "")
            + f"\n内容:{text}"
            + extra_line
        )
    context_chunk_str = "\n\n".join(context_chunk_list)
    history_text = build_history_text(history)
    if memory_text:
        history_text = f"{history_text}\n\n{memory_text}"
    item_name_str = "本次关联主体:" + ",".join(item_names) if item_names else "未确认具体主体（依据检索内容作答）"
    if extra_subject_note:
        item_name_str = f"{item_name_str}\n{extra_subject_note}"

    return load_prompt(
        "tourism/answer_out",
        context=context_chunk_str,
        history=history_text,
        item_names=item_name_str,
        question=rewritten_query,
        current_date=current_date or "未知",
        tool_weather=tool_weather or "（本次无实时天气数据）",
        tool_route=tool_route or "（本次无路线规划数据）",
    )


@step_log("final_answer")
def final_answer(state: dict, prompt: str) -> str:
    """调用大模型生成最终答案，支持流式与普通两种模式。"""
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    # P3（cl-003）：代办/预订类请求先声明能力边界（流式时先推声明 delta 再出正文）
    query = state.get("original_query") or state.get("rewritten_query") or ""
    booking_prefix = _BOOKING_DISCLAIMER if _BOOKING_INTENT_PATTERN.search(query or "") else ""
    if booking_prefix:
        logger.info("代办/预订类请求命中，答案前置能力边界声明")
        if is_stream:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": booking_prefix})
    lm_client = answer_llm_client(state, 1500)  # 限制生成长度，防止过度输出拖慢响应（E：档位路由可选）
    final_result = ""
    if is_stream:
        blank_comp = BlankLineCompressor()
        for chunk in lm_client.stream(prompt):
            if is_cancelled(session_id):
                logger.info(f"用户停止生成，中断文旅回答输出，session_id={session_id}")
                break
            delta_content = chunk.content
            # per-delta 轻清理：剥离引用标记 + 压缩连续空行（模型常输出 \n\n\n\n 大片空行）
            delta_content = blank_comp.feed(_CITATION_PATTERN.sub("", delta_content))
            final_result += delta_content
            if delta_content:
                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        response = reliable_invoke(lm_client, prompt, state)
        final_result = response.content
    # 后处理：压缩空行 + 清理孤立图片行（引用标记已在 delta 级剥离，这里兜底）
    final_result = normalize_answer_text(final_result)
    # P3（cl-003）：能力边界声明拼在正文最前（出处分节之前）
    if booking_prefix:
        final_result = booking_prefix + final_result
    # 追加出处标注（代码级拼接不依赖 LLM）：知识库文件名 + 网页标题链接
    source_section = build_source_section(state.get("reranked_docs") or [])
    if source_section:
        final_result += source_section
    set_task_result(session_id, "answer", final_result)
    state["answer"] = final_result
    return final_result


@step_log("extract_image_urls")
def extract_image_urls(reranked_docs: list[dict]) -> list[str]:
    """从参考文档中提取所有图片 URL（直接 URL + markdown 图片）。"""
    image_urls: list[str] = []
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")
    for doc in reranked_docs:
        url = doc.get("url")
        text = doc.get("text")
        if url and url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")) and url not in image_urls:
            image_urls.append(url)
        if text:
            for image_url in reg.findall(text):
                if image_url not in image_urls:
                    image_urls.append(image_url)
    return image_urls


@step_log("build_source_section")
def build_source_section(reranked_docs: list[dict], top: int = 5) -> str:
    """
    从精排候选中收集出处，生成回答末尾的"参考资料"分节（代码级拼接不依赖 LLM）。
    文旅检索包含知识库与网页两类来源，两类分列并自动去重——
    - 知识库块（type != web）：取 file_title（来源文件名）
    - 网页块（type == web）：取 标题 + URL（有标题拼 markdown 链接，无标题退化为裸 URL）

    Args:
        reranked_docs: 精排后的候选文档列表。
        top: 最多取前 N 个候选的出处。

    Returns:
        str: 形如 "\\n\\n**参考资料**\\n- 上海文化.md（知识库）\\n- [海派文化](url)（网页）"；
             无可用出处时返回空串。
    """
    kb_sources: list[str] = []
    web_sources: list[str] = []
    seen_kb: set[str] = set()
    seen_web: set[str] = set()
    for chunk in (reranked_docs or [])[:top]:
        if chunk.get("type") == "web":
            url = (chunk.get("url") or "").strip()
            if not url or url in seen_web:
                continue
            seen_web.add(url)
            title = (chunk.get("title") or "").strip()
            web_sources.append(f"[{title}]({url})" if title else url)
        else:
            file_title = (chunk.get("file_title") or "").strip()
            if not file_title or file_title in seen_kb:
                continue
            seen_kb.add(file_title)
            # 本地块携带 source_path（上传文件本地路径）时附加展示，对齐需求"来源路径或资源链接"
            source_path = (chunk.get("source_path") or "").strip()
            kb_sources.append(f"{file_title}（路径：{source_path}）" if source_path else file_title)
    items = [f"- {name}（知识库）" for name in kb_sources]
    items += [f"- {ref}（网页）" for ref in web_sources]
    if not items:
        return ""
    return "\n\n**参考资料**\n" + "\n".join(items)


@step_log("save_assistant_message")
def save_assistant_message(state: dict) -> None:
    """将助手回答保存到历史记录。"""
    history_repository.save_message(
        session_id=state["session_id"],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
        domain=state.get("domain", ""),
        is_plan=bool(state.get("is_plan", False)),
    )
