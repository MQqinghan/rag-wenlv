"""
文旅主体确认服务模块：负责问题改写、主体（景点名/文化主题）抽取与主体库确认。
主体库为 tourism_entity；无确认主体时【降级为全库检索】而非返回澄清话术，
以支持"北欧文化有什么特点"这类无明确景点归属的纯文化问题。
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

import re

from app.infra.llm import llm_provider
from app.infra.persistence import history_repository
from app.infra.vectorstore import milvus_gateway
from app.rag.common.history_text_utils import (
    HISTORY_DEIXIS_MAX_CHARS,
    build_history_text,
)
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.logger import logger, step_log

QUERY_HISTORY_LIMIT = 10
ENTITY_CONFIRM_THRESHOLD = 0.65
ENTITY_CANDIDATE_THRESHOLD = 0.50
ENTITY_OPTIONS_TOPK = 2
# 需要靠上一轮助手长回答消解的指代表达：命中时才放宽历史截断，
# 其余场景一律沿用 160 字上限，守住"历史主体污染改写"这条防线。
_DEIXIS_NEED_HISTORY_PATTERN = re.compile(
    r"那里|那儿|那个地方|这个地方|那座|书里|书中|这本书|这本书里|提到的地方|说的那个|前文|上面说"
)


@step_log("validate_query_identity")
def validate_query_identity(state: dict) -> tuple[str, str]:
    """校验查询状态中是否包含主体确认所需的核心字段。"""
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if not original_query or not session_id:
        logger.error("session_id和original_query不能为空")
        raise ValueError("session_id和original_query不能为空")
    return original_query, session_id


@step_log("load_history")
def load_history(session_id: str) -> list[dict]:
    """读取当前会话最近的历史消息。"""
    return history_repository.list_recent(session_id, limit=QUERY_HISTORY_LIMIT)


@step_log("rewrite_query_and_extract_attractions")
def rewrite_query_and_extract_attractions(history_messages: list[dict], original_query: str) -> dict:
    """
    一次模型调用同时完成问题改写与主体抽取（主体 = 景点名/文化主题）。

    Returns:
        dict: 至少包含 `rewritten_query` 与 `attractions` 两个字段。
    """
    # 指代消解放宽：问句里出现"书里提到的地方"这类表述时，主体只存在于上一轮助手的长回答中，
    # 160 字截断会把中后段地名切掉，改写层补不出主体 → 后续检索与规划全塌。
    if _DEIXIS_NEED_HISTORY_PATTERN.search(original_query):
        history_text = build_history_text(
            history_messages, assistant_max_chars=HISTORY_DEIXIS_MAX_CHARS
        )
    else:
        history_text = build_history_text(history_messages)

    def _call_model() -> dict:
        client = llm_provider.chat(json_mode=True)
        prompt = load_prompt(
            "tourism/rewritten_query_and_attractions",
            history_text=history_text,
            query=original_query,
        )
        messages = [
            SystemMessage(content="你是一个文旅知识库助手，擅长理解用户意图和提取景点/文化主题。"),
            HumanMessage(content=prompt),
        ]
        return (client | JsonOutputParser()).invoke(messages)

    # 改写+主体抽取结果在相同(问题, 历史)下高度稳定，命中缓存省掉一次模型往返（约 0.8~1.5s）
    result = cached_invoke(
        namespace="tourism_query_rewrite",
        cache_parts=(original_query, history_text),
        producer=_call_model,
        cache_label=original_query[:20],
        semantic=True,
    )
    if "rewritten_query" not in result:
        logger.warning(f"模型重写问题失败，给 rewritten_query 赋予原始问题：{original_query}")
        result["rewritten_query"] = original_query
    if "attractions" not in result:
        logger.warning("模型识别主体失败，给 attractions 赋予空列表")
        result["attractions"] = []
    return result


@step_log("search_entity_candidates")
def search_entity_candidates(attractions: list[str]) -> dict[str, list[dict]]:
    """
    基于候选主体名称到 tourism_entity 主体库检索相近主体。

    Returns:
        dict[str, list[dict]]: 以原始主体名为键、相似主体候选列表为值的映射。
    """
    vector_dict: dict[str, list[dict]] = {}
    if not attractions:
        return vector_dict
    try:
        item_vectors = llm_provider.embed_documents(attractions)
    except Exception as e:
        logger.warning(f"主体向量化失败，跳过主体确认：{e}")
        return vector_dict

    for index, attraction in enumerate(attractions):
        dense_vector = item_vectors["dense"][index]
        sparse_vector = item_vectors["sparse"][index]
        reqs = milvus_gateway.create_requests(dense_vector, sparse_vector)
        try:
            response = milvus_gateway.hybrid_search(
                collection_name=milvus_gateway.tourism_entity_collection,
                reqs=reqs,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"],
            )
        except Exception as e:
            logger.warning(f"主体检索失败（集合可能未创建）：{e}")
            vector_dict[attraction] = []
            continue

        current_list: list[dict] = []
        for item in (response[0] if response else []):
            current_list.append(
                {
                    "item_name": item.get("entity", {}).get("item_name", ""),
                    "score": item.get("distance", 0),
                }
            )
        vector_dict[attraction] = current_list
    return vector_dict


@step_log("select_confirmed_entities")
def select_confirmed_entities(vector_dict: dict[str, list[dict]]) -> list[str]:
    """
    从主体候选中选出高置信确认项（>=阈值）。
    候选/低置信主体不生成澄清话术，直接留空走全库检索兜底。
    """
    confirmed: list[str] = []
    for _, item_list in vector_dict.items():
        item_list.sort(key=lambda x: x["score"], reverse=True)
        high_list = [item for item in item_list if item["score"] >= ENTITY_CONFIRM_THRESHOLD]
        if high_list:
            confirmed.append(high_list[0]["item_name"])
    return confirmed


@step_log("apply_attraction_result")
def apply_attraction_result(state: dict, confirmed: list[str], rewritten_query: str) -> None:
    """
    将主体确认结果写回查询状态。
    有确认主体 → 带 item_names 走定向检索；无确认主体 → item_names 为空，后续全库检索。
    """
    state["item_names"] = confirmed
    state["rewritten_query"] = rewritten_query
    if "answer" in state:
        del state["answer"]


@step_log("save_user_message")
def save_user_message(state: dict) -> None:
    """将用户消息及其改写结果写入历史记录。

    is_plan 一并落库：下一轮延续类追问（"我想坐飞机去""我要去杭州，不是成都"）
    靠它继承规划分支，避免退化成普通检索问答。
    """
    history_repository.save_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        domain=state.get("domain", ""),
        is_plan=bool(state.get("is_plan", False)),
    )


@step_log("confirm_attractions")
def confirm_attractions(state: dict) -> dict:
    """
    文旅主体确认主流程：改写问题 + 抽取主体 + 主体库确认 + 写回状态。

    Returns:
        dict: 写回主体确认结果后的最新状态。
    """
    original_query, session_id = validate_query_identity(state)
    history_messages = load_history(session_id)
    # 回填 state["history"]：答案生成（answer_output_service）与行程拼装（itinerary_service）
    # 都从该字段取多轮上下文，此前无人填充导致生成阶段恒为空、延续类追问只能靠模型编造。
    # 此处读取发生在 save_user_message 之前，天然不含当前轮，正是"历史"。
    state["history"] = history_messages
    llm_result = rewrite_query_and_extract_attractions(history_messages, original_query)
    attractions = llm_result.get("attractions", [])
    rewritten_query = llm_result.get("rewritten_query", original_query)

    confirmed: list[str] = []
    if attractions:
        confirmed = select_confirmed_entities(search_entity_candidates(attractions))
        logger.info(
            f"文旅主体确认：抽取[{attractions}]，确认[{confirmed}]（未确认主体将降级为全库检索）"
        )

    apply_attraction_result(state, confirmed, rewritten_query)
    save_user_message(state)
    return state
