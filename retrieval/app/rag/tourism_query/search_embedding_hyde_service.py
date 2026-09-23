"""
文旅多改写检索服务模块（由原 HyDE 单路升级而来）。

一次 LLM 调用同时产出「2 条视角改写 + 1 段 HyDE 假设答案」，
合并成一次批量编码后并发检索 3 路，去重合并后返回。
相比原 HyDE 单路，LLM 调用次数不变，召回视角从 1 个扩到 3 个。

跳过条件沿用原策略：短查询（<15字）或已确认主体时不走扩展，
因为此时 embedding 直接检索已足够，扩展反而引入噪声并浪费约 2s。
"""
from app.rag.common.multi_query_service import (
    MULTI_QUERY_ENABLE,
    build_multi_queries,
    generate_query_variants,
    retrieve_multi_queries,
)
from app.rag.tourism_query.search_embedding_service import (
    RETRIEVAL_DEFAULT_LIMIT,
    resolve_filter_regions,
    search_tourism_chunks_by_vector,
)
from app.shared.runtime.logger import logger, step_log

# 多改写提示词与系统角色
_MULTI_QUERY_PROMPT = "tourism/multi_query"
_MULTI_QUERY_SYSTEM = "你是文旅知识库的检索查询扩展专家，只能输出合法 JSON。"


@step_log("search_embedding_hyde")
def search_embedding_hyde(state: dict) -> list[dict]:
    """
    文旅多改写检索服务总入口。
    对短查询（<15字）或已确认主体的查询跳过，返回空列表，RRF 仍可用其他路结果融合。

    Args:
        state: 查询图当前状态，需包含 rewritten_query（可含 item_names）。

    Returns:
        list[dict]: 多改写检索得到的文旅切块列表（跳过时为空）。
    """
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    if not rewritten_query:
        raise ValueError("rewritten_query不存在,无法继续业务!")
    item_names = state.get("item_names") or []
    # B2 城市过滤（与向量路同口径）：item_names 为空且问句含城市时按 region 过滤
    filter_regions = resolve_filter_regions(state)
    # 跳过条件：短查询（embedding 已足够）或主体已确认（无需假设扩展）
    if len(rewritten_query) < 15 or item_names:
        return []
    return search_chunks_with_multi_query(
        rewritten_query=rewritten_query, item_names=item_names, filter_regions=filter_regions
    )


@step_log("search_chunks_with_multi_query")
def search_chunks_with_multi_query(
    *,
    rewritten_query: str,
    item_names: list[str],
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
    history_text: str = "",
    filter_regions: list[str] | None = None,
) -> list[dict]:
    """
    生成多路查询扩展，批量编码后并发检索 tourism_chunks，去重合并返回。

    Args:
        rewritten_query: 用于检索的改写后问题。
        item_names: 已确认的主体名称列表（用于过滤）。
        limit: 单路返回条数上限。
        history_text: 历史会话文本，供改写参考。
        filter_regions: B2 城市过滤关键词；None/空 = 不过滤。

    Returns:
        list[dict]: 多路去重合并后的切块列表。
    """
    variants = generate_query_variants(
        prompt_name=_MULTI_QUERY_PROMPT,
        rewritten_query=rewritten_query,
        history_text=history_text,
        system_message=_MULTI_QUERY_SYSTEM,
    )
    queries = build_multi_queries(
        rewritten_query=rewritten_query, variants=variants, enable_multi=MULTI_QUERY_ENABLE
    )
    if not queries:
        logger.warning("多改写未生成有效查询，本次扩展路返回空结果")
        return []

    return retrieve_multi_queries(
        queries=queries,
        filter_names=item_names,
        search_one=lambda dense, sparse, names, top: search_tourism_chunks_by_vector(
            dense_vector=dense,
            sparse_vector=sparse,
            item_names=names,
            limit=top,
            filter_regions=filter_regions,
        ),
        limit=limit,
    )
