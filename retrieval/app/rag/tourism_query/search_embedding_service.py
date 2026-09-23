"""
文旅向量检索服务模块，负责对 tourism_chunks 集合执行混合向量检索。
与公共层差异：支持【无主体全库检索】——item_names 为空时不带过滤条件，
从而覆盖"北欧文化有什么特点"这类无景点归属的纯文化问题。
B2 治本（2026-09-11 主人拍板方案二）：item_names 为空但问句含明确城市时，
按 region 过滤（RETRIEVAL_REGION_FILTER 开关，默认 OFF），治理跨城文档混入。
"""
from app.infra.llm import llm_provider
from app.infra.vectorstore import milvus_gateway
from app.shared.config.common import env_bool
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.city_utils import build_region_like_expr, extract_cities

RETRIEVAL_DEFAULT_LIMIT = 5
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)


@step_log("resolve_filter_regions")
def resolve_filter_regions(state: dict) -> list[str]:
    """从问句解析城市过滤关键词（B2 城市过滤，开关 RETRIEVAL_REGION_FILTER 默认 OFF）。

    规则：
    - 开关关闭 / 已确认主体（item_names 非空，走定向检索）/ 问句无城市 → 返回空（不过滤）；
    - 命中城市时返回 [城市, 所属省, ...] 关键词列表（供 region like 双匹配）。
    """
    if not env_bool("RETRIEVAL_REGION_FILTER", default=False):
        return []
    if state.get("item_names"):
        return []
    question = state.get("original_query") or state.get("rewritten_query") or ""
    cities = extract_cities(question)
    if not cities:
        return []
    keywords: list[str] = []
    from app.shared.utils.city_utils import CITY_TO_PROVINCE
    for city in cities:
        if city not in keywords:
            keywords.append(city)
        province = CITY_TO_PROVINCE.get(city)
        if province and province not in keywords:
            keywords.append(province)
    logger.info(f"B2 城市过滤生效：问句命中城市={cities}，region 关键词={keywords}")
    return keywords


@step_log("search_embedding")
def search_embedding(state: dict) -> list[dict]:
    """
    文旅普通向量检索服务总入口。

    Args:
        state: 查询图当前状态，需包含 rewritten_query（可含 item_names）。

    Returns:
        list[dict]: 检索得到的文旅切块列表。
    """
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    if not rewritten_query:
        logger.error("rewritten_query不存在,无法继续业务!")
        raise ValueError("rewritten_query不存在,无法继续业务!")
    item_names = state.get("item_names") or []
    return search_tourism_chunks(
        rewritten_query=rewritten_query,
        item_names=item_names,
        filter_regions=resolve_filter_regions(state),
    )


@step_log("build_entity_expr")
def build_entity_expr(item_names: list[str]) -> str:
    """构建 Milvus 过滤表达式，限定按主体（景点/文化主题）过滤。"""
    return f"item_name in {item_names}"


@step_log("normalize_retrieved_chunk")
def normalize_retrieved_chunk(chunk: dict) -> dict:
    """
    将 Milvus 检索结果归一化为查询链内部统一使用的文档结构，
    携带文旅专属字段供答案生成/来源标注使用。
    """
    entity = chunk.get("entity", chunk)
    return {
        "chunk_id": chunk.get("id") or entity.get("chunk_id"),
        "item_name": entity.get("item_name", ""),
        "title": entity.get("title"),
        "parent_title": entity.get("parent_title"),
        "part": entity.get("part"),
        "file_title": entity.get("file_title"),
        "content_type": entity.get("content_type", ""),
        "region": entity.get("region", ""),
        "cultural_theme": entity.get("cultural_theme", ""),
        "category": entity.get("category", ""),
        "source_type": entity.get("source_type", ""),
        "extra_meta": entity.get("extra_meta") or {},
        "source_path": entity.get("source_path", ""),
        "content": entity.get("content", ""),
        "score": chunk.get("distance", 0.0),
        "type": "milvus",
        "url": None,
    }


@step_log("search_tourism_chunks_by_vector")
def search_tourism_chunks_by_vector(
    *,
    dense_vector: list[float],
    sparse_vector: dict,
    item_names: list[str],
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
    filter_regions: list[str] | None = None,
) -> list[dict]:
    """
    基于已编码的向量执行一次混合向量检索。

    与 search_tourism_chunks 的差异：跳过编码步骤，供多改写检索复用——
    多路查询可被合并成一次批量编码后逐路调用本函数，避免逐路重复编码。

    Args:
        dense_vector: 稠密向量。
        sparse_vector: 稀疏向量。
        item_names: 已确认的主体名称列表；为空时执行全库检索（纯文化问题兜底）。
        limit: 最大返回文档数。
        filter_regions: B2 城市过滤关键词（如 ["三亚","海南"]）；None/空 = 不过滤。
            仅在 item_names 为空（全库检索）时生效，主体定向优先于城市过滤。

    Returns:
        list[dict]: 检索得到的切块结果列表。
    """
    expr = build_entity_expr(item_names) if item_names else None
    if expr:
        logger.info(f"文旅定向检索，过滤主体：{item_names}")
    elif filter_regions:
        expr = build_region_like_expr(filter_regions)
        logger.info(f"文旅城市过滤检索（B2 治理跨城污染）：{expr}")
    else:
        logger.info("文旅全库检索（无确认主体，纯文化问题兜底）")

    reqs = milvus_gateway.create_requests(
        dense_vector,
        sparse_vector,
        expr=expr,
        limit=limit,
    )
    resp = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.tourism_chunks_collection,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        norm_score=True,
        limit=limit,
        output_fields=[
            "chunk_id", "item_name", "content", "title", "parent_title", "part",
            "file_title", "content_type", "region", "cultural_theme", "category", "source_type",
            "extra_meta", "source_path",
        ],
    )
    return [normalize_retrieved_chunk(chunk) for chunk in (resp[0] if resp else [])]


@step_log("search_tourism_chunks")
def search_tourism_chunks(
    *,
    rewritten_query: str,
    item_names: list[str],
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
    filter_regions: list[str] | None = None,
) -> list[dict]:
    """
    基于改写问题对 tourism_chunks 执行一次混合向量检索（编码 + 检索一步到位）。

    Args:
        rewritten_query: 用于检索的改写后问题。
        item_names: 已确认的主体名称列表；为空时执行全库检索（纯文化问题兜底）。
        limit: 最大返回文档数。
        filter_regions: B2 城市过滤关键词（resolve_filter_regions 输出）；None/空 = 不过滤。

    Returns:
        list[dict]: 检索得到的切块结果列表。
    """
    embedding_result = llm_provider.embed_documents([rewritten_query])
    return search_tourism_chunks_by_vector(
        dense_vector=embedding_result["dense"][0],
        sparse_vector=embedding_result["sparse"][0],
        item_names=item_names,
        limit=limit,
        filter_regions=filter_regions,
    )
