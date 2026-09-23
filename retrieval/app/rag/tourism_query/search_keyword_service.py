"""
文旅关键词检索服务模块：基于本地 BM25 索引对 tourism_chunks 做字面匹配召回。

与向量检索的分工：
- 向量检索（BGE-M3 稠密+稀疏）：语义泛化强，擅长"意思相近但用词不同"的查询
- 关键词检索（BM25）：字面精确匹配强，擅长景点名、政策术语、专有名词的精确命中
两路结果后续由 RRF 融合，互相补位。

主体过滤与向量检索保持一致：有确认主体时限定 item_name，无主体时全库匹配。
"""
from app.infra.vectorstore import milvus_gateway
from app.rag.common.bm25_service import search_by_bm25
from app.rag.tourism_query.search_embedding_service import (
    normalize_retrieved_chunk,
    resolve_filter_regions,
)
from app.shared.runtime.logger import logger, step_log

# 索引需保留的字段：与向量检索路的 output_fields 对齐，保证归一化结果结构一致
BM25_OUTPUT_FIELDS = [
    "chunk_id", "item_name", "content", "title", "parent_title", "part",
    "file_title", "content_type", "region", "cultural_theme", "category", "source_type",
    "extra_meta", "source_path",
]


@step_log("search_keyword")
def search_keyword(state: dict) -> list[dict]:
    """
    文旅关键词检索服务总入口。

    Args:
        state: 查询图当前状态，需包含 rewritten_query（可含 item_names）。

    Returns:
        list[dict]: 关键词检索得到的文旅切块列表；索引不可用时为空列表。
    """
    query = state.get("rewritten_query") or state.get("original_query")
    if not query:
        logger.error("rewritten_query不存在,无法执行关键词检索!")
        raise ValueError("rewritten_query不存在,无法执行关键词检索!")
    item_names = state.get("item_names") or []
    if item_names:
        logger.info(f"文旅关键词定向检索，过滤主体：{item_names}")
    else:
        logger.info("文旅关键词全库检索（无确认主体）")

    results = search_by_bm25(
        query=query,
        collection_name=milvus_gateway.tourism_chunks_collection,
        output_fields=BM25_OUTPUT_FIELDS,
        normalizer=normalize_retrieved_chunk,
        filter_field="item_name",
        filter_names=item_names,
    )

    # B2 城市过滤（BM25 本地索引不支持 like 表达式，在此做结果级后过滤）：
    # 与向量路同口径——仅全库检索（无确认主体）且问句命中城市时生效。
    filter_regions = resolve_filter_regions(state)
    if filter_regions and results:
        before = len(results)
        results = [
            chunk for chunk in results
            if any(kw in (chunk.get("region") or "") for kw in filter_regions)
        ]
        logger.info(f"B2 城市过滤（BM25 后过滤）：{before} -> {len(results)} 条，关键词={filter_regions}")

    return results
