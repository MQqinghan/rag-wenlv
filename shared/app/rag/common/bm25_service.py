"""
本地 BM25 关键词检索服务模块。

定位与价值：
BGE-M3 的稀疏向量是"学习型稀疏"（类 SPLADE），擅长语义泛化，但对【专有名词的字面命中】
反而弱于经典 BM25——景点名、人名、章节名、门票政策术语、型号编号这类查询，
BM25 的 IDF 加权字面匹配能召回向量检索漏掉的精确片段，两者互补性很强。

实现选择：进程内倒排索引，不动 Milvus schema、不需要重建集合与重导数据。
- 索引数据通过 Milvus 标量 query 分页拉取（不加载向量，代价极低）
- 索引在进程内缓存并按 TTL 自动刷新；导入完成后可显式失效
- 单次检索为纯内存计算，当前数据量下耗时在毫秒级

规模边界：BM25_MAX_DOCS 为保护阈值，超过后自动降级（跳过关键词路，只走向量），
避免超大库把文本全量驻留内存。若未来数据量持续增长，应切换到 Milvus 原生全文检索。
"""
import re
import threading
import time
from typing import Callable

import jieba
from rank_bm25 import BM25Okapi

from app.infra.vectorstore import milvus_gateway
from app.shared.config.common import env_bool, env_int
from app.shared.runtime.logger import logger, step_log

# ====================== BM25 全局配置 ======================
BM25_ENABLE: bool = env_bool("BM25_ENABLE", default=True)          # 总开关，关闭后关键词路直接返回空
BM25_TOP_N: int = env_int("BM25_TOP_N", default=5)                  # 关键词路返回条数（进 RRF 前）
BM25_MAX_DOCS: int = env_int("BM25_MAX_DOCS", default=50000)        # 单集合参与索引的文档上限（保护内存）
BM25_TTL_SECONDS: int = env_int("BM25_INDEX_TTL_SECONDS", default=600)  # 索引缓存有效期
BM25_QUERY_PAGE_SIZE: int = 1000                                    # 从 Milvus 拉取文档的分页大小
BM25_MIN_SCORE: float = 0.0                                         # 得分下限（0 表示一个查询词都没命中）

# 检索侧停用词：虚词/助词/标点不参与 BM25 计分，避免长文本靠语气词刷分
_BM25_STOPWORDS = frozenset({
    "的", "了", "是", "在", "和", "与", "及", "或", "有", "为", "对", "把", "被", "给", "让",
    "这", "那", "之", "其", "此", "该", "个", "们", "我", "你", "他", "她", "它", "咱",
    "什么", "怎么", "如何", "哪些", "请问", "可以", "能够", "会", "要", "就", "也", "都", "还",
    "因为", "所以", "但是", "而且", "如果", "虽然", "然后", "并且", "不是", "没有",
    "一下", "这个", "那个", "这样", "那样", "一种", "方面", "情况", "内容", "相关",
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
})

# 纯标点/空白字符：分词后直接丢弃
_PUNCT_PATTERN = re.compile(r"^[\s\W_]+$", re.UNICODE)

_index_lock = threading.Lock()
# 结构：collection_name -> BM25Index
_index_cache: dict[str, "BM25Index"] = {}


def bm25_tokenize(text: str) -> list[str]:
    """
    对文本做检索用分词：jieba 切词后过滤停用词、单字与纯标点。

    注意：本函数被索引构建热循环逐文档调用（一次构建可达数百次），
    因此不挂 @step_log 装饰器，避免其 inspect.stack() 定位开销拖慢建索引。

    Args:
        text: 待分词文本。

    Returns:
        list[str]: 归一化后的词项列表。
    """
    if not text:
        return []
    try:
        tokens = jieba.lcut(text)
    except Exception:
        tokens = []
    result: list[str] = []
    for token in tokens:
        token = token.strip().lower()
        if len(token) < 2:
            continue
        if token in _BM25_STOPWORDS:
            continue
        if _PUNCT_PATTERN.match(token):
            continue
        result.append(token)
    return result


class BM25Index:
    """单个集合的进程内 BM25 索引：持有原始文档与分词结果，支持带过滤的 TopN 检索。"""

    def __init__(self, raw_docs: list[dict]):
        """
        Args:
            raw_docs: Milvus 风格的文档列表，元素形如 {"id":..., "distance":..., "entity": {...}}。
        """
        self.raw_docs: list[dict] = raw_docs
        self.tokenized_docs: list[list[str]] = [
            bm25_tokenize(doc.get("entity", {}).get("content", "")) for doc in raw_docs
        ]
        # 过滤掉空分词文档，避免 BM25Okapi 对全空文档除零
        self.bm25 = BM25Okapi(self.tokenized_docs) if self.tokenized_docs else None
        self.built_at: float = time.time()

    def is_expired(self) -> bool:
        """索引是否已过 TTL，过期需重建以感知新导入的数据。"""
        return (time.time() - self.built_at) > BM25_TTL_SECONDS

    def search(
        self,
        query: str,
        *,
        top_n: int,
        filter_field: str = "",
        filter_names: list[str] | None = None,
    ) -> list[dict]:
        """
        执行 BM25 检索并按得分倒序返回 TopN。

        Args:
            query: 检索词（通常是改写后的问题）。
            top_n: 返回条数上限。
            filter_field: 主体过滤字段名（如 item_name / book_name）；为空表示不过滤。
            filter_names: 允许通过的主体名列表；为空或 None 表示不限制。

        Returns:
            list[dict]: Milvus 风格的命中结果（含 id/distance/entity），可直接复用现有归一化函数。
        """
        if self.bm25 is None or not self.raw_docs:
            return []

        query_tokens = bm25_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        allowed = set(filter_names) if (filter_field and filter_names) else None

        candidates: list[tuple[float, int]] = []
        for index, score in enumerate(scores):
            if score <= BM25_MIN_SCORE:
                continue
            if allowed is not None:
                entity = self.raw_docs[index].get("entity", {})
                if entity.get(filter_field) not in allowed:
                    continue
            candidates.append((float(score), index))

        if not candidates:
            return []

        # 分数相同时用下标保证结果稳定（避免多线程下排序抖动）
        candidates.sort(key=lambda item: (-item[0], item[1]))
        hits: list[dict] = []
        for score, index in candidates[:top_n]:
            doc = dict(self.raw_docs[index])
            doc["distance"] = score
            hits.append(doc)
        return hits


@step_log("fetch_chunks_for_index")
def fetch_chunks_for_index(collection_name: str, output_fields: list[str]) -> list[dict]:
    """
    从 Milvus 分页拉取全量切块（只取标量字段，不加载向量），包装成 Milvus 检索结果风格。

    Args:
        collection_name: 目标集合名。
        output_fields: 需要拉取的字段列表。

    Returns:
        list[dict]: 形如 {"id":..., "distance":0.0, "entity": {...}} 的文档列表。
    """
    client = milvus_gateway.client()
    if client is None:
        logger.warning(f"BM25 索引构建跳过：Milvus 客户端为空，集合[{collection_name}]")
        return []
    if not client.has_collection(collection_name=collection_name):
        logger.warning(f"BM25 索引构建跳过：集合[{collection_name}]不存在")
        return []

    fields = list(dict.fromkeys([*output_fields, "chunk_id"]))
    raw_docs: list[dict] = []
    offset = 0
    while offset < BM25_MAX_DOCS:
        batch = client.query(
            collection_name=collection_name,
            filter="chunk_id >= 0",
            output_fields=fields,
            limit=BM25_QUERY_PAGE_SIZE,
            offset=offset,
        )
        if not batch:
            break
        for row in batch:
            # 与 hybrid_search 返回结构对齐，便于下游直接复用 normalize_retrieved_chunk
            raw_docs.append({
                "id": row.get("chunk_id"),
                "distance": 0.0,
                "entity": row,
            })
        if len(batch) < BM25_QUERY_PAGE_SIZE:
            break
        offset += len(batch)

    logger.info(f"BM25 索引数据拉取完成，集合[{collection_name}]，共 {len(raw_docs)} 条")
    return raw_docs


@step_log("get_bm25_index")
def get_bm25_index(collection_name: str, output_fields: list[str]) -> BM25Index | None:
    """
    获取（或按需构建）指定集合的 BM25 索引，带进程内缓存与 TTL 刷新。

    Args:
        collection_name: 目标集合名。
        output_fields: 索引构建时需要保留的字段。

    Returns:
        BM25Index | None: 索引实例；集合不存在、无数据或超规模时返回 None。
    """
    if not BM25_ENABLE:
        return None

    with _index_lock:
        cached = _index_cache.get(collection_name)
        if cached is not None and not cached.is_expired():
            return cached

    # 构建索引是重 IO 操作，放在锁外执行，避免阻塞其他集合的并发查询
    raw_docs = fetch_chunks_for_index(collection_name, output_fields)
    if not raw_docs:
        return None
    if len(raw_docs) > BM25_MAX_DOCS:
        logger.warning(
            f"集合[{collection_name}]文档数 {len(raw_docs)} 超过 BM25_MAX_DOCS={BM25_MAX_DOCS}，"
            f"跳过关键词路（建议改用 Milvus 原生全文检索）"
        )
        return None

    start_ts = time.time()
    index = BM25Index(raw_docs)
    cost_ms = int((time.time() - start_ts) * 1000)
    logger.info(f"BM25 索引构建完成，集合[{collection_name}]，{len(raw_docs)} 条，耗时={cost_ms}ms")

    with _index_lock:
        _index_cache[collection_name] = index
    return index


def invalidate_bm25_cache(collection_name: str | None = None) -> None:
    """
    失效 BM25 索引缓存。知识库导入完成后应调用，避免新数据被 TTL 内的旧索引挡住。

    Args:
        collection_name: 指定集合；为空则清空全部。
    """
    with _index_lock:
        if collection_name:
            _index_cache.pop(collection_name, None)
        else:
            _index_cache.clear()
    logger.info(f"BM25 索引缓存已失效：collection={collection_name or 'ALL'}")


@step_log("search_by_bm25")
def search_by_bm25(
    *,
    query: str,
    collection_name: str,
    output_fields: list[str],
    normalizer: Callable[[dict], dict],
    top_n: int = BM25_TOP_N,
    filter_field: str = "",
    filter_names: list[str] | None = None,
) -> list[dict]:
    """
    关键词检索总入口：取索引 → BM25 打分 → 归一化成查询链统一文档结构。

    Args:
        query: 检索词。
        collection_name: 目标集合名。
        output_fields: 索引需要的字段。
        normalizer: 归一化函数，复用各域现有的 normalize_retrieved_chunk。
        top_n: 返回条数上限。
        filter_field: 主体过滤字段名。
        filter_names: 允许通过的主体名列表。

    Returns:
        list[dict]: 归一化后的文档列表；索引不可用时返回空列表（不影响主链路）。
    """
    if not BM25_ENABLE or not query:
        return []
    try:
        index = get_bm25_index(collection_name, output_fields)
        if index is None:
            return []
        hits = index.search(query, top_n=top_n, filter_field=filter_field, filter_names=filter_names)
        docs = [normalizer(hit) for hit in hits]
        logger.info(f"BM25 关键词检索命中 {len(docs)} 条，集合[{collection_name}]")
        return docs
    except Exception as e:
        # 关键词路是增强项，任何异常都只降级不影响向量主链路
        logger.warning(f"BM25 关键词检索失败，已降级为空结果：{e}")
        return []
