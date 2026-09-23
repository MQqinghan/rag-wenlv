"""
多改写检索服务模块：一次 LLM 调用扩展出多路查询，一次批量编码，并发完成多路召回。

与旧 HyDE 单路的差异（同样一次 LLM 调用，但召回覆盖显著变宽）：
旧链路：1 次 LLM 生成假设答案 → 拼接成 1 条查询 → 1 次编码 → 1 次检索（1 路召回）
新链路：1 次 LLM 同时产出「2 条视角改写 + 1 段 HyDE 假设答案」→ 拼成 3 条查询
        → 合并成 1 次批量编码（GPU 一次算完，规避 BGE-M3 全局锁的串行问题）
        → 线程池并发检索 → 按 chunk_id 去重合并（3 路召回）

因此多改写路的【墙钟时间几乎不增加】，却把召回视角从 1 个扩到 3 个。
"""
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm import llm_provider
from app.shared.config.common import env_bool, env_int
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

# 多改写总开关：关闭后退回为"仅 HyDE 假设答案"单路检索
MULTI_QUERY_ENABLE: bool = env_bool("MULTI_QUERY_ENABLE", default=True)
# 最多使用的改写条数（防止 LLM 返回过多导致检索放大）
MULTI_QUERY_MAX_REWRITES: int = env_int("MULTI_QUERY_MAX_REWRITES", default=2)
# 并发检索线程数
MULTI_QUERY_WORKERS: int = env_int("MULTI_QUERY_WORKERS", default=4)
# 单条改写查询的最大长度（超长会稀释向量语义）
MULTI_QUERY_MAX_LENGTH: int = 100
# 假设答案参与检索时的截断长度
HYDE_ANSWER_MAX_LENGTH: int = 300

# 改写文本清洗：去掉序号前缀与首尾引号，避免"1. "之类噪声进入向量编码
_REWRITE_NOISE_PATTERN = re.compile(r"^\s*(?:\d+[.、)]|[-*•])\s*")


@step_log("clean_rewrite_text")
def clean_rewrite_text(text: str) -> str:
    """
    清洗单条改写查询：去序号前缀、去首尾引号、压空白、截断超长。

    Args:
        text: 模型产出的原始改写文本。

    Returns:
        str: 清洗后的查询文本；清洗后为空则返回空字符串。
    """
    if not text or not isinstance(text, str):
        return ""
    cleaned = _REWRITE_NOISE_PATTERN.sub("", text.strip())
    cleaned = cleaned.strip("\"'“”‘’《》 ").strip()
    if len(cleaned) > MULTI_QUERY_MAX_LENGTH:
        cleaned = cleaned[:MULTI_QUERY_MAX_LENGTH]
    return cleaned


@step_log("generate_query_variants")
def generate_query_variants(
    *,
    prompt_name: str,
    rewritten_query: str,
    history_text: str = "",
    system_message: str = "你是检索查询扩展专家，只能输出合法 JSON。",
) -> dict:
    """
    一次 LLM 调用同时产出多路改写查询与 HyDE 假设答案。

    Args:
        prompt_name: 提示词名（如 tourism/multi_query）。
        rewritten_query: 改写后的用户问题。
        history_text: 历史会话文本。
        system_message: 系统提示词。

    Returns:
        dict: {"rewrites": list[str], "hyde_answer": str}；解析失败时返回空结构。
    """
    client = llm_provider.chat(json_mode=True)
    prompt = load_prompt(prompt_name, history_text=history_text, query=rewritten_query)
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
    try:
        result = (client | JsonOutputParser()).invoke(messages)
    except Exception as e:
        logger.warning(f"多改写生成失败，降级为无改写：{e}")
        return {"rewrites": [], "hyde_answer": ""}

    raw_rewrites = result.get("rewrites") or []
    if isinstance(raw_rewrites, str):
        raw_rewrites = [raw_rewrites]
    rewrites: list[str] = []
    for item in raw_rewrites:
        cleaned = clean_rewrite_text(item)
        if cleaned and cleaned not in rewrites:
            rewrites.append(cleaned)
        if len(rewrites) >= MULTI_QUERY_MAX_REWRITES:
            break

    hyde_answer = (result.get("hyde_answer") or "").strip()
    if len(hyde_answer) > HYDE_ANSWER_MAX_LENGTH:
        hyde_answer = hyde_answer[:HYDE_ANSWER_MAX_LENGTH]

    logger.info(f"多改写生成完成：{len(rewrites)} 条改写，假设答案 {len(hyde_answer)} 字")
    return {"rewrites": rewrites, "hyde_answer": hyde_answer}


@step_log("build_multi_queries")
def build_multi_queries(*, rewritten_query: str, variants: dict, enable_multi: bool) -> list[str]:
    """
    把改写结果与假设答案拼成最终的多路查询列表（不含原始 query，原始 query 由独立节点负责）。

    Args:
        rewritten_query: 改写后的用户问题。
        variants: generate_query_variants 的返回结构。
        enable_multi: 是否启用多改写（关闭时只保留 HyDE 拼接查询）。

    Returns:
        list[str]: 待检索的查询列表，已去重。
    """
    queries: list[str] = []
    if enable_multi:
        queries.extend(variants.get("rewrites") or [])

    hyde_answer = variants.get("hyde_answer") or ""
    if hyde_answer:
        # HyDE 经典拼接：原问题 + 假设答案，让向量靠近"答案所在区域"
        queries.append(f"{rewritten_query},{hyde_answer}")

    # 去重且排除与原始 query 完全相同的项（避免与向量主路的重复召回）
    deduped: list[str] = []
    for query in queries:
        if query and query != rewritten_query and query not in deduped:
            deduped.append(query)
    return deduped


@step_log("retrieve_multi_queries")
def retrieve_multi_queries(
    *,
    queries: list[str],
    filter_names: list[str],
    search_one: Callable[[list[float], dict, list[str], int], list[dict]],
    limit: int,
) -> list[dict]:
    """
    批量编码 + 并发检索 + 去重合并。

    Args:
        queries: 待检索的查询列表。
        filter_names: 主体过滤名单（文旅为景点名），为空表示不限制。
        search_one: 单路检索回调，签名为 (dense_vector, sparse_vector, filter_names, limit) -> list[dict]。
        limit: 单路返回条数上限。

    Returns:
        list[dict]: 多路合并并按去重后顺序排列的文档列表。
    """
    if not queries:
        return []

    # 关键优化：把所有查询合并成一次批量编码。
    # BGE-M3 非线程安全、encode 被全局锁串行化，逐条调用会让耗时随路数线性叠加。
    embedding_result = llm_provider.embed_documents(queries)

    def _search_by_index(index: int) -> list[dict]:
        return search_one(
            embedding_result["dense"][index],
            embedding_result["sparse"][index],
            filter_names,
            limit,
        )

    merged: list[dict] = []
    seen: set = set()
    with ThreadPoolExecutor(max_workers=min(MULTI_QUERY_WORKERS, len(queries))) as pool:
        for docs in pool.map(_search_by_index, range(len(queries))):
            for doc in docs or []:
                chunk_id = doc.get("chunk_id")
                # 同一 chunk 被多路命中时只保留首次（分数最高的那一路先返回）
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                merged.append(doc)

    logger.info(f"多改写检索完成：{len(queries)} 路查询，去重后 {len(merged)} 条结果")
    return merged
