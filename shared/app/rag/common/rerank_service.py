"""
重排服务模块，负责对本地 RRF 融合结果与外网网页结果做全局统一语义精排。
查询链中第一次真正做跨来源统一精排：本地与网页候选进入同一评分体系比较。

超长候选文本处理策略（性能关键）：
重排模型输入上限 512 token，超长候选必须先压缩再打分。历史实现是对每条超长候选
【串行】调用 LLM 做生成式精简——单次查询最多触发 9 次 LLM 调用、累计 30~48s，
是整个查询链路最大的性能瓶颈。现改为按来源分治的零 LLM 压缩：
- 网页类（type=web）：snippet 本身就是摘要，直接按字符硬截断，信息损失可忽略
- 本地类（type=milvus）：句子级【抽取式】压缩，按与问题的词汇重叠度挑关键句并保序，
  保证被 tokenizer 截断时留下的是与问题最相关的内容
仅当 RERANK_LLM_REFINE_ENABLE=1 时才回退到 LLM 精简，且强制并行执行。
"""
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm import llm_provider
from app.shared.config.common import env_bool, env_int
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger

# ====================== 重排全局配置 ======================
RERANK_MAX_TOPK: int = 10                # 动态截断最多保留10条结果
RERANK_MIN_TOPK: int = 1                 # 动态截断最少保留1条结果
RERANK_GAP_RATIO: float = 2              # 分数断崖比例阈值（用于动态截断）
RERANK_GAP_ABS: float = 2                # 分数断崖绝对值阈值
# KB 保底席位：答案生成只取 reranked_docs[:5]，实测(tr-002)网页候选可能把本地 KB chunk
# 全部挤出前 5，答案失去知识库出处。本窗口内为本地候选保底 RERANK_MIN_KB_TOPK 个席位
RERANK_KB_QUOTA_WINDOW: int = 5
RERANK_MIN_KB_TOPK: int = 3
RERANK_MAX_INPUT_TOKENS: int = 512       # 重排模型最大输入token长度
RERANK_SUMMARY_CHAR_RATIO: float = 1.3   # 中文token与字符换算比例 1token≈1.3字符
RERANK_MIN_SUMMARY_CHARS: int = 50       # 文本精简后最小字符数

# 是否启用 LLM 生成式精简（默认关闭：抽取式压缩零延迟且效果足够，LLM 精简单次可引入 30s+ 串行耗时）
RERANK_LLM_REFINE_ENABLE: bool = env_bool("RERANK_LLM_REFINE_ENABLE", default=False)
# LLM 精简的并行度（仅在开启 LLM 精简时生效，避免串行累加）
RERANK_REFINE_WORKERS: int = env_int("RERANK_REFINE_WORKERS", default=6)
# 句子切分正则：中英文句末标点 + 换行
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
# 抽取式压缩用停用词：高频虚词/标点，不参与与问题的重叠度计算
_STOP_TERMS = frozenset({
    "的", "了", "是", "在", "和", "与", "及", "或", "有", "为", "对", "把", "被", "给",
    "这", "那", "之", "其", "此", "该", "个", "们", "我", "你", "他", "她", "它",
    "什么", "怎么", "如何", "哪些", "请问", "可以", "能", "会", "要", "就", "也", "都",
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
})


@step_log("rerank_documents")
def rerank_documents(state: dict) -> list[dict]:
    """
    重排节点主入口
    流程：校验输入 → 合并本地+网页 → 模型打分排序 → 动态截断
    输出最终高质量候选文档列表

    Args:
        state: 查询图当前状态，需包含 rrf_chunks / web_search_docs / rewritten_query。

    Returns:
        list[dict]: 动态截断后的精排候选文档列表。
    """
    # 1. 校验输入
    rrf_chunks, web_search_docs = validate_rerank_inputs(state)
    # 1.5 规划类问题剔除网页候选（行程拼装以本地知识库 + 工具简报为准）。
    # 实测(plan-001)：规划问句会带出大量网页片段，rerank 后把 KB chunk 全部挤出 top5，
    # 行程模板只看到网页摘要 → 日期/交通数值幻觉（编造"2024年3月25日""地铁44分钟"）。
    # KB 候选为空时保留网页路兜底，避免无上下文可用。
    if state.get("is_plan") and rrf_chunks and web_search_docs:
        logger.info(
            f"规划类问题剔除网页候选 {len(web_search_docs)} 条，"
            f"仅保留本地 KB {len(rrf_chunks)} 条参与精排"
        )
        web_search_docs = []
    # 2. 统一格式合并两路结果
    merged = merge_rrf_and_web(rrf_chunks, web_search_docs)
    # 3. 重排模型打分 + 排序
    sorted_docs = score_and_sort_chunks(state, merged)
    # 4. 动态截断 + KB 保底席位（防止网页候选刷屏挤掉知识库出处），返回最优结果
    return enforce_kb_quota(dynamic_topk(sorted_docs), sorted_docs)


@step_log("validate_rerank_inputs")
def validate_rerank_inputs(state: dict) -> tuple[list[dict], list[dict]]:
    """
    校验重排节点输入是否合法
    允许本地结果为空、联网结果为空，但不允许两者同时为空

    Args:
        state: 查询图当前状态。

    Returns:
        tuple[list[dict], list[dict]]: 依次返回本地融合结果、联网搜索结果。
    """
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])

    # 必须至少一路有数据，否则 rerank 无内容可处理
    if not rrf_chunks and not web_search_docs:
        logger.error("rrf_chunks 与 web_search_docs 均为空，rerank 重排无有效数据！")
        raise ValueError("rerank 重排失败：本地融合结果与联网搜索结果均为空")

    return rrf_chunks, web_search_docs


@step_log("merge_rrf_and_web")
def merge_rrf_and_web(
    rrf_chunks: list[dict],
    web_search_docs: list[dict],
) -> list[dict]:
    """
    统一合并本地知识库结果 + 联网搜索结果
    统一字段格式，方便后续重排模型统一打分

    网页结果绕过 RRF、score 置 0.0，由 reranker 按「问题-文本」重新打分；
    在 enforce_kb_quota 的保底席位与规划闸门里按网页对待，会被挤占/剔除。

    Args:
        rrf_chunks: 本地 RRF 融合后的候选列表。
        web_search_docs: 联网搜索得到的网页候选列表。

    Returns:
        list[dict]: 统一为 title/text/url/type/score 结构的候选列表。
    """
    final_chunk_list: list[dict] = []

    # 处理本地RRF融合结果
    for chunk in rrf_chunks or []:
        final_chunk_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),    # 本地知识库取content字段
            "url": None,                     # 本地数据无URL
            "type": chunk.get("type", "milvus"),
            "score": chunk.get("score", 0.0),
            # 来源文件名（Milvus chunk 字段），供回答末尾的出处标注使用
            "file_title": chunk.get("file_title"),
            # 文旅内容类型与专属字段（chunk 字段），供分类/详情回答引用
            "content_type": chunk.get("content_type"),
            "extra_meta": chunk.get("extra_meta"),
            # 来源路径（上传文件本地路径），供回答末尾"参考资料"展示；网页块无此字段
            "source_path": chunk.get("source_path"),
        })

    # 处理联网搜索网页结果
    for doc in web_search_docs or []:
        final_chunk_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"),      # 网页结果取摘要snippet
            "url": doc.get("url"),           # 网页保留URL
            "type": "web",
            "score": 0.0,
            "content_type": None,
            "extra_meta": None,
            "source_path": None,
        })


    logger.info(
        f"多路数据统一格式完成,rrf路:{len(rrf_chunks)}条,web路:{len(web_search_docs)}条,"
        f"合并后:{len(final_chunk_list)}条"
    )
    return final_chunk_list


@step_log("split_sentences")
def split_sentences(text: str) -> list[str]:
    """
    按中英文句末标点与换行切分句子，保留标点本身（保证压缩后仍是可读文本）。

    Args:
        text: 待切分的原文。

    Returns:
        list[str]: 切分后的句子列表（已去空白空句）。
    """
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in parts if part and part.strip()]


@step_log("extract_terms")
def extract_terms(text: str) -> set[str]:
    """
    提取用于相关性打分的词项集合：优先 jieba 分词，失败时退化为字符 2-gram。
    过滤停用词与单字噪声，避免高频虚词稀释重叠度信号。

    Args:
        text: 输入文本。

    Returns:
        set[str]: 词项集合。
    """
    if not text:
        return set()
    try:
        import jieba

        tokens = jieba.lcut(text)
    except Exception:
        tokens = []
    if tokens:
        return {
            token.lower()
            for token in tokens
            if len(token) > 1 and token.lower() not in _STOP_TERMS and token.strip()
        }
    # 退化路径：无分词器时用字符 2-gram
    cleaned = re.sub(r"\s+", "", text)
    return {cleaned[i:i + 2].lower() for i in range(len(cleaned) - 1)}


@step_log("condense_text_for_rerank")
def condense_text_for_rerank(*, question: str, text: str, limit: int) -> str:
    """
    句子级抽取式压缩：按与问题的词汇重叠度挑出关键句，并按原顺序拼接。

    与 LLM 生成式精简的差异：不改写原文、零网络延迟（毫秒级），
    但能让"与问题最相关的句子"排到前面——重排模型截断时保留的正是这些句子。

    Args:
        question: 当前问题，用于计算句子相关性。
        text: 超长的候选文档文本。
        limit: 压缩后的最大字符数。

    Returns:
        str: 压缩后的候选文本。
    """
    if len(text) <= limit:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text[:limit]

    query_terms = extract_terms(question)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = extract_terms(sentence)
        if not sentence_terms:
            continue
        overlap = len(query_terms & sentence_terms)
        # 覆盖率（问题侧，权重更高）+ 密度（句子侧，抑制长句刷分）+ 位置奖励（前3句通常含主旨）
        coverage = overlap / max(len(query_terms), 1)
        density = overlap / len(sentence_terms)
        position_bonus = 0.15 if index < 3 else 0.0
        scored.append((coverage * 2.0 + density + position_bonus, index, sentence))

    if not scored:
        return text[:limit]

    # 先按分数取句，再按原文顺序还原，保证语义连贯
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked: list[tuple[int, str]] = []
    total_chars = 0
    for _, index, sentence in scored:
        if total_chars >= limit:
            break
        remain = limit - total_chars
        picked.append((index, sentence if len(sentence) <= remain else sentence[:remain]))
        total_chars += len(sentence)

    if not picked:
        return text[:limit]

    picked.sort(key=lambda item: item[0])
    condensed = "".join(sentence for _, sentence in picked)
    logger.debug(f"抽取式压缩完成：{len(text)}字符 → {len(condensed)}字符（上限{limit}）")
    return condensed


@step_log("truncate_text_for_rerank")
def truncate_text_for_rerank(text: str, limit: int) -> str:
    """
    硬截断：网页 snippet 本身已是摘要，截断的信息损失可忽略，无需任何压缩计算。

    Args:
        text: 候选文本。
        limit: 最大字符数。

    Returns:
        str: 截断后的文本。
    """
    return text if len(text) <= limit else text[:limit]


@step_log("collect_oversized_items")
def collect_oversized_items(
    question: str,
    final_chunk_list: list[dict],
    limit: int,
) -> list[tuple[int, str]]:
    """
    用字符数快速筛出需要压缩的候选，避免对每条候选都做 token 编码。

    字符数阈值由 token 上限反推：limit_chars = (MAX_TOKENS - query_tokens - 4) / CHAR_RATIO，
    这里用问题字符数近似 query_tokens，宁可多压几条也不漏压（漏压会被 tokenizer 硬截断丢信息）。

    Args:
        question: 当前问题。
        final_chunk_list: 统一结构后的候选文档列表。
        limit: 已算好的单条最大字符数。

    Returns:
        list[tuple[int, str]]: (候选下标, 原文) 列表。
    """
    oversized: list[tuple[int, str]] = []
    for index, item in enumerate(final_chunk_list):
        text = item.get("text") or ""
        if len(text) > limit:
            oversized.append((index, text))
    logger.info(f"超长候选筛出 {len(oversized)}/{len(final_chunk_list)} 条（字符阈值 {limit}）")
    return oversized


@step_log("build_question_pairs")
def build_question_pairs(question: str, final_chunk_list: list[dict], reranker) -> list[list[str]]:
    """
    构建重排模型输入对：[问题, 文本]，超长候选按来源分治压缩。

    压缩策略（零 LLM，毫秒级）：
    - 网页类（type=web）：硬截断
    - 本地类（type=milvus）：句子级抽取式压缩
    - RERANK_LLM_REFINE_ENABLED=1 时才走 LLM 精简，且所有候选并行压缩

    Args:
        question: 用于重排的改写后问题。
        final_chunk_list: 统一结构后的候选文档列表。
        reranker: 重排模型实例，复用其自带 tokenizer 计算 token 长度。

    Returns:
        list[list[str]]: 可直接送入 compute_score 的问答对列表。
    """
    tokenizer = reranker.tokenizer
    # 对问题进行token编码（不添加特殊符号）
    query_tokens = tokenizer.encode(question, add_special_tokens=False)
    # 重排模型拼接近似为 ['<s>', 问题, '</s>', '</s>', 答案, '</s>']，额外占4个特殊token
    limit = max(
        RERANK_MIN_SUMMARY_CHARS,
        int((RERANK_MAX_INPUT_TOKENS - len(query_tokens) - 4) / RERANK_SUMMARY_CHAR_RATIO),
        20,
    )

    # 先用字符数粗筛，绝大多数短候选直接跳过（省掉逐条 token 编码）
    condensed_map: dict[int, str] = {}
    oversized = collect_oversized_items(question, final_chunk_list, limit)
    if oversized:
        question_pairs_parallel = RERANK_LLM_REFINE_ENABLE and len(oversized) > 1
        if question_pairs_parallel:
            # LLM 精简模式：并行执行，避免串行累加（历史实现 9 条串行 = 34s）
            with ThreadPoolExecutor(max_workers=min(RERANK_REFINE_WORKERS, len(oversized))) as pool:
                futures = {
                    index: pool.submit(
                        summarize_long_rerank_text, question=question, answer=text, limit=limit
                    )
                    for index, text in oversized
                }
                for index, future in futures.items():
                    condensed_map[index] = future.result()
        else:
            for index, text in oversized:
                item = final_chunk_list[index]
                if RERANK_LLM_REFINE_ENABLE:
                    condensed_map[index] = summarize_long_rerank_text(
                        question=question, answer=text, limit=limit
                    )
                elif item.get("type") == "web":
                    # 网页摘要：硬截断即可，截断的都是尾部次要信息
                    condensed_map[index] = truncate_text_for_rerank(text, limit)
                else:
                    # 本地长文：抽取式压缩，把与问题最相关的句子顶到前面
                    condensed_map[index] = condense_text_for_rerank(
                        question=question, text=text, limit=limit
                    )

    question_pairs: list[list[str]] = []
    for index, item in enumerate(final_chunk_list):
        answer = condensed_map.get(index, item.get("text") or "")
        question_pairs.append([question, answer])

    return question_pairs


@step_log("summarize_long_rerank_text")
def summarize_long_rerank_text(question: str, answer: str, limit: int) -> str:
    """
    超长文本精简：当文本超过重排模型最大输入长度时，调用LLM精简内容
    保证输入长度合规，同时保留与问题相关的核心信息

    Args:
        question: 当前问题。
        answer: 超长的候选文档文本。
        limit: 精炼后的最大字符数。

    Returns:
        str: 精炼后的候选文本。
    """
    prompt = load_prompt(
        "common/rerank_text_refine",
        question=question,
        answer=answer,
        limit=limit,
    )
    messages = [
        SystemMessage(content="你现在是文本精简提炼专家。根据用户发送的文本完成文本精炼要求。"),
        HumanMessage(content=prompt),
    ]
    # 调用大模型精简文本并返回
    refined_answer = (llm_provider.chat() | StrOutputParser()).invoke(messages)
    logger.debug(f"重排前文本精炼完成,原始长度:{len(answer)},精炼后长度:{len(refined_answer)}")
    return refined_answer


@step_log("score_and_sort_chunks")
def score_and_sort_chunks(state: dict, final_chunk_list: list[dict]) -> list[dict]:
    """
    调用重排模型对所有候选文档打分，并按分数从高到低排序

    Args:
        state: 查询图当前状态，用于读取当前问题。
        final_chunk_list: 统一结构后的候选文档列表。

    Returns:
        list[dict]: 写回相关性分数并降序排序后的候选列表。
    """
    if not final_chunk_list:
        return []

    # 获取用户查询问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    # 获取重排模型实例
    reranker = llm_provider.reranker_model()
    # 构建模型输入对
    question_pairs = build_question_pairs(rewritten_query, final_chunk_list, reranker)
    # 模型打分（归一化）
    score_list = reranker.compute_score(question_pairs, normalize=True)

    # 将分数写入文档
    for score, chunk in zip(score_list, final_chunk_list):
        chunk["score"] = round(score, 4)

    # 按分数降序排序
    final_chunk_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return final_chunk_list


@step_log("dynamic_topk")
def dynamic_topk(chunk_list_score_sorted: list[dict]) -> list[dict]:
    """
    动态值截断：根据分数断崖自动决定保留多少条结果
    不是固定取前N条，而是找到分数突变的位置截断

    Args:
        chunk_list_score_sorted: 已按分数降序排序的候选列表。

    Returns:
        list[dict]: 截断后的最终候选列表。
    """
    min_topk = RERANK_MIN_TOPK
    max_topk = min(RERANK_MAX_TOPK, len(chunk_list_score_sorted))
    gap_ratio = RERANK_GAP_RATIO
    max_gap = RERANK_GAP_ABS
    topk = max_topk  # 默认取最大条数

    # 遍历寻找分数断崖
    if topk > min_topk:
        for index in range(min_topk - 1, max_topk - 1):
            score_1 = chunk_list_score_sorted[index].get("score", 0.0)
            score_2 = chunk_list_score_sorted[index + 1].get("score", 0.0)
            abs_score = score_1 - score_2  # 分数差
            ratio_score = abs_score / (score_1 + 1e-7)  # 比例差

            # 发现断崖 → 在此处截断
            if abs_score > max_gap or ratio_score > gap_ratio:
                topk = index + 1
                break

    # 返回截断后的结果
    return chunk_list_score_sorted[:topk]


@step_log("enforce_kb_quota")
def enforce_kb_quota(topk_docs: list[dict], sorted_docs: list[dict]) -> list[dict]:
    """
    KB 保底席位：在精排输出前 RERANK_KB_QUOTA_WINDOW 条（与答案生成取 reranked_docs[:5]
    对齐）中，为本地 KB 候选保底 RERANK_MIN_KB_TOPK 个席位。不足时用排序表中窗口外的
    KB 候选，从低到高置换窗口内的网页席位。

    实测(tr-002)：RRF top5 里的《成都美食推荐》经精排后被网页候选挤出前 5，
    答案既丢金标出处、又只能引用网页摘要。纯本地/纯网页场景（含规划类去 web 后）无感透传。
    注：normalize=True 时分数为 0~1 的 sigmoid 值，断崖阈值(差值>2)永不触发，
    dynamic_topk 实际等价于取 min(10, N) 条——网页挤出的防线就是本保底席位。

    Args:
        topk_docs: dynamic_topk 截断后的精排候选（已按分数降序）。
        sorted_docs: 全量候选（同样按分数降序），用于窗口外 KB 回填。

    Returns:
        list[dict]: 完成 KB 保底置换后的最终候选列表。
    """
    if RERANK_MIN_KB_TOPK <= 0 or not topk_docs:
        return topk_docs

    window = list(topk_docs[:RERANK_KB_QUOTA_WINDOW])
    kb_missing = RERANK_MIN_KB_TOPK - sum(1 for d in window if d.get("type") != "web")
    if kb_missing <= 0:
        return topk_docs

    window_ids = {id(d) for d in window}
    backfill = [
        d for d in sorted_docs
        if d.get("type") != "web" and id(d) not in window_ids
    ][:kb_missing]
    if not backfill:
        return topk_docs  # 窗口外已无 KB 候选（本地路本就为空），不强造席位

    # 从排名最低的网页席位开始置换，尽量保住高分网页候选
    web_slots = [i for i, d in enumerate(window) if d.get("type") == "web"]
    replaced = 0
    for slot, doc in zip(reversed(web_slots), backfill):
        window[slot] = doc
        replaced += 1
    logger.info(f"KB保底席位: 回填 {replaced} 条本地候选置换窗口内网页席位")
    return window + list(topk_docs[RERANK_KB_QUOTA_WINDOW:])
