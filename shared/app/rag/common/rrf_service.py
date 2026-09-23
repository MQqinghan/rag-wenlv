"""
RRF 融合服务模块，负责将本地多路检索结果（普通向量 + 多改写 + 关键词）进行倒数排名融合。
只处理本地知识库召回，不参与外网网页结果合并，也不承担最终全局重排能力。
"""
from app.shared.runtime.logger import logger, step_log

# RRF 融合默认配置
# RRF 公式平滑系数，避免排名过低导致分数趋近于 0，同时防止前几名优势过于极端
RRF_K = 60
# 最终返回的融合结果数量
RRF_TOP = 5


@step_log("fuse_by_rrf")
def fuse_by_rrf(state: dict) -> list[dict]:
    """
    RRF 融合服务总入口
    负责将本地三路检索结果（普通向量 + 多改写 + 关键词）进行加权融合
    三路默认权重均为 1.0（RRF 只看排名不看原始分数，权重相等即公平投票）

    Args:
        state: 查询图当前状态，需包含 embedding_chunks、hyde_embedding_chunks 与 keyword_chunks。

    Returns:
        list[dict]: 按融合分数倒序排列的 Top N 融合结果。
    """
    # 1. 校验三路输入结果是否合法
    embedding_chunks, hyde_embedding_chunks, keyword_chunks = validate_rrf_inputs(state)

    # 2. 构造 RRF 入参：(结果列表, 权重)
    param_list = [
        (embedding_chunks, 1.0),       # 普通向量检索，权重1.0
        (hyde_embedding_chunks, 1.0),  # 多改写（含 HyDE）增强检索，权重1.0
        (keyword_chunks, 1.0),         # BM25 关键词检索，权重1.0
    ]

    # 3. 执行 RRF 融合并返回最终结果
    return reciprocal_rank_fusion(param_list)


@step_log("validate_rrf_inputs")
def validate_rrf_inputs(state: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """
    校验 RRF 融合所需的三路本地检索结果是否合法
    允许多路为空（多改写/关键词均可能因开关或跳过条件返回空），但不允许三路都为空

    Args:
        state: 查询图当前状态。

    Returns:
        tuple[list[dict], list[dict], list[dict]]: 依次返回普通向量、多改写、关键词检索结果。
    """
    # 获取普通向量检索结果、多改写增强检索结果、关键词检索结果
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_chunks = state.get("hyde_embedding_chunks", [])
    keyword_chunks = state.get("keyword_chunks", [])

    # 核心校验：三路都为空时不再抛异常炸图（2026-09-11 neg-003 实证：
    # Milvus 连接抖动/网络代理故障会使三路全空，ValueError 直接让整个查询图 500，
    # 用户看到的是异常而非拒答）。降级为返回空列表，由下游素材闸门/空召回兜底
    # 接管（answer_output_service / resolve_plan_material 均有空结果拒答路径）。
    if not embedding_chunks and not hyde_chunks and not keyword_chunks:
        logger.error("embedding_chunks / hyde_embedding_chunks / keyword_chunks 均为空，RRF 降级返回空列表（下游走拒答兜底）")
        return [], [], []

    return embedding_chunks, hyde_chunks, keyword_chunks


@step_log("reciprocal_rank_fusion")
def reciprocal_rank_fusion(
        param_list: list[tuple[list[dict], float]],
        *,
        k: int = RRF_K,
        top: int = RRF_TOP,
) -> list[dict]:
    """
    RRF（倒数排名融合）核心算法实现
    不依赖原始分数，只根据排名位置计算融合得分，支持多路加权融合

    Args:
        param_list: 列表，每一项是 (检索结果列表, 权重)
        k: 平滑系数，默认60
        top: 最终返回Top N条结果

    Returns:
        list[dict]: 按RRF融合分数倒序排列后的统一结果列表
    """
    # 存储每个 chunk_id 的总融合分数
    score_dict: dict[str, float] = {}
    # 存储每个 chunk_id 对应的原始片段信息
    entity_dict: dict[str, dict] = {}

    # 遍历每一路检索结果及其权重
    for chunks_list, weight in param_list:
        # 遍历当前路的所有片段，rank 从 1 开始计数（第一名=1）
        for rank, chunk in enumerate(chunks_list, start=1):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue  # 无chunk_id则跳过，无法融合

            # RRF 核心公式：1/(k + 排名) * 权重，累加到总分
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (k + rank)) * weight
            # 保留片段原文信息（只存一次，避免覆盖）
            entity_dict.setdefault(chunk_id, chunk)

    # 组装最终结果：把分数和原文信息合并
    document_list = []
    for chunk_id, score in score_dict.items():
        document = entity_dict.get(chunk_id, {}).copy()
        # 将计算好的 RRF 总分写入结果
        document["score"] = score
        document_list.append(document)

    # 按融合分数从高到低排序
    document_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # 返回 Top N 条最终融合结果
    final_documents = document_list[:top]
    logger.info(f"RRF融合完成,输入路数:{len(param_list)},输出条数:{len(final_documents)}")
    return final_documents
