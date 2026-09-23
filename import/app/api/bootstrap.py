"""
应用启动引导：在应用进程（导入/查询服务）启动早期注册预热与缓存失效钩子。

依赖反转（依赖注入）：shared 层（warmup / data_version_service）只暴露注册表，不直接依赖业务层
（rag / infra）。本模块属于 api 组装层，允许依赖各层，在 import 时把具体钩子注册进 shared 注册表，
既消除静态反向依赖（支撑分布式拆包），又保持原有启动行为完全不变。
"""
from collections.abc import Callable

from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.warmup import register_warmup_hook
from app.shared.runtime.data_version_service import register_query_cache_invalidator


@step_log("warmup_embedding_model")
def _warmup_embedding_model() -> None:
    """预加载 BGE-M3 并跑通一次稠密+稀疏向量编码。"""
    from app.infra.llm import llm_provider

    llm_provider.embed_documents(["预热"])
    logger.debug("BGE-M3 预热完成（含 CUDA 初始化）")


@step_log("warmup_reranker_model")
def _warmup_reranker_model() -> None:
    """预加载 bge-reranker-large 并跑通一次打分。"""
    from app.infra.llm import llm_provider

    reranker = llm_provider.reranker_model()
    reranker.compute_score([["预热", "预热文档"]], normalize=True)
    logger.debug("重排模型预热完成（含 CUDA 初始化）")


@step_log("warmup_milvus_client")
def _warmup_milvus_client() -> None:
    """提前建立 Milvus 连接，避免首查时才做握手。"""
    from app.infra.vectorstore import milvus_gateway

    client = milvus_gateway.client()
    if client is None:
        logger.warning("Milvus 客户端预热失败：连接返回空")
        return
    logger.debug("Milvus 客户端连接预热完成")


@step_log("warmup_bm25_index")
def _warmup_bm25_index() -> None:
    """预构建 BM25 关键词索引，把首次查询的数秒分词成本前移到启动阶段。"""
    from app.infra.vectorstore import milvus_gateway
    from app.rag.common.bm25_service import get_bm25_index
    from app.rag.tourism_query.search_keyword_service import BM25_OUTPUT_FIELDS as tourism_fields

    get_bm25_index(milvus_gateway.tourism_chunks_collection, tourism_fields)


def _invalidate_bm25_cache() -> None:
    from app.rag.common.bm25_service import invalidate_bm25_cache

    invalidate_bm25_cache()


def _register_hooks() -> None:
    register_warmup_hook("embedding", _warmup_embedding_model)
    register_warmup_hook("reranker", _warmup_reranker_model)
    register_warmup_hook("milvus", _warmup_milvus_client)
    register_warmup_hook("bm25_index", _warmup_bm25_index)
    register_query_cache_invalidator("BM25索引缓存", _invalidate_bm25_cache)


_register_hooks()
