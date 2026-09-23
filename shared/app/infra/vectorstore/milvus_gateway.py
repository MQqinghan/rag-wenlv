"""
Milvus 门面模块，统一封装向量库客户端与检索相关操作。
"""
from typing import Any

from app.shared.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.infra.config import infra_config


class MilvusGateway:
    @property
    def tourism_chunks_collection(self) -> str:
        """
        获取文旅切块集合名称。

        Returns:
            str: Milvus 中存放文旅知识切块的集合名。
        """
        return infra_config.milvus.tourism_chunks_collection

    @property
    def tourism_entity_collection(self) -> str:
        """
        获取文旅主体集合名称。

        Returns:
            str: Milvus 中存放文旅主体（景点/主题/游记标题）向量的集合名。
        """
        return infra_config.milvus.tourism_entity_collection

    def client(self):
        """
        获取 Milvus 客户端实例。

        Returns:
            Any: 底层 Milvus 客户端对象。
        """
        return get_milvus_client()

    def create_requests(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        *,
        expr: str = None,
        limit: int = 5,
    ):
        """
        创建 Milvus 混合检索请求对象。

        Args:
            dense_vector: 稠密向量表示。
            sparse_vector: 稀疏向量表示。
            expr: 可选过滤表达式，用于限定检索范围。
            limit: 单路检索返回条数上限。

        Returns:
            Any: 底层 Milvus 混合检索请求列表。
        """
        return create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[Any],
        ranker_weights: tuple[float, float] = (0.5, 0.5),
        norm_score: bool = False,
        limit: int = 5,
        output_fields: list[str] | None = None,
        search_params: dict | None = None,
    ):
        """
        执行 Milvus 混合检索。

        Args:
            collection_name: 目标集合名称。
            reqs: 检索请求列表。
            ranker_weights: 稠密与稀疏路召回结果的融合权重。
            norm_score: 是否对分数做归一化。
            limit: 最终返回条数上限。
            output_fields: 需要返回的字段列表。
            search_params: 额外检索参数。

        Returns:
            Any: Milvus 返回的原始检索结果。
        """
        return hybrid_search(
            client=self.client(),
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )


milvus_gateway = MilvusGateway()
