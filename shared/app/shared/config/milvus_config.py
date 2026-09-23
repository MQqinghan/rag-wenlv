"""
Milvus 配置模块，负责读取向量库相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class MilvusConfig:
    milvus_url: str
    # 文旅集合：切片（chunks）+ 主体（entity）
    tourism_chunks_collection: str
    tourism_entity_collection: str

milvus_config = MilvusConfig(
    milvus_url=env_str("MILVUS_URL"),
    tourism_chunks_collection=env_str("TOURISM_CHUNKS_COLLECTION", "tourism_chunks"),
    tourism_entity_collection=env_str("TOURISM_ENTITY_COLLECTION", "tourism_entity"),
)