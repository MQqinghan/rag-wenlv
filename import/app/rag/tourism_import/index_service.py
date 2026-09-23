"""
文旅入库服务模块，负责创建 Milvus tourism_chunks 与 tourism_entity 集合并写入数据。
集合 schema 含文旅业务字段：item_name / content_type / region / cultural_theme / category / source_type / extra_meta。
"""
from pymilvus import DataType

from app.shared.runtime.logger import logger, step_log
from app.infra.vectorstore import milvus_gateway
from app.rag.common.chunk_config import (
    MILVUS_CHUNK_CONTENT_MAX_LENGTH,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """校验导入状态中是否已经生成切块结果。"""
    chunks = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空,无法继续业务!!")
        raise ValueError("chunks为空,无法继续业务!!")
    return chunks


@step_log("prepare_tourism_chunks_collection")
def prepare_tourism_chunks_collection() -> None:
    """
    准备 tourism_chunks 集合（不存在则创建）。
    schema 含文旅业务标量字段，支撑查询侧"按主体/地区/内容类型/文化主题过滤"。
    """
    milvus_client = milvus_gateway.client()
    collection_name = milvus_gateway.tourism_chunks_collection
    if milvus_client.has_collection(collection_name=collection_name):
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    # 文旅业务字段（高频过滤字段，必须显式列以支持建索引）
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="region", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="cultural_theme", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field(field_name="extra_meta", datatype=DataType.JSON)
    # 来源路径（上传文件本地路径），对齐需求"来源路径或资源链接"
    schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=512)
    # 通用字段
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=MILVUS_CHUNK_CONTENT_MAX_LENGTH)
    # 向量字段
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.success(f"文旅切块集合[{collection_name}]创建成功")


@step_log("prepare_tourism_entity_collection")
def prepare_tourism_entity_collection() -> None:
    """准备 tourism_entity 主体集合（景点/文化主题/游记标题向量索引）。"""
    milvus_client = milvus_gateway.client()
    collection_name = milvus_gateway.tourism_entity_collection
    if milvus_client.has_collection(collection_name=collection_name):
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="region", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="cultural_theme", datatype=DataType.VARCHAR, max_length=256)
    # 来源路径（上传文件本地路径），对齐需求"来源路径或资源链接"；与 tourism_chunks 保持一致
    schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.success(f"文旅主体集合[{collection_name}]创建成功")


@step_log("remove_old_chunks_by_file_title")
def remove_old_chunks_by_file_title(file_title: str) -> None:
    """
    按文件名删旧数据——file_title 来自上传文件名，同一文件重导时 100% 稳定命中。

    幂等删除策略（重要）：只按 file_title 删，绝不按 item_name 删。
    同一旅游主体可能有多个来源文件（如"西安攻略"与"西安美食/住宿"），
    若按主体名删旧会把同主体其它文件的切块一并删除（同名主体的数据会被误删）。
    不同文件各自保留即可。
    """
    escaped = escape_milvus_string(file_title)
    milvus_gateway.client().delete(
        collection_name=milvus_gateway.tourism_chunks_collection,
        filter=f"file_title == {escaped}",
    )


@step_log("insert_chunks")
def insert_chunks(chunks: list[dict]) -> None:
    """批量插入 chunks 到 tourism_chunks 集合。"""
    result = milvus_gateway.client().insert(
        collection_name=milvus_gateway.tourism_chunks_collection,
        data=chunks,
    )
    logger.info(f"文旅数据插入成功! 总条数:{result.get('insert_count', 0)}")


@step_log("upsert_entity")
def upsert_entity(
    meta,
    file_title: str,
    dense_vector: list[float],
    sparse_vector: dict,
    source_path: str = "",
) -> None:
    """主体入库（幂等 upsert）：同文件重导时先删旧记录再插入。

    只按 file_title 删（file_title 来自上传文件名，重导 100% 稳定命中）。
    不按 item_name 删：同一主体可有多个来源文件，各自保留一条主体行，
    避免后导文件把同主体其它文件的主体覆盖删除。
    source_path 为来源文件本地路径（第6点共通来源路径字段，缺省为空串兼容旧调用）。
    """
    milvus_client = milvus_gateway.client()
    collection_name = milvus_gateway.tourism_entity_collection
    prepare_tourism_entity_collection()

    item_name = meta.item_name
    try:
        escaped_title = escape_milvus_string(file_title)
        milvus_client.delete(collection_name=collection_name, filter=f"file_title == {escaped_title}")
    except Exception as e:
        logger.warning(f"主体按文件名[{file_title}]删除失败，继续插入：{e}")

    data = [{
        "file_title": file_title,
        "item_name": item_name,
        "content_type": meta.content_type.value,
        "region": meta.region or "",
        "cultural_theme": meta.cultural_theme or "",
        "source_path": source_path,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }]
    milvus_client.insert(collection_name=collection_name, data=data)
    logger.success(f"主体[{item_name}]向量入库成功，集合[{collection_name}]")


@step_log("index_chunks")
def index_chunks(state: dict) -> dict:
    """入库服务总入口：校验 → 建集合 → 删旧（仅按 file_title）→ 插入新。

    幂等语义：同一文件重导 = 覆盖更新；不同文件即使同主体也互不覆盖，
    保证同一主体的攻略/美食/住宿/线路等来源文件都能共存检索。
    """
    chunks = require_chunks(state)
    prepare_tourism_chunks_collection()
    file_title = state.get("file_title", "")
    if file_title:
        remove_old_chunks_by_file_title(file_title)
    insert_chunks(chunks)
    return state


if __name__ == "__main__":
    # 单元测试：构造带向量的 chunks 验证入库（需 Milvus 连接）
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    load_dotenv(os.path.join(project_root, ".env"))

    logger.info("===== tourism index_service 单元测试 =====")
    dim = 1024
    test_state = {
        "task_id": "test_tourism_index_001",
        "item_name": "兵马俑",
        "file_title": "西安景点资料",
        "chunks": [
            {
                "content": "兵马俑位于陕西西安，是秦始皇陵陪葬坑，5A景区，门票120元。",
                "title": "景点简介",
                "item_name": "兵马俑",
                "content_type": "景点信息",
                "region": "陕西西安",
                "cultural_theme": "",
                "category": "历史古迹",
                "source_type": "pdf",
                "extra_meta": {"level": "5A", "ticket_price": "120元"},
                "parent_title": "西安景点资料",
                "part": 1,
                "file_title": "西安景点资料",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            }
        ]
    }

    if not os.getenv("MILVUS_URL"):
        logger.warning("未设置 MILVUS_URL，跳过入库测试")
    else:
        try:
            result_state = index_chunks(test_state)
            logger.info(f"入库完成，chunks 数：{len(result_state.get('chunks', []))}")
        except Exception as e:
            logger.error(f"入库测试失败（检查 Milvus 连接）: {e}")
    logger.info("===== tourism index_service 测试结束 =====")
