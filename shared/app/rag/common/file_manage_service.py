"""
已导入文件管理服务模块：按领域列出知识库中已导入的文件、按文件名撤回（删除向量数据）。

每个领域的知识数据落在两张 Milvus 集合上（记录均带 file_title，源自上传文件名）：
- 文旅：tourism_chunks（切块）+ tourism_entity（主体）

撤回 = 按 file_title 精确删除该域两张集合中对应行。删除结果以「直接查询残留行数」复核为准，
不依赖 count(*) 聚合（可能读到陈旧快照）。
"""
from collections import Counter

from app.infra.vectorstore import milvus_gateway
from app.shared.runtime.logger import logger
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string

DOMAIN_TOURISM = "tourism"

# 域 -> (展示名, 切块集合属性名, 主体集合属性名)；集合真实名通过 milvus_gateway 属性读取（跟随 .env）
_DOMAIN_META = {
    DOMAIN_TOURISM: {
        "label": "文旅",
        "chunk_coll_attr": "tourism_chunks_collection",
        "entity_coll_attr": "tourism_entity_collection",
    },
}


def _domain_meta(domain: str) -> dict:
    meta = _DOMAIN_META.get(domain)
    if not meta:
        raise ValueError(f"非法领域[{domain}]，必须为 {DOMAIN_TOURISM}")
    return meta


def _collection_names(meta: dict) -> tuple[str, str]:
    """返回 (切块集合名, 主体集合名)。"""
    gw = milvus_gateway
    return getattr(gw, meta["chunk_coll_attr"]), getattr(gw, meta["entity_coll_attr"])


def list_domain_files(domain: str) -> list[dict]:
    """
    列出指定领域已导入的全部文件（只读）。
    遍历该域两张集合的 file_title 聚合计数；单集合读取失败只跳过不阻断整体。
    返回按文件名排序：[{"file_title", "chunk_count", "entity_count"}, ...]
    """
    meta = _domain_meta(domain)
    chunk_coll, entity_coll = _collection_names(meta)
    client = milvus_gateway.client()

    chunk_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    for coll_name, counter in ((chunk_coll, chunk_counts), (entity_coll, entity_counts)):
        try:
            if not client.has_collection(collection_name=coll_name):
                logger.debug(f"集合[{coll_name}]不存在，跳过统计")
                continue
            iterator = client.query_iterator(
                collection_name=coll_name,
                output_fields=["file_title"],
                batch_size=2000,
            )
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    file_title = (row.get("file_title") or "").strip()
                    if file_title:
                        counter[file_title] += 1
        except Exception as e:  # noqa: BLE001 - 单集合失败降级，不阻断文件清单
            logger.warning(f"读取集合[{coll_name}]失败（跳过该集合）：{e}")

    titles = sorted(set(chunk_counts) | set(entity_counts))
    return [
        {
            "file_title": title,
            "chunk_count": chunk_counts.get(title, 0),
            "entity_count": entity_counts.get(title, 0),
        }
        for title in titles
    ]


def revoke_domain_file(domain: str, file_title: str) -> dict:
    """
    撤回指定领域中的一个已导入文件：按 file_title 精确删除该域两张集合中的全部行。
    返回每张集合的删除明细：{"chunks": {...}, "entities": {...}}
    明细含 deleted（delete 接口报告数）与 remaining（直接查询残留行数，删除复核以此为准）。
    集合不存在 / 删除异常均记录但不抛错（fail-open），由调用方汇总提示。
    """
    meta = _domain_meta(domain)
    chunk_coll, entity_coll = _collection_names(meta)
    client = milvus_gateway.client()

    # 防注入：复用项目统一的 Milvus 字符串转义
    escaped = escape_milvus_string(file_title)
    expr = f"file_title == {escaped}"

    detail: dict = {}
    for key, coll_name in (("chunks", chunk_coll), ("entities", entity_coll)):
        try:
            if not client.has_collection(collection_name=coll_name):
                detail[key] = {
                    "collection": coll_name,
                    "deleted": 0,
                    "remaining": 0,
                    "note": "集合不存在，无需删除",
                }
                continue
            del_result = client.delete(collection_name=coll_name, filter=expr) or {}
            deleted = int(del_result.get("delete_count", 0) or 0)
            # 复核残留：直接按真实字段查询（count(*) 聚合可能读到删除前的陈旧快照）
            remaining_rows = client.query(
                collection_name=coll_name,
                filter=expr,
                output_fields=["file_title"],
                limit=1,
            )
            remaining = len(remaining_rows)
            detail[key] = {
                "collection": coll_name,
                "deleted": deleted,
                "remaining": remaining,
            }
            logger.info(f"撤回文件[{file_title}]：集合[{coll_name}] 删除 {deleted} 条，残留 {remaining} 条")
        except Exception as e:  # noqa: BLE001 - 删除异常降级记录，交由调用方提示
            logger.warning(f"撤回文件[{file_title}]：集合[{coll_name}] 删除失败：{e}")
            detail[key] = {"collection": coll_name, "error": str(e)}
    return detail
