# -*- coding: utf-8 -*-
"""
B2 存量数据 region 补标脚本（2026-09-11 主人拍板方案二配套）。

背景：存量 tourism_chunks / tourism_entity 的 region 字段填充率仅 55.6%，
B2 涉及的"成都住宿推荐/杭州交通指南/三亚住宿推荐"等文档 region 全为空，
检索侧城市过滤（RETRIEVAL_REGION_FILTER）依赖该字段覆盖率。

用法（venv python 运行，需 Milvus 在线）：
    python scripts/backfill_region.py            # 干跑：只打印将回填的内容，不写库
    python scripts/backfill_region.py --apply    # 实写：查询->回填 region->upsert

规则：
- 只处理 region 为空的记录；
- 用共享层城市名单（app/shared/utils/city_utils.extract_cities）从
  file_title + item_name + title + content(前300字) 做确定性子串匹配；
- 多城市以逗号连接；未命中城市的不动（保留空 region，属全域资料）；
- upsert 带原主键（chunk_id / pk）与原向量，仅改 region 字段。
"""
import argparse
import os
import sys

sys.path.insert(0, r"D:\xiangmu\RAG_shared_infra")

from dotenv import load_dotenv

load_dotenv(r"D:\xiangmu\RAG_文旅_retrieval\.env")

from pymilvus import MilvusClient

from app.shared.utils.city_utils import extract_cities
from app.shared.runtime.logger import logger


def get_client() -> MilvusClient:
    url = os.getenv("MILVUS_URL", "")
    if ":" not in url.split("//")[-1]:
        url = url.rstrip("/") + ":19530"
    logger.info(f"连接 Milvus: {url}")
    return MilvusClient(uri=url)


def guess_region(row: dict) -> str:
    """从一行数据里确定性抽取城市，返回应回填的 region（空串=不改）。"""
    text = " ".join([
        str(row.get("file_title") or ""),
        str(row.get("item_name") or ""),
        str(row.get("title") or ""),
        str(row.get("content") or "")[:300],
    ])
    cities = extract_cities(text)
    return ",".join(cities) if cities else ""


def backfill(client: MilvusClient, collection: str, pk_field: str, apply: bool) -> None:
    # upsert 要求完整行（schema 显式字段均非 nullable），必须读全所有字段，
    # 否则 DataNotMatchException: Insert missed an field ...
    rows = client.query(
        collection_name=collection,
        filter='region == ""',
        output_fields=["*", "dense_vector", "sparse_vector"],
        limit=2000,
    )
    logger.info(f"[{collection}] region 为空的记录：{len(rows)} 条")
    updated = 0
    for row in rows:
        region = guess_region(row)
        if not region:
            continue
        updated += 1
        logger.info(f"  file_title={row.get('file_title')!r} item={row.get('item_name')!r} -> region={region!r}")
        if apply:
            row["region"] = region
            try:
                client.upsert(collection_name=collection, data=[row])
            except Exception as e:  # noqa: BLE001
                logger.error(f"  upsert 失败（跳过该条）：{e}")
                updated -= 1
    logger.info(f"[{collection}] {'已回填' if apply else '干跑，可回填'} {updated}/{len(rows)} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 存量 region 补标")
    parser.add_argument("--apply", action="store_true", help="实写 Milvus（默认干跑）")
    args = parser.parse_args()
    client = get_client()
    backfill(client, "tourism_chunks", "chunk_id", args.apply)
    backfill(client, "tourism_entity", "pk", args.apply)
    logger.info(f"补标结束（mode={'APPLY' if args.apply else 'DRY-RUN'}）")


if __name__ == "__main__":
    main()
