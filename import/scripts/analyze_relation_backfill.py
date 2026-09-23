# -*- coding: utf-8 -*-
"""
关系字段确定性回填 · 只读量测脚本（A1 覆盖度分析）。

**只读**：仅查询 Milvus，不写任何数据。
用途：在做回填/检索扩展之前，先量出「确定性词典匹配」能覆盖多少切片、值是否合理。

用法：
  cd D:\\xiangmu\\RAG_文旅_import
  & "D:\\xiangmu\\RAG_shared_infra\\.venv\\Scripts\\python.exe" scripts\\analyze_relation_backfill.py

输出：控制台摘要 + output/relation_backfill_analysis.txt（含全部逐条明细）
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.infra.vectorstore import milvus_gateway  # noqa: E402
from app.rag.tourism_import.attraction_mention_service import (  # noqa: E402
    SOURCE_CONTENT_TYPES,
    build_attraction_lexicon,
    harvest_attraction_terms,
    plan_relation_backfill,
)

OUT_PATH = ROOT / "output" / "relation_backfill_analysis.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT = open(OUT_PATH, "w", encoding="utf-8", buffering=1)


def log(*args) -> None:
    line = " ".join(str(a) for a in args)
    OUT.write(line + "\n")
    print(line)


FIELDS = ["chunk_id", "file_title", "item_name", "content_type", "region",
          "title", "parent_title", "content", "extra_meta"]


def fetch_all() -> list[dict]:
    client = milvus_gateway.client()
    coll = milvus_gateway.tourism_chunks_collection
    rows: list[dict] = []
    it = client.query_iterator(collection_name=coll, filter="",
                               output_fields=FIELDS, batch_size=500)
    while True:
        batch = it.next()
        if not batch:
            break
        rows.extend(batch)
    it.close()
    return rows


def main() -> int:
    rows = fetch_all()
    log(f"chunks = {len(rows)}")

    # ---- 1) 建词典：按 file_title 聚合景点类文档正文 ----
    # 注意：只用 content（不含 title）—— title 与 content 首行重复，
    # 会把「只出现一次的小标题」误判为出现 2 次，导致修饰性小标题入典。
    by_file: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if (r.get("content_type") or "") in SOURCE_CONTENT_TYPES:
            by_file[r.get("file_title") or "(无标题)"].append(r.get("content") or "")

    sources = [(t, "\n".join(v)) for t, v in by_file.items()]
    extra_terms = harvest_attraction_terms(rows)
    lexicon = build_attraction_lexicon(sources, extra_terms=extra_terms)
    log(f"\n词典来源文件数 = {len(sources)}（类型 {SOURCE_CONTENT_TYPES}）")
    for t in sorted(by_file):
        log(f"   - {t}（{len(by_file[t])} chunk）")
    log(f"\n补充词表（库内景点级线索）= {len(extra_terms)}："
        + "、".join(sorted(extra_terms)))
    log(f"\n词典规模 = {len(lexicon)}")
    log("词典内容：" + "、".join(sorted(lexicon, key=lambda x: (-len(x), x))))

    # ---- 2) 回填计划（纯函数，不写库）----
    plan = plan_relation_backfill(rows, lexicon)

    by_type = Counter(r.get("content_type") or "(空)" for r in rows)
    planned = Counter(p["content_type"] for p in plan)
    filled_already = Counter()
    for r in rows:
        em = r.get("extra_meta") or {}
        if isinstance(em, dict):
            for k in ("nearby_attractions", "related_attractions", "attractions"):
                if em.get(k):
                    filled_already[(r.get("content_type") or "(空)", k)] += 1

    log("\n=== 覆盖率（按 content_type）===")
    log("  content_type | 总 chunk | 已有关系值 | 本次可回填 | 命中率")
    for ct, n in by_type.most_common():
        p = planned.get(ct, 0)
        plain = sum(c for (c, _k), v in filled_already.items() if c == ct for c in [v])
        rate = f"{p / n * 100:.1f}%" if n else "-"
        log(f"  {ct} | {n} | {plain} | {p} | {rate}")

    total_rel_types = sum(by_type[c] for c in
                          ("酒店信息", "文化知识介绍", "游记攻略", "线路推荐"))
    log(f"\n承载关系字段的类型合计 chunk = {total_rel_types}")
    log(f"本次可回填 chunk = {len(plan)}"
        f"（{len(plan) / total_rel_types * 100:.1f}% of 承载类型）" if total_rel_types else "")

    # ---- 3) 逐条明细 + 值分布 ----
    log("\n=== 逐条回填明细 ===")
    for p in sorted(plan, key=lambda x: (x["content_type"], x["file_title"] or "")):
        log(f"  [{p['content_type']}] {p['file_title']} | {p['key']}={json.dumps(p['values'], ensure_ascii=False)}")

    val_counter = Counter()
    for p in plan:
        for v in p["values"]:
            val_counter[v] += 1
    log("\n=== 命中值频次（Top 40）===")
    for v, n in val_counter.most_common(40):
        log(f"  {v}: {n}")

    # ---- 4) 未被覆盖的切片（供判断局限）----
    log("\n=== 承载类型中未命中的切片样例（前 20）===")
    planned_ids = {p["chunk_id"] for p in plan}
    miss = [r for r in rows
            if (r.get("content_type") or "") in
            ("酒店信息", "文化知识介绍", "游记攻略", "线路推荐")
            and r.get("chunk_id") not in planned_ids]
    for r in miss[:20]:
        log(f"  [{r.get('content_type')}] {r.get('file_title')} | title={r.get('title')} | {(r.get('content') or '')[:60]!r}")
    log(f"  （未命中合计 {len(miss)}）")

    log("\nDONE（本次未写入任何数据）")
    OUT.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
