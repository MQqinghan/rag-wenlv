"""
Redis 缓存后端（T9）单测：零外部服务（用 fakeredis 模拟），可秒级离线运行。

覆盖：
    1. RedisBackend 精确缓存 set/get 命中、未命中
    2. TTL 过期（懒清理）
    3. JSON 值类型往返（str / dict / list / 中文）
    4. 不可 JSON 序列化的值：跳过写入、不抛异常
    5. clear(namespace) 只清该命名空间；clear(None) 全清
    6. 多进程共享：两个 client（共享同一 fakeredis server）互相可见
    7. RedisSemanticStore：相似命中 / 不命中 / 过期 / 跨实例共享

运行：
    ./.venv/Scripts/python.exe test/test_redis_cache.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 本用例纯离线（fakeredis 自建 server）：必须在 import llm_cache 之前强制内存后端，
# 否则 .env 里 CACHE_BACKEND=redis 时模块级 get_backend() 会去 ping 真实 Redis
# （无服务即导入失败）。load_dotenv(override=False) 不会覆盖已存在的同名环境变量。
os.environ["CACHE_BACKEND"] = "memory"

import fakeredis  # noqa: E402

from app.shared.runtime.llm_cache import (  # noqa: E402
    CACHE_TTL_SECONDS,
    RedisBackend,
    RedisSemanticStore,
)


def _new_client(server: fakeredis.FakeServer):
    """与真实环境一致：decode_responses=True。"""
    return fakeredis.FakeStrictRedis(server=server, decode_responses=True)


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, info: str = "") -> None:
        results.append((name, cond, info))

    server = fakeredis.FakeServer()
    b1 = RedisBackend(_new_client(server))
    b2 = RedisBackend(_new_client(server))  # 模拟另一进程（同 server）

    # 1. 命中 / 未命中
    hit, _ = b1.get("ns_a", "missing")
    check("未命中返回 (False, None)", hit is False)
    b1.set("ns_a", "k1", "v1", CACHE_TTL_SECONDS)
    hit, val = b1.get("ns_a", "k1")
    check("set 后可命中且值一致", hit is True and val == "v1")

    # 2. TTL 过期（ttl=-1 → 立即过期）
    b1.set("ns_a", "k_exp", "v_exp", -1)
    hit, _ = b1.get("ns_a", "k_exp")
    check("TTL 到期后未命中（懒清理）", hit is False)

    # 3. JSON 值类型往返
    b1.set("ns_json", "d", {"a": 1, "b": [1, 2, 3], "c": "中文"}, CACHE_TTL_SECONDS)
    _, vd = b1.get("ns_json", "d")
    check("dict 值 JSON 往返", vd == {"a": 1, "b": [1, 2, 3], "c": "中文"})
    b1.set("ns_json", "l", [1, "二", 3.5], CACHE_TTL_SECONDS)
    _, vl = b1.get("ns_json", "l")
    check("list 值 JSON 往返", vl == [1, "二", 3.5])

    # 4. 不可序列化值：跳过写入且不抛
    try:
        b1.set("ns_bad", "obj", object(), CACHE_TTL_SECONDS)
        hit, _ = b1.get("ns_bad", "obj")
        check("不可序列化值跳过写入（不抛）", hit is False)
    except Exception as e:  # pragma: no cover
        check("不可序列化值跳过写入（不抛）", False, f"抛异常: {e}")

    # 5. clear：namespace 级 / 全清
    b1.set("ns_a", "k2", "v2", CACHE_TTL_SECONDS)
    b1.set("ns_b", "k3", "v3", CACHE_TTL_SECONDS)
    b1.clear("ns_a")
    ha, _ = b1.get("ns_a", "k2")
    hb, _ = b1.get("ns_b", "k3")
    check("clear(namespace) 只清该命名空间", ha is False and hb is True)
    b1.clear(None)
    hb2, _ = b1.get("ns_b", "k3")
    check("clear(None) 全清", hb2 is False)

    # 6. 多进程共享（b2 是另一个 client，同 server）
    b1.set("ns_share", "sk", {"pid": "A"}, CACHE_TTL_SECONDS)
    hit, v = b2.get("ns_share", "sk")
    check("跨实例（多进程）共享命中", hit is True and v == {"pid": "A"})

    # 7. 语义缓存
    s1 = RedisSemanticStore(_new_client(server))
    s2 = RedisSemanticStore(_new_client(server))
    vec_a = [1.0, 0.0, 0.0, 0.0]
    vec_same = [1.0, 0.0, 0.0, 0.0]
    vec_far = [0.0, 1.0, 0.0, 0.0]
    hit, _ = s1.get_similar("sem_ns", vec_a, 0.92)
    check("语义空桶未命中", hit is False)
    s1.put("sem_ns", vec_a, {"answer": "ha"})
    hit, v = s1.get_similar("sem_ns", vec_same, 0.92)
    check("语义相似命中（cosine=1）", hit is True and v == {"answer": "ha"})
    hit, _ = s1.get_similar("sem_ns", vec_far, 0.92)
    check("语义不相似未命中（cosine=0）", hit is False)
    # 跨实例共享
    hit, v = s2.get_similar("sem_ns", vec_same, 0.92)
    check("语义跨实例（多进程）共享命中", hit is True and v == {"answer": "ha"})
    # 语义清空
    s1.clear("sem_ns")
    hit, _ = s1.get_similar("sem_ns", vec_same, 0.92)
    check("语义 clear(namespace) 生效", hit is False)

    # 汇总
    print("== Redis 缓存后端单测（fakeredis）==")
    passed = 0
    for name, ok, info in results:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{info}]" if info and not ok else ""))
        passed += 1 if ok else 0
    print(f"\n{passed}/{len(results)} 通过")
    if passed != len(results):
        raise SystemExit(1)
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
