"""
LLM 调用缓存模块：对确定性高、重复率高的模型调用做两级缓存。

两级设计：
- L1 精确缓存：输入完全相同（md5 键）则直接命中，跳过模型调用（原逻辑平移）。
- L2 语义缓存：输入语义相似（BGE-M3 向量 cosine ≥ 阈值）也视为命中，
  用于「同义改写 / 近似问题」省掉重复模型往返，是降本的主要杠杆。

统一缓存层：
- 所有缓存读写经 `CacheBackend` 抽象，两种实现：
  - `MemoryBackend`（默认，进程内，零外部依赖）；
  - `RedisBackend`（`CACHE_BACKEND=redis`，多进程 / 重启后缓存全局共享）。
- 语义缓存同理：`MemorySemanticStore`（默认） / `RedisSemanticStore`（redis 后端）。
- Redis 后端下 `clear_cache()` 可跨进程失效缓存，是多进程 / 多用户场景的降本关键。

设计取舍（保持原约束）：
- 答案生成【不缓存】：同一问题在不同知识库状态下应有不同回答，且带随机性。
- 时间敏感类（如实时数值）通过 `time_sensitive=True` 跳过语义缓存，只走精确，避免跨时间点误命中。
- 语义缓存绑定 `CACHE_TTL_SECONDS`（知识库在此期间更新时结果可能滞后，导入失效复用 clear_cache）。
- 语义编码复用本地 BGE-M3 单例，不外发、近乎零额外成本。

线程安全：FastAPI 后台任务跑在线程池中，并发查询会同时读写缓存，故 Memory 后端用互斥锁保护；
Redis 后端由 Redis 自身保证并发安全，无需进程内锁。
"""
import hashlib
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable

from app.shared.config.common import env_bool, env_int
from app.shared.model.embedding_utils import generate_embeddings
from app.shared.runtime.logger import logger

# ---- 开关与参数 ----
CACHE_ENABLE: bool = env_bool("LLM_CACHE_ENABLE", default=True)
CACHE_MAXSIZE: int = env_int("LLM_CACHE_MAXSIZE", default=256)
CACHE_TTL_SECONDS: int = env_int("LLM_CACHE_TTL_SECONDS", default=600)
# 语义缓存（L2）
SEMANTIC_CACHE_ENABLE: bool = env_bool("SEMANTIC_CACHE_ENABLE", default=True)
SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_MAXSIZE: int = env_int("SEMANTIC_CACHE_MAXSIZE", default=256)
# 后端选择：memory（默认，零外部依赖）/ redis（多进程共享）
CACHE_BACKEND: str = os.getenv("CACHE_BACKEND", "memory").lower()
# Redis 连接串（仅 CACHE_BACKEND=redis 时使用）
REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# Redis 键前缀（与其它业务键隔离，便于统一清理）
_REDIS_NS_PREFIX = "llmcache:ns:"
_REDIS_SEM_PREFIX = "llmcache:sem:"
_REDIS_SEM_SEQ_PREFIX = "llmcache:semseq:"


# ---- 后端抽象 ----
class CacheBackend(ABC):
    """统一的缓存后端接口：进程内 / Redis 共用同一套读写语义。"""

    @abstractmethod
    def get(self, namespace: str, key: str) -> tuple[bool, Any]:
        ...

    @abstractmethod
    def set(self, namespace: str, key: str, value: Any, ttl: int) -> None:
        ...

    @abstractmethod
    def clear(self, namespace: str | None) -> None:
        ...


class MemoryBackend(CacheBackend):
    """进程内后端：OrderedDict + LRU 淘汰 + TTL 懒清理。"""

    def __init__(self) -> None:
        self._store: dict[str, "OrderedDict[str, tuple[float, Any]]"] = {}
        self._lock = threading.Lock()

    def get(self, namespace: str, key: str) -> tuple[bool, Any]:
        with self._lock:
            bucket = self._store.get(namespace)
            if not bucket:
                return False, None
            entry = bucket.get(key)
            if entry is None:
                return False, None
            expire_ts, value = entry
            if expire_ts < time.time():
                bucket.pop(key, None)
                return False, None
            bucket.move_to_end(key)
            return True, value

    def set(self, namespace: str, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            bucket = self._store.setdefault(namespace, OrderedDict())
            bucket[key] = (time.time() + ttl, value)
            bucket.move_to_end(key)
            while len(bucket) > CACHE_MAXSIZE:
                bucket.popitem(last=False)

    def clear(self, namespace: str | None) -> None:
        with self._lock:
            if namespace:
                self._store.pop(namespace, None)
            else:
                self._store.clear()


class RedisBackend(CacheBackend):
    """
    Redis 后端：多进程 / 重启后缓存全局共享（T9）。

    存储结构：每个 namespace 一个 Hash
        HSET llmcache:ns:<namespace> <key> <json({"e": expire_ts, "v": value})>

    - 用显式 expire_ts 而非 Redis 原生 TTL，保证与 MemoryBackend 语义一致
      （同一 namespace 内各键独立过期，不受整表 TTL 影响）。
    - 值经 JSON 序列化（str / dict / list 均可）；不可序列化则跳过写入（降级为不缓存）。
    - 依赖注入 `client`，便于测试（可传 fakeredis）。
    """

    def __init__(self, client: Any, maxsize: int = CACHE_MAXSIZE) -> None:
        self._r = client
        self._maxsize = maxsize

    @staticmethod
    def _ns_key(namespace: str) -> str:
        return f"{_REDIS_NS_PREFIX}{namespace}"

    def get(self, namespace: str, key: str) -> tuple[bool, Any]:
        raw = self._r.hget(self._ns_key(namespace), key)
        if raw is None:
            return False, None
        try:
            entry = json.loads(raw)
            expire_ts = float(entry.get("e", 0))
        except (ValueError, TypeError):
            return False, None
        if expire_ts < time.time():
            self._r.hdel(self._ns_key(namespace), key)
            return False, None
        return True, entry.get("v")

    def set(self, namespace: str, key: str, value: Any, ttl: int) -> None:
        try:
            payload = json.dumps({"e": time.time() + ttl, "v": value}, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"[LLM缓存] 值不可 JSON 序列化，跳过缓存写入：{e}")
            return
        nk = self._ns_key(namespace)
        self._r.hset(nk, key, payload)
        # 近似上限：field 数超限时给整表兜底 TTL，防无限增长
        # （Redis Hash 无 LRU 顺序，精确裁剪语义由 TTL 懒清理 + 上限兜底共同保证）
        try:
            if self._r.hlen(nk) > self._maxsize:
                self._r.expire(nk, ttl)
        except Exception:
            pass

    def clear(self, namespace: str | None) -> None:
        if namespace:
            self._r.delete(self._ns_key(namespace))
        else:
            for k in self._r.scan_iter(match=f"{_REDIS_NS_PREFIX}*"):
                self._r.delete(k)


_redis_client: Any = None
# P1-3：Redis 失败回落标记。一旦探测失败，进程内记忆，避免每次调用都重连探测。
_redis_unavailable: bool = False


def _get_redis_client() -> Any:
    """
    Redis 客户端单例（延迟建连，仅 CACHE_BACKEND=redis 时调用）。

    P1-3 失败降级：连不上或依赖缺失时不再 fail-fast 抛错（避免单点故障拖垮进程），
    而是返回 None 并打 warning，由 get_backend / get_semantic_backend 回落 MemoryBackend。
    """
    global _redis_client, _redis_unavailable
    if _redis_unavailable:
        return None
    if _redis_client is None:
        try:
            import redis  # 延迟导入：未启用 Redis 时无需该依赖
        except ImportError as e:  # pragma: no cover
            logger.warning(
                f"[LLM缓存] 未安装 redis 依赖，回落 memory 后端（CACHE_BACKEND=redis 但未装 redis 包）：{e}"
            )
            _redis_unavailable = True
            return None
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            client.ping()
        except Exception as e:
            logger.warning(
                f"[LLM缓存] Redis 连接失败（{REDIS_URL}），回落 memory 后端：{e}。"
                f"请检查 Redis 是否启动，或改回 CACHE_BACKEND=memory"
            )
            _redis_unavailable = True
            return None
        _redis_client = client
        logger.info(f"[LLM缓存] 已启用 Redis 后端：{REDIS_URL}")
    return _redis_client


def get_backend() -> CacheBackend:
    """
    缓存后端工厂。

    - `memory`（默认）：进程内 MemoryBackend，零外部依赖；
    - `redis`：RedisBackend，多进程 / 重启后共享（需 redis 依赖 + 可达 Redis 服务）。
      P1-3：Redis 不可达时回落 MemoryBackend 并打 warning，不 fail-fast。
    """
    if CACHE_BACKEND == "redis":
        client = _get_redis_client()
        if client is None:
            return MemoryBackend()
        return RedisBackend(client)
    return MemoryBackend()


_backend = get_backend()
_stats: dict[str, int] = {"hit": 0, "miss": 0, "semantic_hit": 0}


# ---- 语义缓存（L2） ----
def _cosine(a: list[float], b: list[float]) -> float:
    """BGE-M3 dense 已 L2 归一化，cosine = 点积。"""
    if len(a) != len(b) or not a:
        return -1.0
    return sum(x * y for x, y in zip(a, b))


def _encode(text: str) -> list[float]:
    """本地 BGE-M3 编码，返回 dense 向量（已归一化）。"""
    result = generate_embeddings([text])
    return result["dense"][0]


class MemorySemanticStore:
    """按 namespace 存 (vector, expire_ts, value)，供语义相似度命中（进程内）。"""

    def __init__(self) -> None:
        self._buckets: dict[str, "OrderedDict[str, tuple[list[float], float, Any]]"] = {}
        self._lock = threading.Lock()
        self._seq = 0

    def get_similar(self, namespace: str, vector: list[float], threshold: float) -> tuple[bool, Any]:
        with self._lock:
            bucket = self._buckets.get(namespace)
            if not bucket:
                return False, None
            now = time.time()
            best_sim = -1.0
            best_val: Any = None
            expired: list[str] = []
            for k, (vec, exp_ts, val) in bucket.items():
                if exp_ts < now:
                    expired.append(k)
                    continue
                sim = _cosine(vector, vec)
                if sim > best_sim:
                    best_sim = sim
                    best_val = val
            for k in expired:
                bucket.pop(k, None)
            if best_val is not None and best_sim >= threshold:
                return True, best_val
            return False, None

    def put(self, namespace: str, vector: list[float], value: Any) -> None:
        with self._lock:
            bucket = self._buckets.setdefault(namespace, OrderedDict())
            self._seq += 1
            key = str(self._seq)
            bucket[key] = (vector, time.time() + CACHE_TTL_SECONDS, value)
            bucket.move_to_end(key)
            while len(bucket) > SEMANTIC_CACHE_MAXSIZE:
                bucket.popitem(last=False)

    def clear(self, namespace: str | None) -> None:
        with self._lock:
            if namespace:
                self._buckets.pop(namespace, None)
            else:
                self._buckets.clear()


class RedisSemanticStore:
    """
    Redis 语义缓存：多进程共享（T9）。

    存储结构：每个 namespace 一个 Hash
        HSET llmcache:sem:<namespace> <seq> <json({"vec": [...], "e": expire_ts, "v": value})>

    - field 用 namespace 内自增序号（INCR）保证唯一、可按序裁剪。
    - get_similar 拉取整表在进程内算 cosine（语义桶上限 256，代价可接受）。
    """

    def __init__(self, client: Any, maxsize: int = SEMANTIC_CACHE_MAXSIZE) -> None:
        self._r = client
        self._maxsize = maxsize

    @staticmethod
    def _key(namespace: str) -> str:
        return f"{_REDIS_SEM_PREFIX}{namespace}"

    def get_similar(self, namespace: str, vector: list[float], threshold: float) -> tuple[bool, Any]:
        raw_map = self._r.hgetall(self._key(namespace))
        if not raw_map:
            return False, None
        now = time.time()
        best_sim = -1.0
        best_val: Any = None
        expired: list[str] = []
        for field, raw in raw_map.items():
            try:
                entry = json.loads(raw)
                exp_ts = float(entry.get("e", 0))
            except (ValueError, TypeError):
                expired.append(field)
                continue
            if exp_ts < now:
                expired.append(field)
                continue
            sim = _cosine(vector, entry.get("vec") or [])
            if sim > best_sim:
                best_sim = sim
                best_val = entry.get("v")
        if expired:
            self._r.hdel(self._key(namespace), *expired)
        if best_val is not None and best_sim >= threshold:
            return True, best_val
        return False, None

    def put(self, namespace: str, vector: list[float], value: Any) -> None:
        try:
            payload = json.dumps(
                {"vec": list(vector), "e": time.time() + CACHE_TTL_SECONDS, "v": value},
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"[LLM缓存] 语义值不可 JSON 序列化，跳过语义写入：{e}")
            return
        seq = self._r.incr(f"{_REDIS_SEM_SEQ_PREFIX}{namespace}")
        k = self._key(namespace)
        self._r.hset(k, str(seq), payload)
        try:
            if self._r.hlen(k) > self._maxsize:
                fields = sorted(self._r.hkeys(k), key=lambda x: int(x))
                for f in fields[: len(fields) - self._maxsize]:
                    self._r.hdel(k, f)
        except Exception:
            pass

    def clear(self, namespace: str | None) -> None:
        if namespace:
            self._r.delete(self._key(namespace))
            self._r.delete(f"{_REDIS_SEM_SEQ_PREFIX}{namespace}")
        else:
            for k in self._r.scan_iter(match=f"{_REDIS_SEM_PREFIX}*"):
                self._r.delete(k)
            for k in self._r.scan_iter(match=f"{_REDIS_SEM_SEQ_PREFIX}*"):
                self._r.delete(k)


def get_semantic_backend() -> MemorySemanticStore | RedisSemanticStore:
    """语义缓存后端工厂：与精确缓存同源（memory / redis）。

    P1-3：Redis 不可达时回落 MemorySemanticStore 并打 warning，不 fail-fast。
    """
    if CACHE_BACKEND == "redis":
        client = _get_redis_client()
        if client is None:
            return MemorySemanticStore()
        return RedisSemanticStore(client)
    return MemorySemanticStore()


_semantic = get_semantic_backend()


# ---- 对外 API ----
def build_cache_key(*parts: str) -> str:
    """根据若干文本片段生成稳定缓存键。"""
    raw = "||".join(part for part in parts if part is not None)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cached_invoke(
    *,
    namespace: str,
    cache_parts: tuple[str, ...],
    producer: Callable[[], Any],
    cache_label: str = "",
    semantic: bool = False,
    semantic_text: str | None = None,
    time_sensitive: bool = False,
) -> Any:
    """
    带缓存的模型调用包装器：先查缓存，未命中则执行 producer 并回写。

    Args:
        namespace: 命名空间（同时承担域隔离，文旅/闲聊各自独立，防误命中）。
        cache_parts: 构成精确缓存键的文本片段。
        producer: 未命中时的真实调用（通常是模型链 invoke）。
        cache_label: 日志用的可读标识。
        semantic: 是否启用 L2 语义缓存（仅对稳定映射类调用有意义）。
        semantic_text: 用于语义编码的文本，通常即用户问题本身（不含历史），默认取 cache_parts[0]。
        time_sensitive: 时间敏感类（实时数值等）设 True，跳过语义缓存只走精确。

    Returns:
        Any: 缓存值或 producer 的执行结果。
    """
    if not CACHE_ENABLE:
        return producer()

    # L1 精确命中
    key = build_cache_key(*cache_parts)
    hit, value = _backend.get(namespace, key)
    if hit:
        _stats["hit"] += 1
        logger.info(f"[LLM缓存] 精确命中 {namespace}{'/' + cache_label if cache_label else ''}，跳过模型调用")
        return value

    # 语义编码文本：优先用显式传入的 semantic_text，否则用 cache_parts 拼接，
    # 使语义空间与精确键一致——避免仅按 query 编码时，跨轮不同 history 的指代被误命中
    # （如「那里天气怎么样」在不同对话轮次可能指代不同地点）。
    enc_text = semantic_text if semantic_text is not None else "||".join(
        str(p) for p in cache_parts if p is not None
    )

    # L2 语义命中
    if semantic and SEMANTIC_CACHE_ENABLE and not time_sensitive:
        try:
            vec = _encode(enc_text)
            sh, sval = _semantic.get_similar(namespace, vec, SEMANTIC_CACHE_THRESHOLD)
            if sh:
                _stats["semantic_hit"] += 1
                logger.info(f"[LLM缓存] 语义命中 {namespace}{'/' + cache_label if cache_label else ''}，跳过模型调用")
                return sval
        except Exception as e:
            logger.warning(f"[LLM缓存] 语义编码失败，降级为直连：{e}")

    _stats["miss"] += 1
    result = producer()
    _backend.set(namespace, key, result, CACHE_TTL_SECONDS)
    if semantic and SEMANTIC_CACHE_ENABLE and not time_sensitive:
        try:
            vec = _encode(enc_text)
            _semantic.put(namespace, vec, result)
        except Exception as e:
            logger.warning(f"[LLM缓存] 语义写入失败，仅保留精确缓存：{e}")
    return result


def cache_stats() -> dict[str, Any]:
    """返回缓存命中统计快照（精确 + 语义）。"""
    hit, miss, sh = _stats["hit"], _stats["miss"], _stats["semantic_hit"]
    total = hit + miss
    return {
        "hit": hit,
        "miss": miss,
        "semantic_hit": sh,
        "hit_rate_percent": round(hit / total * 100, 1) if total else 0.0,
        "semantic_hit_rate_percent": round(sh / total * 100, 1) if total else 0.0,
        "backend": CACHE_BACKEND,
    }


def clear_cache(namespace: str | None = None) -> None:
    """
    清空缓存：指定命名空间则只清该空间，否则全清。
    知识库导入完成后可调用，避免改写/抽取/语义结果滞后于新数据。
    Redis 后端下本操作跨进程生效（多进程共享场景关键）。
    """
    _backend.clear(namespace)
    _semantic.clear(namespace)
    if namespace is None:
        _stats["hit"] = 0
        _stats["miss"] = 0
        _stats["semantic_hit"] = 0
    logger.info(f"LLM缓存已清空：namespace={namespace or 'ALL'}")
