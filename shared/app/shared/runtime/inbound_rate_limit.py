# -*- coding: utf-8 -*-
"""
入站限流（拒绝式）—— 与出站阻塞式 `rate_limit_utils.apply_api_rate_limit` 严格区分。

对应 docs/网关鉴权与限流设计.md 三、限流设计：
- 计数后端：Redis `INCR` + `EXPIRE` 原子计数（多副本一致）；Redis 不可用时内存兜底（单机）。
- 语义：**拒绝式**（返回 allowed=False，由调用方回 429 + Retry-After）；
  绝不用 time.sleep 阻塞 worker。
- 维度：单用户 QPS / 单用户日配额 / 单 IP QPS。

密钥/地址复用 .env：RATE_LIMIT_REDIS_URL（缺省回落 REDIS_URL）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_int, env_str
from app.shared.runtime.logger import logger


@dataclass
class RateLimitResult:
    """限流判定结果。"""

    allowed: bool
    dimension: str = ""
    limit: int = 0
    retry_after_s: int = 0
    used: int = 0


# ---- 内存兜底计数（Redis 不可用时单机生效） ----
_MEM_LOCK = threading.Lock()
_MEM_COUNTERS: dict[str, tuple[int, float]] = {}  # key -> (count, expire_at_ts)

# ---- Redis 客户端（惰性探测一次） ----
_redis_client = None
_redis_ready: bool | None = None


def _get_redis():
    """惰性初始化 Redis 计数后端；失败返回 None（走内存兜底）。"""
    global _redis_client, _redis_ready
    if _redis_ready is not None:
        return _redis_client
    try:
        import redis  # noqa: PLC0415 - 延迟导入

        url = env_str("RATE_LIMIT_REDIS_URL", "") or env_str("REDIS_URL", "redis://127.0.0.1:6379/0")
        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        _redis_client = client
        _redis_ready = True
        logger.info("[入站限流] 计数后端=Redis")
    except Exception as exc:  # noqa: BLE001 - 不可用即降级
        _redis_client = None
        _redis_ready = False
        logger.warning(f"[入站限流] Redis 不可用，回退内存计数（单机）：{exc}")
    return _redis_client


def reset_backend_cache() -> None:
    """测试用：清空后端探测缓存。"""
    global _redis_client, _redis_ready
    _redis_client = None
    _redis_ready = None


def _incr(key: str, window_s: int) -> int:
    """窗口内自增并返回计数（Redis 原子；失败/不可用走内存）。"""
    client = _get_redis()
    if client is not None:
        try:
            pipe = client.pipeline()
            pipe.incr(key, 1)
            pipe.expire(key, window_s, nx=True)
            return int(pipe.execute()[0])
        except Exception as exc:  # noqa: BLE001 - 计数失败宁可放行（不影响主链路）
            logger.warning(f"[入站限流] Redis 计数失败，本次放行：{exc}")
            return 1
    now = time.time()
    with _MEM_LOCK:
        cnt, expire_at = _MEM_COUNTERS.get(key, (0, now + window_s))
        if now >= expire_at:
            cnt, expire_at = 0, now + window_s
        cnt += 1
        _MEM_COUNTERS[key] = (cnt, expire_at)
        return cnt


def check_rate_limit(*, user_id: str = "", ip: str = "", now_ts: int | None = None) -> RateLimitResult:
    """
    入站配额检查（拒绝式）。

    维度与默认值见 docs/网关鉴权与限流设计.md 3.1：
      单用户 QPS 2 / 单用户日配额 200 / 单 IP QPS 10。
    `RATE_LIMIT_ENABLE=false` 时直接放行。
    """
    if not env_bool("RATE_LIMIT_ENABLE", default=True):
        return RateLimitResult(True)
    ts = int(now_ts if now_ts is not None else time.time())

    if user_id:
        qps = max(1, env_int("RATE_LIMIT_USER_QPS", 2))
        used = _incr(f"rl:uqps:{user_id}:{ts}", 2)
        if used > qps:
            return RateLimitResult(False, "user_qps", qps, 1, used)

        daily = max(1, env_int("RATE_LIMIT_USER_DAILY", 200))
        day = time.strftime("%Y%m%d", time.localtime(ts))
        used_day = _incr(f"rl:uday:{user_id}:{day}", 86400)
        if used_day > daily:
            return RateLimitResult(False, "user_daily", daily, 86400 - (ts % 86400), used_day)

    if ip:
        ip_qps = max(1, env_int("RATE_LIMIT_IP_QPS", 10))
        used_ip = _incr(f"rl:ipqps:{ip}:{ts}", 2)
        if used_ip > ip_qps:
            return RateLimitResult(False, "ip_qps", ip_qps, 1, used_ip)

    return RateLimitResult(True)
