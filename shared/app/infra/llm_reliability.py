# -*- coding: utf-8 -*-
"""E·Harness 可靠性加固：LLM 调用的统一超时 / 熔断 / 降级。

设计原则（对齐《企业 Agent 落地十二大工程框架》Harness 维度）：
- 不盲目换便宜模型，靠「路由 + 降级 + 超时 + 熔断」保可用性与成本。
- 熔断是防御性兜底：某档模型持续失败时快速失败并降级，避免雪崩与无谓重试。
- 全部异常安全：降级/超时/熔断都可返回 fallback，绝不吞掉主链路错误导致静默失败。

用法：
    from app.infra.llm_reliability import reliable_llm_call
    out = reliable_llm_call(llm_provider.chat_by_tier(tier, json_mode=True) | parser,
                            messages, tier=tier, timeout=60, fallback=DEFAULT_ANSWER)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from app.shared.runtime.logger import logger

# 进程内共享的熔断器注册表（按 name 维度隔离，例如按模型档位）
_BREAKERS: dict[str, "CircuitBreaker"] = {}
_BREAKERS_LOCK = threading.Lock()


class CircuitBreaker:
    """简易熔断器：连续失败超阈则打开，冷却后半开探测，成功则闭合。"""

    def __init__(self, name: str, failure_threshold: int = 5, cooldown_seconds: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._failures < self.failure_threshold:
                return True
            # 打开中：检查冷却是否到期
            if time.time() - self._opened_at >= self.cooldown_seconds:
                # 进入半开：重置计数允许一次探测
                self._failures = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()
                logger.warning(f"[熔断] {self.name} 触发打开，{self.cooldown_seconds}s 内快速失败")


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    with _BREAKERS_LOCK:
        if name not in _BREAKERS:
            _BREAKERS[name] = CircuitBreaker(name, **kwargs)
        return _BREAKERS[name]


def call_with_timeout(fn: Callable[[], Any], timeout: float, fallback: Any = None) -> Any:
    """在独立线程中执行 fn，超时则返回 fallback（或抛 TimeoutError）。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            logger.warning(f"[超时] 调用超时（{timeout}s），返回降级结果")
            if fallback is not None:
                return fallback
            raise
        except Exception:
            if fallback is not None:
                return fallback
            raise


def reliable_llm_call(
    chain,
    messages,
    *,
    tier: str = "standard",
    timeout: float = 60.0,
    fallback: Any = None,
    breaker_name: Optional[str] = None,
) -> Any:
    """对 LLM 链做「熔断器 + 超时 + 降级」包装。

    - 熔断打开时直接返回 fallback（不发起调用）。
    - 调用超时（timeout）或抛异常时：记录失败、返回 fallback。
    - 成功则记录成功。
    """
    name = breaker_name or f"llm-{tier}"
    breaker = get_breaker(name)
    if not breaker.allow():
        logger.warning(f"[熔断] {name} 处于打开态，直接降级")
        if fallback is not None:
            return fallback
        raise RuntimeError(f"circuit breaker open: {name}")

    def _run():
        return chain.invoke(messages)

    try:
        result = call_with_timeout(_run, timeout, fallback)
    except Exception as e:  # noqa: BLE001
        breaker.record_failure()
        if fallback is not None:
            logger.warning(f"[LLM可靠性] 调用失败，降级：{e!r}")
            return fallback
        raise
    # 若超时触发了 fallback（call_with_timeout 返回 fallback），不计入成功
    if result is fallback and fallback is not None:
        breaker.record_failure()
    else:
        breaker.record_success()
    return result
