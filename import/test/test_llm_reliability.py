# -*- coding: utf-8 -*-
"""E·Harness 可靠性加固单测（超时 / 熔断 / 降级）—— 离线、零外部请求。

覆盖：
  1. CircuitBreaker：初始放行 / 达阈值打开 / 冷却后半开 / 成功复位
  2. get_breaker：同名复用同一实例
  3. call_with_timeout：正常返回 / 异常降级 / 超时降级
  4. reliable_llm_call：成功记成功 / 异常降级 / 熔断打开直接降级不调用 /
     连续失败打开熔断 / 无 fallback 时熔断打开抛错
  5. 观察项：超时的真实wall-clock行为（当前实现为等待工作线程结束）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infra.llm_reliability import (
    CircuitBreaker,
    call_with_timeout,
    get_breaker,
    reliable_llm_call,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


class FakeChain:
    """模拟 langchain Runnable：invoke(messages) -> value 或抛异常。"""

    def __init__(self, ret=None, exc=None, delay: float = 0.0):
        self.ret = ret
        self.exc = exc
        self.delay = delay
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.ret


# ============================================================
# 1. CircuitBreaker
# ============================================================
print("== 1. CircuitBreaker ==")
cb = CircuitBreaker("t1", failure_threshold=3, cooldown_seconds=60)
check("初始放行", cb.allow() is True)
cb.record_failure()
cb.record_failure()
check("未达阈值仍放行", cb.allow() is True)
cb.record_failure()
check("达阈值后打开（拒绝）", cb.allow() is False)

cb2 = CircuitBreaker("t2", failure_threshold=1, cooldown_seconds=0.05)
cb2.record_failure()
check("阈值1 立即打开", cb2.allow() is False)
time.sleep(0.08)
check("冷却后半开放行", cb2.allow() is True)
check("半开已重置计数（继续放行）", cb2.allow() is True)

cb3 = CircuitBreaker("t3", failure_threshold=2, cooldown_seconds=60)
cb3.record_failure()
cb3.record_success()
check("成功后计数复位", cb3.allow() is True)
cb3.record_failure()
check("复位后需重新累计", cb3.allow() is True)

check("同名 breaker 复用同一实例", get_breaker("g1") is get_breaker("g1"))
check("不同名 breaker 相互隔离", get_breaker("g2") is not get_breaker("g3"))

# ============================================================
# 2. call_with_timeout
# ============================================================
print("== 2. call_with_timeout ==")
check("快调用返回值", call_with_timeout(lambda: 42, timeout=1.0) == 42)


def _boom():
    raise ValueError("boom")


check("异常 → fallback", call_with_timeout(_boom, timeout=1.0, fallback="FB") == "FB")

try:
    call_with_timeout(_boom, timeout=1.0, fallback=None)
    check("异常且无 fallback → 抛出", False)
except ValueError:
    check("异常且无 fallback → 抛出", True)

_t0 = time.time()
_out = call_with_timeout(lambda: (time.sleep(0.5), "late")[1], timeout=0.05, fallback="FB")
_elapsed = time.time() - _t0
check("超时 → fallback", _out == "FB")
print(f"    [观察] sleep(0.5) + timeout(0.05) 实际耗时 {_elapsed:.2f}s "
      f"（当前实现因 ThreadPoolExecutor 上下文退出会 join 工作线程，"
      f"超时不缩短调用方等待，仅保证降级返回）")

# ============================================================
# 3. reliable_llm_call
# ============================================================
print("== 3. reliable_llm_call ==")
_SENTINEL = object()  # 用唯一对象做 fallback，避免值相等导致的身份误判

c = FakeChain(ret="ANSWER")
check("成功返回真实答案",
      reliable_llm_call(c, [], tier="rel-a", timeout=1.0, fallback=_SENTINEL) == "ANSWER")
check("成功路径已调用模型", c.calls == 1)
check("成功后熔断器闭合", get_breaker("llm-rel-a").allow() is True)

c = FakeChain(exc=RuntimeError("llm down"))
check("异常 → 降级 fallback",
      reliable_llm_call(c, [], tier="rel-b", timeout=1.0, fallback=_SENTINEL) is _SENTINEL)

# 熔断打开：直接降级、不发起调用
b_open = get_breaker("llm-rel-c")
for _ in range(b_open.failure_threshold):
    b_open.record_failure()
check("熔断打开前置条件", b_open.allow() is False)
c = FakeChain(ret="SHOULD_NOT_RUN")
check("熔断打开 → 直接降级",
      reliable_llm_call(c, [], tier="rel-c", timeout=1.0, fallback=_SENTINEL) is _SENTINEL)
check("熔断打开不发起调用", c.calls == 0)
try:
    reliable_llm_call(FakeChain(ret="X"), [], tier="rel-c", timeout=1.0, fallback=None)
    check("熔断打开且无 fallback → 抛错", False)
except RuntimeError:
    check("熔断打开且无 fallback → 抛错", True)

# 连续失败 → 自动打开
for _ in range(get_breaker("llm-rel-d").failure_threshold):
    reliable_llm_call(FakeChain(exc=RuntimeError("x")), [], tier="rel-d",
                      timeout=1.0, fallback=_SENTINEL)
check("连续失败后自动打开", get_breaker("llm-rel-d").allow() is False)

# 超时 → 降级（timed-out 的慢链）
c = FakeChain(ret="late", delay=0.3)
check("超时 → 降级 fallback",
      reliable_llm_call(c, [], tier="rel-e", timeout=0.05, fallback=_SENTINEL) is _SENTINEL)

print()
print(f"== llm_reliability 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
