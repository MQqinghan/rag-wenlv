# -*- coding: utf-8 -*-
"""
高德网关限流治理离线单测（不发真实 HTTP 请求，mock httpx.Client）。

覆盖 2026-09-09 新增的限流修复：
1. 命中限流 → 退避重试 → 最终成功；
2. 限流重试额度耗尽 → 返回最后一次限流体（不抛异常、不冒充成功）；
3. 业务错误（INVALID_USER_KEY）→ 不重试（重试无意义且会加剧限流）；
4. 网络异常沿用原有 retry_times 额度，不挤占限流额度；
5. 进程级节流器按最小间隔限速（线程安全）。

运行：cd 项目根 && .venv\\Scripts\\python.exe test\\test_amap_rate_limit.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infra import amap_gateway as gw

# 单测加速：退避基数调到 1ms，默认节流关闭
gw.AMAP_BACKOFF_BASE_MS = 1
gw._throttle = gw._Throttle(0)


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """按脚本依次返回响应；`state` 为跨实例共享的游标（每次 _http_get 会新建 Client）。"""

    def __init__(self, script: list[dict], calls: list[str], state: dict):
        self._script = script
        self._calls = calls
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, params=None):
        self._calls.append(url)
        idx = self._state["i"]
        self._state["i"] = idx + 1
        return _FakeResp(self._script[min(idx, len(self._script) - 1)])


def _install(monkey_script: list[dict]) -> list[str]:
    """用脚本替换 httpx.Client，返回调用记录列表。"""
    calls: list[str] = []
    state = {"i": 0}

    def _factory(*args, **kwargs):
        return _FakeClient(monkey_script, calls, state)

    gw.httpx.Client = _factory  # type: ignore[assignment]
    return calls


_RATE_LIMIT_BODY = {"status": "0", "infocode": "10021", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT"}
_OK_BODY = {"status": "1", "pois": [{"id": "p1", "name": "故宫博物院"}]}
_KEY_ERROR_BODY = {"status": "0", "infocode": "10001", "info": "INVALID_USER_KEY"}


def test_rate_limit_then_success():
    """场景1：前 2 次限流、第 3 次成功 → 返回成功数据，共 3 次调用。"""
    calls = _install([_RATE_LIMIT_BODY, _RATE_LIMIT_BODY, _OK_BODY])
    data = gw.amap_gateway._http_get("http://x/text", {"keywords": "故宫"})
    assert data == _OK_BODY, f"应返回成功响应体，实际：{data}"
    assert len(calls) == 3, f"应重试至成功（3 次调用），实际 {len(calls)} 次"


def test_rate_limit_exhausted_returns_last_body():
    """场景2：限流额度耗尽 → 返回最后一次限流体，不抛异常（上层按 status!=1 软失败）。"""
    calls = _install([_RATE_LIMIT_BODY])
    data = gw.amap_gateway._http_get("http://x/text", {"keywords": "故宫"})
    assert data == _RATE_LIMIT_BODY, f"应原样返回限流体，实际：{data}"
    total = max(1, gw.AMAP_RETRY_TIMES + 1) + max(0, gw.AMAP_RATE_LIMIT_RETRIES)
    assert len(calls) == total, f"应按总额度 {total} 次尝试，实际 {len(calls)} 次"


def test_business_error_no_retry():
    """场景3：业务错误（密钥无效）不重试——重试无意义且会加剧限流。"""
    calls = _install([_KEY_ERROR_BODY])
    data = gw.amap_gateway._http_get("http://x/text", {"keywords": "故宫"})
    assert data == _KEY_ERROR_BODY
    assert len(calls) == 1, f"业务错误不应重试，实际 {len(calls)} 次"
    assert gw._is_rate_limited(_KEY_ERROR_BODY) is False, "INVALID_USER_KEY 不应判定为限流"
    assert gw._is_rate_limited(_RATE_LIMIT_BODY) is True, "CUQPS 超限必须判定为限流"
    assert gw._is_rate_limited(_OK_BODY) is False, "成功响应不应判定为限流"


def test_network_error_uses_own_budget():
    """场景4：网络异常沿用 retry_times 额度，不借限流额度重试。"""
    calls: list[str] = []

    class _BoomClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            calls.append(url)
            raise RuntimeError("network down")

    gw.httpx.Client = _BoomClient  # type: ignore[assignment]
    try:
        gw.amap_gateway._http_get("http://x/text", {"keywords": "故宫"})
        raise AssertionError("网络异常应上抛")
    except RuntimeError:
        pass
    assert len(calls) == gw.AMAP_RETRY_TIMES + 1, (
        f"网络异常应只重试 {gw.AMAP_RETRY_TIMES + 1} 次，实际 {len(calls)} 次"
    )


def test_throttle_min_interval_thread_safe():
    """场景5：节流器按最小间隔限速，多线程共享同一额度。"""
    throttle = gw._Throttle(120)  # 120ms 最小间隔
    counter = {"n": 0}
    lock = threading.Lock()

    def worker():
        for _ in range(3):
            throttle.acquire()
            with lock:
                counter["n"] += 1

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    assert counter["n"] == 9, f"9 次 acquire 应全部完成，实际 {counter['n']}"
    # 9 次请求、最小间隔 120ms → 至少等待 8 * 120ms = 0.96s（留 10% 余量防抖）
    assert elapsed >= 0.96 * 0.9, f"节流未生效：9 次请求仅耗时 {elapsed:.3f}s"


def main() -> int:
    tests = [
        test_rate_limit_then_success,
        test_rate_limit_exhausted_returns_last_body,
        test_business_error_no_retry,
        test_network_error_uses_own_budget,
        test_throttle_min_interval_thread_safe,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n汇总：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
