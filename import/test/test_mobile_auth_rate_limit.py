# -*- coding: utf-8 -*-
"""③-T12 鉴权/限流基础件 —— 离线单测（零外部请求）。

覆盖：
  1. jwt_utils：往返 / 篡改 / 过期 / typ 校验 / 指纹
  2. inbound_rate_limit：开关 / 内存兜底 / 三档配额 / 用户隔离
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared.runtime import inbound_rate_limit as rl
from app.shared.runtime import jwt_utils as jw

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


SECRET = "x" * 40

# ============================================================
# 1. jwt_utils
# ============================================================
print("== 1. jwt_utils ==")
token = jw.encode_jwt({"sub": "u1", "openid": "o1"}, SECRET, 3600)
check("token 为三段式", token.count(".") == 2)
payload = jw.decode_jwt(token, SECRET)
check("往返：sub 正确", payload and payload.get("sub") == "u1")
check("往返：typ 默认 access", payload and payload.get("typ") == "access")
check("往返：含 iat/exp", payload and isinstance(payload.get("iat"), int) and payload["exp"] > payload["iat"])
check("错误密钥 → None", jw.decode_jwt(token, "y" * 40) is None)
check("篡改签名 → None", jw.decode_jwt(token[:-2] + "ab", SECRET) is None)
check("篡改载荷 → None", jw.decode_jwt(token.split(".")[0] + ".eyJzdWIiOiJoYWNrIn0." + token.split(".")[2], SECRET) is None)
check("格式非法 → None", jw.decode_jwt("not-a-token", SECRET) is None)
check("空 token → None", jw.decode_jwt("", SECRET) is None)

expired = jw.encode_jwt({"sub": "u"}, SECRET, -10)
check("过期 → None", jw.decode_jwt(expired, SECRET) is None)

refresh = jw.encode_jwt({"sub": "u1"}, SECRET, 3600, typ="refresh")
check("typ=refresh 用 expected_typ=refresh 可解", jw.decode_jwt(refresh, SECRET, expected_typ="refresh") is not None)
check("typ=refresh 用 expected_typ=access 拒绝", jw.decode_jwt(refresh, SECRET, expected_typ="access") is None)
check("空 secret → None", jw.decode_jwt(token, "") is None)
check("指纹长度 16", len(jw.token_fingerprint(token)) == 16)
check("指纹稳定", jw.token_fingerprint(token) == jw.token_fingerprint(token))

# ============================================================
# 2. inbound_rate_limit（内存兜底：强制 _get_redis 返回 None）
# ============================================================
print("== 2. inbound_rate_limit ==")
rl.reset_backend_cache()
rl._MEM_COUNTERS.clear()

with mock.patch.object(rl, "_get_redis", return_value=None), mock.patch.object(rl, "env_bool", return_value=False):
    check("RATE_LIMIT_ENABLE=false → 放行", rl.check_rate_limit(user_id="u", ip="1.1.1.1").allowed is True)

with mock.patch.object(rl, "_get_redis", return_value=None), mock.patch.object(rl, "env_bool", return_value=True), \
        mock.patch.object(rl, "env_int", side_effect=lambda k, d: {"RATE_LIMIT_USER_QPS": 2, "RATE_LIMIT_USER_DAILY": 200, "RATE_LIMIT_IP_QPS": 10}.get(k, d)):
    rl._MEM_COUNTERS.clear()
    ts = 1_800_000_000
    r1 = rl.check_rate_limit(user_id="uA", now_ts=ts)
    r2 = rl.check_rate_limit(user_id="uA", now_ts=ts)
    r3 = rl.check_rate_limit(user_id="uA", now_ts=ts)
    check("第 1 次放行", r1.allowed is True)
    check("第 2 次放行", r2.allowed is True)
    check("第 3 次拒绝（QPS=2）", r3.allowed is False)
    check("拒绝维度=user_qps", r3.dimension == "user_qps")
    check("拒绝附 Retry-After=1", r3.retry_after_s == 1)
    check("另一用户不受影响", rl.check_rate_limit(user_id="uB", now_ts=ts).allowed is True)
    check("下一窗口恢复", rl.check_rate_limit(user_id="uA", now_ts=ts + 1).allowed is True)

    # 日配额：窗口 86400，用大计数逼近
    rl._MEM_COUNTERS.clear()
    with mock.patch.object(rl, "env_int", side_effect=lambda k, d: {"RATE_LIMIT_USER_QPS": 999, "RATE_LIMIT_USER_DAILY": 2, "RATE_LIMIT_IP_QPS": 999}.get(k, d)):
        a = rl.check_rate_limit(user_id="uC", now_ts=ts)
        b = rl.check_rate_limit(user_id="uC", now_ts=ts)
        c = rl.check_rate_limit(user_id="uC", now_ts=ts)
        check("日配额内放行", a.allowed and b.allowed)
        check("超日配额拒绝", c.allowed is False and c.dimension == "user_daily")
        check("日配额 retry_after 为正", c.retry_after_s > 0)

    # IP 维度（无 user_id 也应生效）
    rl._MEM_COUNTERS.clear()
    with mock.patch.object(rl, "env_int", side_effect=lambda k, d: {"RATE_LIMIT_USER_QPS": 999, "RATE_LIMIT_USER_DAILY": 999, "RATE_LIMIT_IP_QPS": 2}.get(k, d)):
        i1 = rl.check_rate_limit(ip="9.9.9.9", now_ts=ts)
        i2 = rl.check_rate_limit(ip="9.9.9.9", now_ts=ts)
        i3 = rl.check_rate_limit(ip="9.9.9.9", now_ts=ts)
        check("IP 前两次放行", i1.allowed and i2.allowed)
        check("IP 第三次拒绝（IP_QPS=2）", i3.allowed is False and i3.dimension == "ip_qps")

rl.reset_backend_cache()

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
