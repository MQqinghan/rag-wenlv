# -*- coding: utf-8 -*-
"""③-T12 网关鉴权与限流 —— 离线单测（零外部请求）。

覆盖：
  1. _wx_code2session：凭据缺失→mock / 无 mock→失败
  2. _upsert_user：Mongo 不可用→内存兜底且同一 openid 稳定
  3. _issue_tokens + _verify_access：往返 / 类型 / 篡改
  4. _revoke_token + _is_revoked：登出后失效
  5. ws_chat：认证关闭放行 / 开启未带 token→4401 / 有效 token 放行 / 限流 429
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "mobile_gateway_under_test3", ROOT / "app" / "api" / "http" / "mobile_gateway.py"
)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)

from app.shared.runtime.jwt_utils import encode_jwt  # noqa: E402

PASS = 0
FAIL = 0
SECRET = "s" * 40


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


class FakeWS:
    """最小 WebSocket 替身：按帧喂入、记录下发。"""

    def __init__(self, frames: list[str], query_params: dict | None = None) -> None:
        self._frames = list(frames)
        self.sent: list[dict] = []
        self.closed: int | None = None
        self.query_params = query_params or {}
        self.client = None

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        if not self._frames:
            raise mg.WebSocketDisconnect()
        return self._frames.pop(0)

    async def send_json(self, obj) -> None:
        self.sent.append(obj)

    async def close(self, code: int = 1000) -> None:
        self.closed = code


def run_ws(frames, query_params=None):
    ws = FakeWS(frames, query_params)
    asyncio.run(mg.ws_chat(ws))
    return ws


# ============================================================
# 1. _wx_code2session
# ============================================================
print("== 1. _wx_code2session ==")
with mock.patch.object(mg, "env_str", side_effect=lambda k, d="": {"WX_APPID": "", "WX_SECRET": "", "AUTH_MOCK_OPENID": "mock-openid-1"}.get(k, d)):
    check("无凭据+有 mock → 返回 mock openid", mg._wx_code2session("code1") == {"openid": "mock-openid-1"})
with mock.patch.object(mg, "env_str", side_effect=lambda k, d="": "" if k != "WX_APPID" else ""):
    check("无凭据+无 mock → 空 dict", mg._wx_code2session("code1") == {})

# ============================================================
# 2. _upsert_user
# ============================================================
print("== 2. _upsert_user ==")
mg._MEM_USERS.clear()
with mock.patch("app.shared.clients.mongo_history_utils.get_history_mongo_tool", side_effect=RuntimeError("mongo down")):
    u1 = mg._upsert_user("openid-A")
    u2 = mg._upsert_user("openid-A")
    u3 = mg._upsert_user("openid-B")
    check("Mongo 不可用 → 内存兜底出 id", bool(u1))
    check("同一 openid 稳定", u1 == u2)
    check("不同 openid 不同 id", u1 != u3)

# ============================================================
# 3. _issue_tokens / _verify_access
# ============================================================
print("== 3. token ==")
mg._MEM_REVOKED.clear()
with mock.patch.object(mg, "_jwt_secret", return_value=SECRET), mock.patch.object(mg, "_shared_redis", return_value=None):
    tokens = mg._issue_tokens("uid-1", "oid-1")
    check("签发含 access/refresh/user_id/expires_in", all(k in tokens for k in ("access_token", "refresh_token", "user_id", "expires_in")))
    payload = mg._verify_access(tokens["access_token"])
    check("access 校验通过并含 sub", payload and payload.get("sub") == "uid-1")
    check("refresh 不能当 access 用", mg._verify_access(tokens["refresh_token"]) is None)
    check("篡改 token 被拒", mg._verify_access(tokens["access_token"][:-3] + "abc") is None)
    # 登出
    mg._revoke_token(tokens["access_token"], 60)
    check("登出后 access 失效", mg._verify_access(tokens["access_token"]) is None)

with mock.patch.object(mg, "_jwt_secret", return_value=""):
    check("无 JWT_SECRET → 签发空", mg._issue_tokens("u", "o") == {})

# ============================================================
# 4. ws_chat 鉴权
# ============================================================
print("== 4. ws_chat 鉴权 ==")
# 4a. AUTH_ENABLE=false → 无需 token
with mock.patch.object(mg, "_auth_enabled", return_value=False), mock.patch.object(
    mg, "check_rate_limit", return_value=mock.Mock(allowed=True)
), mock.patch.object(mg, "_forward_query", return_value={"answer": "hi", "domain": "tourism"}), mock.patch.object(
    mg, "GATEWAY_AUDIT_PATH", "logs/_tmp_test_gw_audit.jsonl"
):
    ws = run_ws(['{"type":"query","query":"成都好玩吗"}'])
    check("关闭鉴权 → 正常 final", any(f["type"] == "final" for f in ws.sent))

# 4b. AUTH_ENABLE=true + 无 token → 4401
with mock.patch.object(mg, "_auth_enabled", return_value=True), mock.patch.object(mg, "_jwt_secret", return_value=SECRET):
    ws = run_ws(['{"type":"query","query":"成都好玩吗"}'])
    check("开启鉴权+无 token → error", ws.sent and ws.sent[0]["type"] == "error")
    check("错误码 unauthorized", ws.sent[0].get("code") == "unauthorized")
    check("关闭连接 4401", ws.closed == 4401)

# 4c. AUTH_ENABLE=true + 有效 token → 放行
mg._MEM_USERS.clear()
with mock.patch.object(mg, "_auth_enabled", return_value=True), mock.patch.object(
    mg, "_jwt_secret", return_value=SECRET
), mock.patch.object(mg, "_shared_redis", return_value=None), mock.patch.object(
    mg, "check_rate_limit", return_value=mock.Mock(allowed=True)
), mock.patch.object(mg, "_forward_query", return_value={"answer": "成都很好", "domain": "tourism"}), mock.patch.object(
    mg, "GATEWAY_AUDIT_PATH", "logs/_tmp_test_gw_audit.jsonl"
):
    good = encode_jwt({"sub": "uid-9", "openid": "o"}, SECRET, 3600, typ="access")
    ws = run_ws(['{"type":"query","query":"成都好玩吗","token":"' + good + '"}'])
    check("有效 token → final 放行", any(f["type"] == "final" for f in ws.sent))
    check("user_id 取自 token", mg.SESSION_OWNER.get(ws.sent[-1]["session_id"]) == "uid-9")

# 4d. 限流拒绝
with mock.patch.object(mg, "_auth_enabled", return_value=False), mock.patch.object(
    mg, "check_rate_limit", return_value=mock.Mock(allowed=False, retry_after_s=1, dimension="user_qps")
), mock.patch.object(mg, "GATEWAY_AUDIT_PATH", "logs/_tmp_test_gw_audit.jsonl"):
    ws = run_ws(['{"type":"query","query":"成都好玩吗"}'])
    check("限流 → error rate_limited", ws.sent and ws.sent[0].get("code") == "rate_limited")
    check("限流 → 附 retry_after", ws.sent[0].get("retry_after") == 1)

# 清理临时审计文件
try:
    Path("logs/_tmp_test_gw_audit.jsonl").unlink(missing_ok=True)
except Exception:  # noqa: BLE001
    pass

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
