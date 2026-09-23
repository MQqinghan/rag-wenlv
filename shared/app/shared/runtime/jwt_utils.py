# -*- coding: utf-8 -*-
"""
零依赖 HS256 JWT（RFC 7519 子集）—— 移动网关自签 token 用。

对应 docs/网关鉴权与限流设计.md 二、身份链路：
- 只用标准库 hmac/hashlib/base64/json，避免为网关引入 PyJWT 依赖面。
- 签名校验用 hmac.compare_digest（恒定时间），防时序侧信道。
- 只做「签名 + 过期」校验，不做算法协商（header.alg 恒为 HS256，避免 alg=none 攻击）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

_ALG = "HS256"


def _b64url_encode(raw: bytes) -> str:
    """URL 安全 base64（去 padding）。"""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(seg: str) -> bytes:
    """URL 安全 base64 解码（补 padding）。"""
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def _sign(signing_input: bytes, secret: str) -> str:
    """HS256 签名 → b64url。"""
    digest = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return _b64url_encode(digest)


def encode_jwt(payload: dict, secret: str, ttl_s: int, *, typ: str = "access") -> str:
    """
    生成 HS256 JWT：自动补 iat/exp/typ（payload 中已有 exp 时以 ttl_s 为准）。

    Args:
        payload: 业务载荷（如 {"sub": user_id, "openid": ...}）。
        secret: HS256 密钥（≥32 字节随机串）。
        ttl_s: 有效期（秒）。
        typ: token 类型（access / refresh）。
    """
    now = int(time.time())
    body = dict(payload)
    body["iat"] = now
    body["exp"] = now + int(ttl_s)
    body["typ"] = typ
    header = {"alg": _ALG, "typ": "JWT"}
    h = _b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    p = _b64url_encode(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{h}.{p}".encode("ascii")
    return f"{h}.{p}.{_sign(signing_input, secret)}"


def decode_jwt(token: str, secret: str, *, expected_typ: str | None = None) -> dict | None:
    """
    校验签名与过期，返回 payload；任何失败（格式/签名/过期/类型不符）返回 None。
    """
    if not token or not secret:
        return None
    parts = str(token).split(".")
    if len(parts) != 3:
        return None
    h, p, s = parts
    signing_input = f"{h}.{p}".encode("ascii")
    if not hmac.compare_digest(s, _sign(signing_input, secret)):
        return None
    try:
        payload = json.loads(_b64url_decode(p).decode("utf-8"))
    except Exception:  # noqa: BLE001 - 任意解码失败一律视为无效 token
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or int(exp) < int(time.time()):
        return None
    if expected_typ and payload.get("typ") != expected_typ:
        return None
    return payload


def token_fingerprint(token: str) -> str:
    """token 指纹（sha256 前 16），用于黑名单/审计——不记原文。"""
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()[:16]
