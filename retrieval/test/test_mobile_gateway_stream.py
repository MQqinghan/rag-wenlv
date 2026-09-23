# -*- coding: utf-8 -*-
"""③-T11 移动网关流式 WSS 转发 —— 离线单测（零外部请求）。

覆盖：
  1. _iter_sse：SSE 帧解析（正常 / raw 兜底 / 多帧）
  2. _handle_stream_query：ready→progress→delta→final 逐帧转发；session_id 采用上游值
  3. 上游发起失败 / 消费异常 → error 帧
  4. 流式审计落盘（mode=stream）
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "mobile_gateway_under_test2", ROOT / "app" / "api" / "http" / "mobile_gateway.py"
)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)

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


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, obj) -> None:
        self.sent.append(obj)


def run_stream(ws: FakeWS, query: str = "成都三日游", session_id: str = "s1", user_id: str = "u1") -> None:
    asyncio.run(mg._handle_stream_query(ws, query, session_id, user_id, time.time()))


# ============================================================
# 1. _iter_sse
# ============================================================
print("== 1. _iter_sse ==")
frames = list(
    mg._iter_sse(
        [
            "event: ready\n",
            "data: {}\n",
            "\n",
            'event: delta\n',
            'data: {"content": "成"}\n',
            "\n",
            "data: not-json\n",
            "\n",
            ": comment line\n",
        ]
    )
)
check("帧数=3", len(frames) == 3)
check("ready 帧", frames[0] == ("ready", {}))
check("delta 帧", frames[1] == ("delta", {"content": "成"}))
check("非法 JSON → raw 兜底且 event 默认 message", frames[2] == ("message", {"raw": "not-json"}))
check("bytes 输入可用", list(mg._iter_sse([b"event: final\n", b'data: {"answer": "ok"}\n'])) == [("final", {"answer": "ok"})])

# ============================================================
# 2. 流式端到端（全 mock）
# ============================================================
print("== 2. 流式端到端 ==")


def _fake_consume(sid, on_frame):
    on_frame("progress", {"node": "node_rrf"})
    on_frame("delta", {"content": "成都"})
    on_frame("delta", {"content": "很好"})
    on_frame("final", {"answer": "成都很好"})


with tempfile.TemporaryDirectory() as td, mock.patch.object(
    mg, "_post_query", return_value={"session_id": "s99"}
) as pq, mock.patch.object(mg, "_consume_stream", side_effect=_fake_consume), mock.patch.object(
    mg, "GATEWAY_AUDIT_PATH", str(Path(td) / "a.jsonl")
):
    ws = FakeWS()
    run_stream(ws, session_id="cli-sess")
    types = [f["type"] for f in ws.sent]
    check("帧序列 ready→progress→delta→delta→final", types == ["ready", "progress", "delta", "delta", "final"])
    check("session_id 采用上游返回值", all(f.get("session_id") == "s99" for f in ws.sent))
    check("触发用 is_stream=True", pq.call_args[0][2] is True)
    check("delta 负载透传", ws.sent[2]["data"] == {"content": "成都"})
    recs = (Path(td) / "a.jsonl").read_text(encoding="utf-8").strip().splitlines()
    check("流式审计落盘一行", len(recs) == 1)
    rec = json.loads(recs[0])
    check("审计 mode=stream", rec.get("mode") == "stream")
    check("审计 code=OK", rec.get("code") == "OK" and rec.get("ok") is True)

# ============================================================
# 3. 异常路径
# ============================================================
print("== 3. 异常路径 ==")
with tempfile.TemporaryDirectory() as td, mock.patch.object(
    mg, "_post_query", side_effect=RuntimeError("upstream down")
), mock.patch.object(mg, "GATEWAY_AUDIT_PATH", str(Path(td) / "a.jsonl")):
    ws = FakeWS()
    run_stream(ws)
    check("上游发起失败 → 单个 error 帧", len(ws.sent) == 1 and ws.sent[0]["type"] == "error")
    check("错误码 UPSTREAM_ERROR", ws.sent[0].get("code") == "UPSTREAM_ERROR")

with tempfile.TemporaryDirectory() as td, mock.patch.object(
    mg, "_post_query", return_value={"session_id": "s"}
), mock.patch.object(mg, "_consume_stream", side_effect=RuntimeError("stream broken")), mock.patch.object(
    mg, "GATEWAY_AUDIT_PATH", str(Path(td) / "a.jsonl")
):
    ws = FakeWS()
    run_stream(ws)
    check("消费异常 → 先 ready 再 error", [f["type"] for f in ws.sent] == ["ready", "error"])
    rec = json.loads((Path(td) / "a.jsonl").read_text(encoding="utf-8").strip())
    check("消费异常审计 code=UPSTREAM_ERROR", rec.get("code") == "UPSTREAM_ERROR")

print(f"\n结果：{PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
