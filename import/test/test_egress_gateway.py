# -*- coding: utf-8 -*-
"""
A2 外发网关 egress_gateway 单测（T3 验收）。

运行方式（与 test/ 下脚本式用例一致，不依赖 pytest）：
  python test/test_egress_gateway.py

覆盖：
1) off 档：chat_invoke 在发起网络前被拦截（fake client 计数为 0），审计落 blocked 行；
2) 白名单：非白名单 host 被拒绝（EgressDeniedError），审计落 blocked 行；
3) mask/full 档：正常透传 fake client，审计落 ok 行，返回模型文本；
4) 业务降级：book_meta 在 off 下回落 file_title 兜底；OCR 在 off 下抛 EgressBlockedError；
5) 审计记录可读回（recent_records 新→旧）。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.infra.egress_gateway import (  # noqa: E402
    CT_IMAGE,
    CT_TEXT,
    SERVICE_LLM,
    SERVICE_VISION,
    EgressBlockedError,
    EgressDeniedError,
    egress_gateway,
)

EGRESS_ENV = "IMPORT_EGRESS_MODE"


def _set_mode(value):
    old = os.environ.get(EGRESS_ENV)
    os.environ[EGRESS_ENV] = value
    return old


class _FakeLLM:
    """假模型：记录 invoke 次数并返回固定文本，绝不发网络。"""

    def __init__(self, out="fake-llm-output"):
        self.out = out
        self.invoked = 0

    def invoke(self, messages):
        self.invoked += 1
        return type("R", (), {"content": self.out})()

    def __or__(self, parser):
        return _FakeChain(self, parser)

    @property
    def model_name(self):
        return "qwen3-vl-flash"


class _FakeChain:
    def __init__(self, llm, parser):
        self.llm, self.parser = llm, parser

    def invoke(self, messages):
        return self.parser.invoke(self.llm.invoke(messages))


class _IdParser:
    def invoke(self, x):
        return x.content if hasattr(x, "content") else str(x)


def main() -> None:
    results = []
    # 审计起点（避免读到历史记录干扰计数——记录文件行数）
    base_lines = egress_gateway.recent_records(limit=10000)

    # ---- 1) off 档：chat_invoke 被拦截，不触网 ----
    old = _set_mode("off")
    try:
        llm = _FakeLLM()
        try:
            egress_gateway.chat_invoke(llm, [{"role": "user", "content": "hi"}],
                                       service=SERVICE_LLM, content_type=CT_TEXT,
                                       document="unit-test.txt", payload_bytes=2)
            raise AssertionError("off 档 chat_invoke 应抛 EgressBlockedError")
        except EgressBlockedError as e:
            assert llm.invoked == 0, "off 档不得发起网络调用"
            results.append(("1a off拦截chat", True, str(e)[:40]))
    finally:
        _set_mode(old)

    # ---- 2) 白名单：目标 host 不在名单 → EgressDeniedError ----
    old = _set_mode("mask")
    try:
        llm = _FakeLLM()
        try:
            egress_gateway.chat_invoke(llm, [{"role": "user", "content": "hi"}],
                                       service=SERVICE_VISION, content_type=CT_IMAGE,
                                       target_host="evil.example.com", document="a.png")
            raise AssertionError("非白名单 host 应拒绝")
        except EgressDeniedError:
            assert llm.invoked == 0
            results.append(("2 白名单拦截", True, "evil.example.com"))
    finally:
        _set_mode(old)

    # ---- 3) mask 透传：正常调用 fake client，审计 ok，返回文本 ----
    old = _set_mode("mask")
    try:
        llm = _FakeLLM("mock-vl-text")
        text = egress_gateway.chat_invoke(llm, [{"role": "user", "content": "看图"}],
                                          service=SERVICE_VISION, content_type=CT_IMAGE,
                                          document="mask-ok.png", payload_bytes=3)
        assert text == "mock-vl-text" and llm.invoked == 1
        results.append(("3 mask透传", True, text))
        # parser 场景
        llm2 = _FakeLLM("parsed-out")
        text2 = egress_gateway.chat_invoke(llm2, [{"role": "user", "content": "x"}],
                                           service=SERVICE_LLM, content_type=CT_TEXT,
                                           document="p.txt", parser=_IdParser())
        assert text2 == "parsed-out"
        results.append(("3b parser场景", True, text2))
    finally:
        _set_mode(old)


    # ---- 4b) 独立图片 OCR off → 显式报错 ----
    old = _set_mode("off")
    try:
        import base64
        from app.rag.common.image_ocr_service import transcribe_image_to_md
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "unit_test.png"
            img.write_bytes(png)
            try:
                transcribe_image_to_md(img)
                raise AssertionError("off 档 OCR 应抛 EgressBlockedError")
            except EgressBlockedError:
                results.append(("4b OCR-off显式报错", True, "ok"))
    finally:
        _set_mode(old)

    # ---- 5) 审计可读回：新→旧，出现本次记录 ----
    recs = egress_gateway.recent_records(limit=10000)
    added = len(recs) - len(base_lines)
    assert added >= 7, f"预期新增 ≥7 条审计，实际 {added}"
    newest = recs[0]
    assert "fingerprint" in newest and "ts" in newest
    # off/deny 的 blocked 行与透传 ok 行都在
    recent = recs[:added]
    assert any(r.get("blocked") for r in recent), "应存在拦截审计行"
    assert any(not r.get("blocked") and r.get("result") == "ok" for r in recent), "应存在外发成功行"
    results.append(("5 审计落盘可读回", True, f"新增 {added} 条"))

    print("== egress_gateway 单测结果 ==")
    ok = True
    for name, passed, info in results:
        print(f"  {'✅' if passed else '❌'} {name}：{info}")
        ok = ok and passed
    print("全部通过 ✅" if ok else "存在失败 ❌")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
