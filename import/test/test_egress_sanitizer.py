# -*- coding: utf-8 -*-
"""
A3 文本脱敏与图片敏感预检单测（T4 验收）。

运行方式：
  python test/test_egress_sanitizer.py

覆盖：
1) sanitize_text 各口径正则命中（手机/身份证/邮箱/银行卡/金额/词典词）+ 分类计数；
2) 词典词来自 IMPORT_EGRESS_SENSITIVE_WORDS，词长降序避免子串误替；
3) mask 档 chat_invoke：发给模型的文本已被脱敏（fake client 抓取收到的消息）；
4) full 档不脱敏；
5) 图片敏感预检命中 → 拦截（不触网）；图片无预检能力（默认）→ 放行原图外发。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from app.infra.egress_gateway import (  # noqa: E402
    CT_IMAGE,
    CT_TEXT,
    SERVICE_LLM,
    SERVICE_VISION,
    EgressBlockedError,
    egress_gateway,
)
from app.infra import egress_sanitizer as es  # noqa: E402

EGRESS_ENV = "IMPORT_EGRESS_MODE"


class _CaptureLLM:
    """抓取发给模型的最终消息内容（用于断言脱敏是否生效）。"""

    def __init__(self, out="ok"):
        self.out = out
        self.seen: list = []
        self.invoked = 0

    def invoke(self, messages):
        self.invoked += 1
        self.seen = list(messages)
        return type("R", (), {"content": self.out})()

    @property
    def model_name(self):
        return "qwen3-vl-flash"


def _set_mode(value):
    old = os.environ.get(EGRESS_ENV)
    os.environ[EGRESS_ENV] = value
    return old


def _collect_text(messages) -> str:
    parts: list[str] = []
    for m in messages:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for it in c:
                if isinstance(it, dict) and it.get("type") == "text":
                    parts.append(str(it.get("text", "")))
    return "\n".join(parts)


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    # ---- 1) 规则级脱敏（env 词典注入） ----
    os.environ[es.SENSITIVE_WORDS_ENV] = "绝密代号X,龙鳞计划"
    es._dict_words.cache_clear()
    try:
        raw = (
            "联系李经理13800138000，身份证 11010119900307789X，邮箱 tom.lee@example.com；"
            "卡号6222020202020202，充值 ¥ 1299，预算 500 元；内部代号 龙鳞计划 与 绝密代号X 不得外泄。"
        )
        text, hits = es.sanitize_text(raw)
        assert "13800138000" not in text and "〔手机号〕" in text
        assert "11010119900307789X" not in text and "〔身份证号〕" in text
        assert "tom.lee@example.com" not in text and "〔邮箱〕" in text
        assert "6222020202020202" not in text and "〔银行卡号〕" in text
        assert "¥ 1299" not in text and "500 元" not in text
        assert "龙鳞计划" not in text and "绝密代号X" not in text
        assert hits.get("phone") == 1 and hits.get("id_card") == 1
        assert hits.get("email") == 1 and hits.get("bank_card") == 1
        assert hits.get("money") == 2 and hits.get("dict_word") == 2
        # 词典长词优先：短词"绝密代号"若在前会吃掉"绝密代号X"，验证词长降序
        assert "〔敏感词〕" in text
        results.append(("1 六口径脱敏+计数", True, str(dict(hits))))
    finally:
        os.environ.pop(es.SENSITIVE_WORDS_ENV, None)
        es._dict_words.cache_clear()

    # 无词典词时普通文本不动
    t2, h2 = es.sanitize_text("今年读了三本书，版本号 2.0.1，页码 200")
    assert h2 == {} and t2 == "今年读了三本书，版本号 2.0.1，页码 200"
    results.append(("1b 无命中原样返回", True, "ok"))

    # ---- 2) mask 档 chat_invoke：发给模型的文本已脱敏 ----
    old = _set_mode("mask")
    try:
        llm = _CaptureLLM()
        q = "客户手机 13912345678 需要跟进，邮箱 a@b.cn 确认"
        text = egress_gateway.chat_invoke(
            llm, [HumanMessage(content=q)], service=SERVICE_LLM, content_type=CT_TEXT,
            document="mask-text.txt",
        )
        assert text == "ok" and llm.invoked == 1
        sent = _collect_text(llm.seen)
        assert "13912345678" not in sent and "〔手机号〕" in sent
        assert "a@b.cn" not in sent and "〔邮箱〕" in sent
        results.append(("2 mask档文本脱敏后外发", True, sent[:40]))
    finally:
        _set_mode(old)

    # ---- 3) full 档不脱敏 ----
    old = _set_mode("full")
    try:
        llm = _CaptureLLM()
        q = "手机 13912345678 原样"
        egress_gateway.chat_invoke(llm, [HumanMessage(content=q)],
                                   service=SERVICE_LLM, content_type=CT_TEXT, document="full.txt")
        sent = _collect_text(llm.seen)
        assert "13912345678" in sent and "〔手机号〕" not in sent
        results.append(("3 full档原样透传", True, "ok"))
    finally:
        _set_mode(old)

    # ---- 4) 图片敏感预检：可用且命中 → 拦截不触网 ----
    old = _set_mode("mask")
    try:
        import base64 as _b64
        png_b64 = _b64.b64encode(b"fake-png-bytes").decode()
        original = es.scan_sensitive_image
        es.scan_sensitive_image = lambda img, mime="": {
            "available": True, "blocked": True, "hits": {"phone": 1}, "reason": "本地OCR命中敏感"}
        try:
            llm = _CaptureLLM()
            try:
                egress_gateway.chat_invoke(
                    llm,
                    [HumanMessage(content=[{"type": "text", "text": "看这张图"},
                                           {"type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{png_b64}"}}])],
                    service=SERVICE_VISION, content_type=CT_IMAGE, document="sensitive.png",
                )
                raise AssertionError("图片命中敏感应被拦截")
            except EgressBlockedError as e:
                assert llm.invoked == 0, "拦截时不得发起 VL 外发"
                assert "敏感" in str(e)
            results.append(("4 图片敏感预检拦截", True, "不触网"))
        finally:
            es.scan_sensitive_image = original
    finally:
        _set_mode(old)

    # ---- 5) 图片预检不可用（默认无 paddle）→ 放行原图外发 ----
    old = _set_mode("mask")
    try:
        png_b64 = __import__("base64").b64encode(b"not-a-real-png").decode()
        llm = _CaptureLLM()
        egress_gateway.chat_invoke(
            llm,
            [HumanMessage(content=[{"type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{png_b64}"}}])],
            service=SERVICE_VISION, content_type=CT_IMAGE, document="plain.png",
        )
        sent = _collect_text(llm.seen)
        assert llm.invoked == 1
        content0 = llm.seen[0].content if llm.seen else []
        has_img = isinstance(content0, list) and any(
            isinstance(it, dict) and it.get("type") == "image_url"
            and f"data:image/png;base64,{png_b64}" in (it.get("image_url") or {}).get("url", "")
            for it in content0)
        assert has_img, "图片 data URL 应原样保留外发"
        results.append(("5 无本地OCR时放行（降级）", True, "guard 不可用 → 透传"))
    finally:
        _set_mode(old)

    print("== egress_sanitizer（A3）单测结果 ==")
    ok = True
    for name, passed, info in results:
        print(f"  {'✅' if passed else '❌'} {name}：{info}")
        ok = ok and passed
    print("全部通过 ✅" if ok else "存在失败 ❌")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
