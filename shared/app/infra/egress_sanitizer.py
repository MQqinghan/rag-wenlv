# -*- coding: utf-8 -*-
"""
外发脱敏与图片敏感拦截（A3，T4 落地）。

职责：
1. 文本脱敏（正则 + 词典）：手机号 / 身份证 / 邮箱 / 银行卡 / 金额 / 词典敏感词
   （内部项目代号与重点人名走词典，由 IMPORT_EGRESS_SENSITIVE_WORDS 逗号分隔配置，
   默认空——人名不做通用 NER，避免误伤正文）。
   命中统一替换为分类占位符（如 〔手机号〕），并返回按类别命中计数供审计。
2. 图片敏感预检（本地 OCR 优先，已拍板）：mask 档下 VL 外发图片前，若本地
   PaddleOCR 可用则先本地 OCR → 文本脱敏判定，命中敏感即拦下（不发起 VL 外发）。
   PaddleOCR 未安装时功能自动降级为"不预检"，审计/日志标注 guard 不可用。

设计约束：
- 纯函数、无网络调用；可单测（图片预检依赖可注入）。
- 脱敏只改外发副本，绝不修改原始导入内容（原始文件/切块保持原样）。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from app.shared.config.common import env_str

# ---- 敏感口径分类（A3 拍板起步清单） ----
CAT_PHONE = "phone"
CAT_ID_CARD = "id_card"
CAT_EMAIL = "email"
CAT_BANK_CARD = "bank_card"
CAT_MONEY = "money"
CAT_DICT_WORD = "dict_word"

#: 命中后的统一占位（按分类命名，便于审计识别类别；不泄露原文）
_PLACEHOLDER = {
    CAT_PHONE: "〔手机号〕",
    CAT_ID_CARD: "〔身份证号〕",
    CAT_EMAIL: "〔邮箱〕",
    CAT_BANK_CARD: "〔银行卡号〕",
    CAT_MONEY: "〔金额〕",
    CAT_DICT_WORD: "〔敏感词〕",
}

#: 词典敏感词环境变量（逗号分隔：内部项目代号 / 重点人名）
SENSITIVE_WORDS_ENV = "IMPORT_EGRESS_SENSITIVE_WORDS"
#: 图片本地 OCR 预检开关：auto=可用即启用（默认）/ on=强制（缺失报错提示）/ off=关闭
IMAGE_SENSITIVE_GUARD_ENV = "IMAGE_SENSITIVE_GUARD"


def _build_patterns() -> list[tuple[str, str, re.Pattern]]:
    """(类别, 占位, 正则)。顺序敏感：先长值/词典，后短值，避免互相干扰。"""
    return [
        # 词典词（内部代号/人名）：按词长降序，避免短词先命中吃掉长词子串
        *[(CAT_DICT_WORD, _PLACEHOLDER[CAT_DICT_WORD], re.compile(re.escape(w)))
          for w in sorted(_dict_words(), key=len, reverse=True)],
        # 身份证必须在银行卡之前：17 位数字段若先被 \d{16,19} 吞掉会误标为银行卡
        # （纯 18 位数字结尾的身份证与银行卡在正则上无法完全消歧，必要时用词典/人工复核兜底）
        (CAT_ID_CARD, _PLACEHOLDER[CAT_ID_CARD],
         re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
        (CAT_BANK_CARD, _PLACEHOLDER[CAT_BANK_CARD],
         re.compile(r"(?<!\d)(?:62\d{14,17}|\d{16,19})(?!\d)")),
        (CAT_PHONE, _PLACEHOLDER[CAT_PHONE],
         re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
        (CAT_EMAIL, _PLACEHOLDER[CAT_EMAIL],
         re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")),
        # 金额：人民币符号前缀或数字+「元」；避免裸数字误伤（版本号/年份/页码）
        (CAT_MONEY, _PLACEHOLDER[CAT_MONEY],
         re.compile(r"(?:[¥￥]\s?\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*元)")),
    ]


@lru_cache(maxsize=1)
def _dict_words() -> tuple[str, ...]:
    """词典词（env 逗号分隔），去重、去空白、去空。"""
    raw = env_str(SENSITIVE_WORDS_ENV, "")
    words = tuple(dict.fromkeys(w.strip() for w in raw.split(",") if w.strip()))
    return words


def sanitize_text(text: str) -> tuple[str, dict[str, int]]:
    """
    文本脱敏：替换全部敏感命中为分类占位符。

    Returns:
        (脱敏后文本, 按类别命中计数 dict)。无命中时返回原文同值（不复制）。
    """
    if not text:
        return text, {}
    out = text
    hits: dict[str, int] = {}
    for cat, placeholder, pat in _build_patterns():
        out, n = pat.subn(placeholder, out)
        if n:
            hits[cat] = hits.get(cat, 0) + n
    return (out if hits else text), hits


def mask_message_text(content: str) -> tuple[str, dict[str, int]]:
    """对单段文本消息内容脱敏（HumanMessage/SystemMessage content 为 str 的情形）。"""
    return sanitize_text(content)


# ---- 图片本地 OCR 预检（PaddleOCR 可选依赖，未装自动降级） ----
_PADDLE_READY: bool | None = None  # None=未探测 / True=可用 / False=不可用
_paddle_ocr = None


def _load_paddle_ocr():
    """惰性加载本地 OCR 引擎；任何失败都标记不可用并降级（不阻断 VL 主链路）。"""
    global _PADDLE_READY, _paddle_ocr
    if _PADDLE_READY is not None:
        return _paddle_ocr
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
        _paddle_ocr = PaddleOCR(
            use_angle_cls=False,
            use_textline_orientation=False,
            lang="ch",
            show_log=False,
        )
        _PADDLE_READY = True
    except Exception as e:  # noqa: BLE001 - 本地 OCR 缺失/初始化失败一律降级
        _PADDLE_READY = False
        from app.shared.runtime.logger import logger  # noqa: PLC0415
        logger.warning(
            f"[外发脱敏] 本地 OCR（PaddleOCR）不可用，图片敏感预检降级跳过："
            f"{type(e).__name__}: {e}。需要时请安装 paddlepaddle+paddleocr 后重启。"
        )
    return _paddle_ocr


def image_guard_status() -> dict:
    """图片预检能力状态（供审计/日志/面板）。"""
    mode = env_str(IMAGE_SENSITIVE_GUARD_ENV, "auto").strip().lower()
    if mode == "off":
        return {"enabled": False, "reason": "IMAGE_SENSITIVE_GUARD=off"}
    ready = _PADDLE_READY
    if ready is None:
        _load_paddle_ocr()
        ready = _PADDLE_READY
    if not ready:
        return {"enabled": False, "reason": "paddleocr 不可用（未安装或初始化失败）"}
    return {"enabled": True, "reason": "paddleocr"}


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """本地 OCR 识别图片文字（PaddleOCR 已确认可用时调用）。"""
    ocr = _paddle_ocr
    if ocr is None:
        return ""
    import numpy as np
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = ocr.ocr(np.array(img), cls=False)
    lines: list[str] = []
    for page in result or []:
        for item in page or []:
            try:
                lines.append(str(item[1][0]))
            except Exception:  # noqa: BLE001
                continue
    return "\n".join(lines)


def scan_sensitive_image(image_bytes: bytes, mime: str = "") -> dict:
    """
    mask 档 VL 外发前的图片敏感预检：
    - guard 未启用/不可用 → {"available": False, "blocked": False, "hits": {}}
    - 可用 → 本地 OCR 文本 → 脱敏判定；命中敏感 → {"available": True, "blocked": True, "hits": {...}}
    - 未命中 → blocked False。
    纯本地执行，无任何网络外发。
    """
    status = image_guard_status()
    if not status["enabled"]:
        return {"available": False, "blocked": False, "hits": {}, "reason": status["reason"]}
    try:
        ocr_text = _ocr_image_bytes(image_bytes)
        _, hits = sanitize_text(ocr_text)
        if hits:
            return {"available": True, "blocked": True, "hits": hits, "reason": "本地OCR命中敏感"}
        return {"available": True, "blocked": False, "hits": {}, "reason": "本地OCR未命中"}
    except Exception as e:  # noqa: BLE001 - OCR 单次失败不阻断，降级放行并记录
        return {"available": True, "blocked": False, "hits": {}, "reason": f"本地OCR失败:{type(e).__name__}"}


def hits_total(hits: dict[str, int]) -> int:
    return sum(hits.values())
