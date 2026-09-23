# -*- coding: utf-8 -*-
"""F·Context 长对话压缩：四阶段流水线（分块 → 首尾保护 → 中间摘要 → 归并）。

设计（对齐《框架》Context「长对话压缩」）：
- 不丢首尾：系统设定/开场与最近若干轮原样保留（head/tail protection）。
- 中间轮次：抽取关键事实生成摘要，避免无脑截断导致「忘了之前说啥」。
- summarize 可注入 LLM 摘要函数；不注入时退化为抽取式（取前 N 字符），保证零依赖可用。
- 纯函数、异常安全：压缩失败返回原列表（绝不破坏主链路上下文）。

输入：list[dict]，每项 {"role": "user"/"assistant"/"system", "content": "..."}
输出：压缩后的 list[dict]，结构可直接回灌给下游 Prompt/上下文。
"""
from __future__ import annotations

from typing import Callable, Optional

_DEFAULT_HEAD_K = 4
_DEFAULT_TAIL_K = 6
_DEFAULT_MID_CHARS = 1500


def _as_text(msg) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    return str(getattr(msg, "content", msg))


def _as_role(msg) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "user"))
    return str(getattr(msg, "role", "user"))


def stage_chunk(messages: list) -> list:
    """阶段1：分块。当前按轮次（每条消息为一轮）直接返回，预留后续按 token/主题切分。"""
    return list(messages)


def stage_protect_ends(messages: list, head_k: int, tail_k: int):
    """阶段2：首尾保护。返回 (head, middle, tail) 三段。"""
    n = len(messages)
    if n <= head_k + tail_k:
        return messages[:head_k], [], messages[head_k:] if n > head_k else []
    head = messages[:head_k]
    tail = messages[-tail_k:]
    middle = messages[head_k:n - tail_k]
    return head, middle, tail


def stage_summarize_middle(middle: list, max_mid_chars: int, summarize: Optional[Callable[[str], str]]) -> Optional[str]:
    """阶段3：中间摘要。"""
    if not middle:
        return None
    raw = "\n".join(f"[{_as_role(m)}] {_as_text(m)}" for m in middle)
    if summarize is not None:
        try:
            return summarize(raw)
        except Exception as e:  # noqa: BLE001
            pass  # 降级到抽取式
    # 抽取式降级：保留前 max_mid_chars 字符
    if len(raw) <= max_mid_chars:
        return raw
    return raw[:max_mid_chars] + "\n…（中间对话已压缩）"


def stage_merge(head, summary, tail) -> list:
    """阶段4：归并。中间摘要以单条 system 摘要消息注入，首尾原样。"""
    out = list(head)
    if summary:
        out.append({"role": "system", "content": f"[对话历史摘要]\n{summary}"})
    out.extend(tail)
    return out


def compress_messages(
    messages: list,
    *,
    head_k: int = _DEFAULT_HEAD_K,
    tail_k: int = _DEFAULT_TAIL_K,
    max_mid_chars: int = _DEFAULT_MID_CHARS,
    summarize: Optional[Callable[[str], str]] = None,
) -> list:
    """对外主入口：长对话四阶段压缩。失败回退原列表。"""
    try:
        msgs = stage_chunk(messages)
        head, middle, tail = stage_protect_ends(msgs, head_k, tail_k)
        summary = stage_summarize_middle(middle, max_mid_chars, summarize)
        return stage_merge(head, summary, tail)
    except Exception as e:  # noqa: BLE001
        # 压缩失败绝不影响主链路
        return list(messages)
