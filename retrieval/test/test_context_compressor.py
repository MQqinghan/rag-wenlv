# -*- coding: utf-8 -*-
"""F·Context 长对话压缩四阶段单测 —— 纯函数、离线、零依赖。

覆盖：
  1. stage_protect_ends：长对话三段切分 / 短对话无 middle / 恰好等于阈值
  2. stage_summarize_middle：空 middle → None / 注入摘要优先 / 摘要异常降级抽取式 / 超长截断带标记
  3. stage_merge：head + system 摘要 + tail 结构
  4. compress_messages：短对话不压缩 / 长对话首尾保护 + 中间摘要 / 支持对象型消息 / 空列表
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.common import context_compressor as cc

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


def mk(n: int):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


# ============================================================
# 1. stage_protect_ends
# ============================================================
print("== 1. stage_protect_ends ==")
msgs = mk(12)
head, middle, tail = cc.stage_protect_ends(msgs, 4, 6)
check("head=前 4 条原样", [m["content"] for m in head] == ["m0", "m1", "m2", "m3"])
check("tail=后 6 条原样", [m["content"] for m in tail] == ["m6", "m7", "m8", "m9", "m10", "m11"])
check("middle=中间 2 条", [m["content"] for m in middle] == ["m4", "m5"])

h, m, t = cc.stage_protect_ends(mk(8), 4, 6)
check("n<head+tail 无 middle", m == [])
check("短对话 tail 取剩余", [x["content"] for x in t] == ["m4", "m5", "m6", "m7"])

h, m, t = cc.stage_protect_ends(mk(10), 4, 6)
check("n==head+tail 无 middle", m == [] and len(h) == 4 and len(t) == 6)

h, m, t = cc.stage_protect_ends([], 4, 6)
check("空列表安全", h == [] and m == [] and t == [])

# ============================================================
# 2. stage_summarize_middle
# ============================================================
print("== 2. stage_summarize_middle ==")
check("空 middle → None", cc.stage_summarize_middle([], 100, None) is None)
check("注入摘要函数优先",
      cc.stage_summarize_middle(mk(3), 100, lambda raw: "SUM") == "SUM")


def _raise(_raw):
    raise RuntimeError("summarizer down")


s = cc.stage_summarize_middle(mk(3), 100, _raise)
check("摘要异常 → 抽取式降级（不崩）", isinstance(s, str) and "m0" in s)

long_mid = [{"role": "user", "content": "x" * 500} for _ in range(5)]
s = cc.stage_summarize_middle(long_mid, 100, None)
# 抽取式 raw 形如 "[user] xxxx..."，截断到 max_mid_chars 后追加压缩标记
check("超长 → 截断并带压缩标记",
      s.startswith("[user]") and "压缩" in s and len(s) < 200)

# ============================================================
# 3. stage_merge
# ============================================================
print("== 3. stage_merge ==")
out = cc.stage_merge(head, "SUM", tail)
check("首条为首部", out[0]["content"] == "m0")
check("摘要以 system 注入", out[4]["role"] == "system" and "对话历史摘要" in out[4]["content"])
check("尾部紧随摘要", out[5]["content"] == "m6")
check("无摘要则仅首尾拼接", len(cc.stage_merge(head, None, tail)) == len(head) + len(tail))

# ============================================================
# 4. compress_messages
# ============================================================
print("== 4. compress_messages ==")
check("空列表安全", cc.compress_messages([]) == [])
short = cc.compress_messages(mk(6))
check("短对话不压缩（结构不变）", len(short) == 6 and short[0]["content"] == "m0")

compressed = cc.compress_messages(mk(20), summarize=lambda raw: "SUMMARY")
check("长对话首尾原样", compressed[0]["content"] == "m0" and compressed[-1]["content"] == "m19")
check("中间被替换为摘要", any("SUMMARY" in m["content"] for m in compressed))
check("总条数显著减少", len(compressed) < 20)

defaulted = cc.compress_messages(mk(20))  # 不注入摘要 → 抽取式降级
check("无摘要函数仍可压缩（抽取式）", len(defaulted) < 20 and defaulted[0]["content"] == "m0")


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


obj_out = cc.compress_messages([_Msg("user", f"o{i}") for i in range(20)])
# head/tail 段为原样返回（不强制转 dict），用 _as_text 取内容验证不崩且顺序正确
check("支持对象型消息（不崩、首尾原样）",
      cc._as_text(obj_out[0]) == "o0" and cc._as_text(obj_out[-1]) == "o19")
check("dict 输入 → 输出仍全为 dict",
      all(isinstance(m, dict) for m in cc.compress_messages(mk(20))))

print()
print(f"== context_compressor 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
