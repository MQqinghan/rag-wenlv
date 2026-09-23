# -*- coding: utf-8 -*-
"""B1 意图路由 LLM 主导化单测（intent_route_llm）—— 离线、LLM 全 mock。

覆盖：
  1. _normalize_led：非法域收敛 / rewritten 回退 / source 按域默认 / tier 收敛 /
     tools 白名单过滤 / confidence 截断 / is_plan 布尔化
  2. route_intent 开关 OFF：完全走 legacy，字段保守默认
  3. 开关 ON：知识型句式被 LLM 判闲聊 → 硬覆盖为 tourism
  4. 开关 ON：LLM 未给合法域 → legacy 兜底求域
  5. 开关 ON：规划信号硬覆盖 is_plan
  6. route_by_llm_led 异常安全：LLM 挂掉返回空域 + rewritten 回退 query
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.common import intent_route_llm as irl
from app.rag.common.intent_route_service import (
    DOMAIN_CHITCHAT,
    DOMAIN_TOURISM,
)

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


# ============================================================
# 1. _normalize_led
# ============================================================
print("== 1. _normalize_led ==")
r = irl._normalize_led({}, "原始问句")
check("非法/缺失域 → 空串", r["domain"] == "")
check("rewritten 缺失回退 query", r["rewritten"] == "原始问句")
check("空域 source 默认 kb", r["source_policy"] == "kb")
check("非法 tier → standard", r["model_tier"] == "standard")
check("tools 缺失 → 空列表", r["tools"] == [])
check("is_plan 缺失 → False", r["is_plan"] is False)
check("confidence 缺失 → 0.0", r["confidence"] == 0.0)

check("tourism 无 source → kb_then_web",
      irl._normalize_led({"domain": DOMAIN_TOURISM}, "q")["source_policy"] == "kb_then_web")
check("chitchat 无 source → kb_then_web",
      irl._normalize_led({"domain": DOMAIN_CHITCHAT}, "q")["source_policy"] == "kb_then_web")
check("合法域保留",
      irl._normalize_led({"domain": DOMAIN_CHITCHAT}, "q")["domain"] == DOMAIN_CHITCHAT)
check("tools 白名单过滤非法项",
      irl._normalize_led({"tools": ["weather", "bad", "rail"]}, "q")["tools"] == ["weather", "rail"])
check("非法 tier 保持 standard",
      irl._normalize_led({"model_tier": "super"}, "q")["model_tier"] == "standard")
check("合法 tier 保留",
      irl._normalize_led({"model_tier": "longctx"}, "q")["model_tier"] == "longctx")
check("confidence 上越界截断",
      irl._normalize_led({"confidence": 9}, "q")["confidence"] == 1.0)
check("confidence 下越界截断",
      irl._normalize_led({"confidence": -3}, "q")["confidence"] == 0.0)
check("confidence 非数 → 0.0",
      irl._normalize_led({"confidence": "x"}, "q")["confidence"] == 0.0)
check("is_plan 真值布尔化", irl._normalize_led({"is_plan": 1}, "q")["is_plan"] is True)
check("空 rewritten 回退 query",
      irl._normalize_led({"rewritten": "   "}, "q")["rewritten"] == "q")

# ============================================================
# 2. route_intent 开关 OFF（legacy 等价）
# ============================================================
print("== 2. route_intent 开关 OFF ==")
with mock.patch.object(irl, "_legacy_domain", return_value=DOMAIN_TOURISM) as lg:
    out = irl.route_intent("随便问问", use_llm_first=False)
    check("OFF: route_mode=legacy", out["route_mode"] == "legacy")
    check("OFF: 调用了 legacy 域判定", lg.call_count == 1)
    check("OFF: source=kb_then_web", out["source_policy"] == "kb_then_web")
    check("OFF: 其余字段保守默认",
          out["tools"] == [] and out["is_plan"] is False
          and out["model_tier"] == "standard" and out["rewritten"] == "随便问问")

with mock.patch.object(irl, "_legacy_domain", return_value=DOMAIN_CHITCHAT):
    out = irl.route_intent("q", use_llm_first=False)
    check("OFF: chitchat → source=kb_then_web", out["source_policy"] == "kb_then_web")

# ============================================================
# 3. 开关 ON：知识型句式硬覆盖
# ============================================================
print("== 3. 开关 ON 知识型硬覆盖 ==")
_led_chitchat = {
    "domain": DOMAIN_CHITCHAT, "rewritten": "成都怎么玩", "source_policy": "kb_then_web",
    "model_tier": "lite", "tools": ["weather"], "is_plan": False, "confidence": 0.8,
}
# 硬覆盖发生在 route_by_llm_led 内部 → 必须 mock LLM 边界（cached_invoke），
# 让真实的路由/覆盖逻辑跑起来（mock 整个 route_by_llm_led 会绕过覆盖）
with mock.patch.object(irl, "cached_invoke", return_value=dict(_led_chitchat)):
    led = irl.route_by_llm_led("成都怎么玩")
    check("route_by_llm_led: 知识型句式推翻闲聊 → tourism",
          led["domain"] == DOMAIN_TOURISM and led["source_policy"] == "kb_then_web")

with mock.patch.object(irl, "cached_invoke", return_value=dict(_led_chitchat)):
    out = irl.route_intent("成都怎么玩", use_llm_first=True)
    check("route_intent: 覆盖生效 → tourism", out["domain"] == DOMAIN_TOURISM)

# ============================================================
# 4. 开关 ON：LLM 无合法域 → legacy 兜底
# ============================================================
print("== 4. 开关 ON 无合法域兜底 ==")
_led_none = {
    "domain": None, "rewritten": "q", "source_policy": "kb",
    "model_tier": "standard", "tools": [], "is_plan": False, "confidence": 0.0,
}
with mock.patch.object(irl, "route_by_llm_led", return_value=dict(_led_none)), \
        mock.patch.object(irl, "_legacy_domain", return_value=DOMAIN_TOURISM) as lg:
    out = irl.route_intent("q", use_llm_first=True)
    check("无合法域 → 回退 legacy", out["domain"] == DOMAIN_TOURISM)
    check("兜底确由 legacy 提供", lg.call_count == 1)

# ============================================================
# 5. 开关 ON：is_plan 硬覆盖
# ============================================================
print("== 5. 开关 ON is_plan 硬覆盖 ==")
_led_tourism = {
    "domain": DOMAIN_TOURISM, "rewritten": "q", "source_policy": "kb_then_web",
    "model_tier": "standard", "tools": [], "is_plan": False, "confidence": 0.5,
}
with mock.patch.object(irl, "route_by_llm_led", return_value=dict(_led_tourism)):
    out = irl.route_intent("帮我规划成都三日游行程", use_llm_first=True)
    check("LLM 漏判但规划信号命中 → is_plan=True", out["is_plan"] is True)

with mock.patch.object(irl, "route_by_llm_led", return_value=dict(_led_tourism)):
    out = irl.route_intent("成都的景点有哪些", use_llm_first=True)
    check("无规划信号 → is_plan 不被误置", out["is_plan"] is False)

# ============================================================
# 6. route_by_llm_led 异常安全
# ============================================================
print("== 6. LLM 异常安全 ==")
with mock.patch.object(irl, "cached_invoke", side_effect=RuntimeError("llm down")):
    led = irl.route_by_llm_led("成都怎么玩")
    check("LLM 异常 → 域为空（交编排层兜底）", led["domain"] is None)
    check("LLM 异常 → rewritten 回退 query", led["rewritten"] == "成都怎么玩")
    check("LLM 异常 → 保守默认字段",
          led["tools"] == [] and led["is_plan"] is False and led["confidence"] == 0.0)

print()
print(f"== intent_route_llm 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
