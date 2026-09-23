# -*- coding: utf-8 -*-
"""G·规划槽位反问闸门 · 离线单测（零外部服务、零 LLM 调用、零 Redis）。

覆盖 app/rag/tourism_query/plan_slot_service.py：
  1. normalize_plan_flags：模型返回规整（缺字段 / 字符串 / 非 dict 一律 fail-open 为 True）
  2. missing_plan_slots + plan_clarify_required：默认阈值 1（五项未齐就问）与保守阈值 4（缺 4 项及以上才问）
  3. build_plan_clarify_answer：只列缺项、含补充引导（确定性文案，不调模型）
  4. pending 标记：mark / is / clear 往返 + 空 session 安全
  5. extract_plan_slots：空问句、Prompt 异常 → 一律 fail-open 返回五项 True（绝不误拦既有规划链路）
  6. 开关：PLAN_SLOT_CLARIFY_ENABLED 解析（默认 OFF）

槽位口径：2026-09-16 由四项扩为五项（新增 budget/预算），
故本文件所有期望值均按 ps.PLAN_SLOTS 自适应构造，避免再次扩项时断言过期。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 控制台为 GBK 时，✅/❌ 等符号会让 print 抛 UnicodeEncodeError（门禁外直接跑时的高频坑）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 强制 memory/文件后端：单测不依赖 Redis 可达性（llm_cache 与本服务都在 import 期读取该变量）
os.environ["CACHE_BACKEND"] = "memory"
os.environ.pop("PLAN_SLOT_CLARIFY_MIN_MISSING", None)

from app.rag.tourism_query import plan_slot_service as ps  # noqa: E402

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


ALL_TRUE = {key: True for key, _l, _h in ps.PLAN_SLOTS}
SLOT_N = len(ps.PLAN_SLOTS)

# ============================================================
# 1. normalize_plan_flags
# ============================================================
print("== 1. normalize_plan_flags ==")
check("五项全 True 原样", ps.normalize_plan_flags(
    {key: True for key, _l, _h in ps.PLAN_SLOTS}
) == ALL_TRUE)
# 只对「显式给出的 4 键」校验取值；未给出的键（如 budget）按 fail-open 视为 True
_expect_str = {key: True for key, _l, _h in ps.PLAN_SLOTS}
_expect_str.update({"destination": True, "origin": False, "days": True, "preference": False})
check("字符串真值解析（true/False/1/no）", ps.normalize_plan_flags(
    {"destination": "true", "origin": "False", "days": "1", "preference": "no"}
) == _expect_str)
_expect_missing = {key: True for key, _l, _h in ps.PLAN_SLOTS}
_expect_missing.update({"destination": False})
check("缺字段 → fail-open True", ps.normalize_plan_flags({"destination": False}) == _expect_missing)
check("非 dict（None）→ 全 True", ps.normalize_plan_flags(None) == ALL_TRUE)
check("非 dict（列表）→ 全 True", ps.normalize_plan_flags(["x"]) == ALL_TRUE)

# ============================================================
# 2. 缺项判定与阈值
# ============================================================
print("== 2. 缺项判定与阈值 ==")
flags_none = {"destination": False, "origin": False, "days": False, "preference": False}
flags_only_dest = {"destination": True, "origin": False, "days": False, "preference": False}
flags_three = {"destination": True, "origin": True, "days": True, "preference": False}

check("仅目的地 → 缺 出发地/天数/偏好", ps.missing_plan_slots(flags_only_dest)
      == ["origin", "days", "preference"])
check("五项全齐 → 无缺项", ps.missing_plan_slots(ALL_TRUE) == [])

# 默认阈值 1：五项未齐就问（主人选定"默认反问一次"）
check("默认阈值 = 1", ps.plan_slot_min_missing() == 1)
check("默认：只给目的地 → 反问", ps.plan_clarify_required(flags_only_dest) is True)
check("默认：只缺偏好 → 也反问", ps.plan_clarify_required(flags_three) is True)
check("默认：五项全齐 → 不反问", ps.plan_clarify_required(ALL_TRUE) is False)

# 保守门槛 4：缺 4 项及以上才问（四项全缺时预算未给出 → fail-open 视为齐备，故恰为 4 项）
os.environ["PLAN_SLOT_CLARIFY_MIN_MISSING"] = "4"
check("保守阈值 = 4", ps.plan_slot_min_missing() == 4)
check("保守：四项全缺 → 反问", ps.plan_clarify_required(flags_none) is True)
check("保守：只给目的地 → 不反问", ps.plan_clarify_required(flags_only_dest) is False)
check("保守：只缺偏好 → 不反问", ps.plan_clarify_required(flags_three) is False)

# 越界值收敛到 [1, 槽位数]
os.environ["PLAN_SLOT_CLARIFY_MIN_MISSING"] = "0"
check("阈值 0 收敛为 1", ps.plan_slot_min_missing() == 1)
os.environ["PLAN_SLOT_CLARIFY_MIN_MISSING"] = "99"
check(f"阈值 99 收敛为槽位数 {SLOT_N}", ps.plan_slot_min_missing() == SLOT_N)
os.environ["PLAN_SLOT_CLARIFY_MIN_MISSING"] = "abc"
check("非法值 → 回落默认 1", ps.plan_slot_min_missing() == 1)
os.environ.pop("PLAN_SLOT_CLARIFY_MIN_MISSING", None)

# ============================================================
# 3. 反问文案
# ============================================================
print("== 3. 反问文案 ==")
ans = ps.build_plan_clarify_answer(["origin", "preference"])
check("含缺失项标签「出发地」", "出发地" in ans)
check("含缺失项标签「旅行偏好」", "旅行偏好" in ans)
check("不含已具备项（目的地）条目", "目的地：" not in ans)
check("含缺失项清单引导", "中的一部分" in ans)
check("空列表兜底 → 全部槽位全列", all(lbl in ps.build_plan_clarify_answer([]) for _k, lbl, _h in ps.PLAN_SLOTS))
check("全列时含「目的地」条目", "目的地：" in ps.build_plan_clarify_answer([]))

# ============================================================
# 4. pending 标记（文件后端）
# ============================================================
print("== 4. pending 标记 ==")
sid = "test_plan_slot_pending_0001"
ps.clear_plan_pending(sid)
check("初始未挂起", ps.is_plan_pending(sid) is False)
ps.mark_plan_pending(sid)
check("mark 后已挂起", ps.is_plan_pending(sid) is True)
ps.clear_plan_pending(sid)
check("clear 后已解除", ps.is_plan_pending(sid) is False)
check("空 session → 视为未挂起", ps.is_plan_pending("") is False)
ps.mark_plan_pending("")
ps.clear_plan_pending("")
check("空 session 读写不抛异常", True)

# ============================================================
# 5. extract_plan_slots fail-open
# ============================================================
print("== 5. extract_plan_slots（fail-open）==")
check("空问句 → 五项 True（放行）", ps.extract_plan_slots("", "") == ALL_TRUE)

_orig_load_prompt = ps.load_prompt


def _boom(*_a, **_k):
    raise RuntimeError("prompt file missing")


ps.load_prompt = _boom
try:
    got = ps.extract_plan_slots("帮我规划一下杭州", "（无）")
finally:
    ps.load_prompt = _orig_load_prompt
check("Prompt 异常 → 五项 True（绝不误拦）", got == ALL_TRUE)

# ============================================================
# 6. 开关解析
# ============================================================
print("== 6. 开关解析 ==")
os.environ.pop("PLAN_SLOT_CLARIFY_ENABLED", None)
check("未设置 → 默认 OFF", ps.plan_slot_clarify_enabled() is False)
os.environ["PLAN_SLOT_CLARIFY_ENABLED"] = "true"
check("true → ON", ps.plan_slot_clarify_enabled() is True)
os.environ["PLAN_SLOT_CLARIFY_ENABLED"] = "False"
check("False → OFF", ps.plan_slot_clarify_enabled() is False)
os.environ["PLAN_SLOT_CLARIFY_ENABLED"] = "on"
check("on → ON（兼容写法）", ps.plan_slot_clarify_enabled() is True)
os.environ.pop("PLAN_SLOT_CLARIFY_ENABLED", None)

print()
print(f"== plan_slot_gate 单测结果：{PASS}/{PASS + FAIL} ==")
if FAIL:
    print(f"存在失败 ❌ ({FAIL})")
    raise SystemExit(1)
print("全部通过 ✅")
