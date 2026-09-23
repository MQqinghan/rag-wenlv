# -*- coding: utf-8 -*-
"""框架模块接线（D–I 整合）离线单测：验证「开关 OFF 零行为 / ON 生效」。

零外部服务：只校验开关语义、纯函数与注册表，不发起 Milvus / LLM / 高德 调用。
开关经 os.environ 即时切换（env_bool 每次读取 os.getenv）。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infra.llm_harness import resolve_tier
from app.rag.common.memory_harness import maybe_compress_history, memory_block
from app.shared.runtime.action_guard import ActionDeniedError, require_action
from app.shared.runtime.action_registry import registry as action_registry
from app.shared.runtime.data_source_registry import registry as data_registry
from app.shared.runtime.skill_registry import registry as skill_registry

import app.rag.common.memory_harness as mh

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


def _set(key: str, value) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


class _FakeStore:
    def get_profile(self, uid):
        return {"city": "成都", "preferred_transport": "高铁"}

    def get_interests(self, uid):
        return {"tourism_dest": ["都江堰"]}

    def get_recent_itineraries(self, uid, limit=5):
        return [{"city": "成都", "days": 3}]


def main() -> None:
    print("== 1. E 档位解析（纯函数） ==")
    check("合法档位透传", resolve_tier({"route_info": {"model_tier": "longctx"}}) == "longctx")
    check("非法档位回默认", resolve_tier({"route_info": {"model_tier": "xxx"}}, "standard") == "standard")
    check("缺失 route_info 回默认", resolve_tier({}, "standard") == "standard")

    print("== 2. F 长对话压缩（开关语义） ==")
    msgs = [{"role": "user", "content": f"第{i}轮问题" * 40} for i in range(20)]
    _set("CONTEXT_COMPRESSION_ENABLED", None)
    check("OFF 原样返回（同一对象）", maybe_compress_history({}, msgs) is msgs)
    _set("CONTEXT_COMPRESSION_ENABLED", "true")
    out = maybe_compress_history({}, msgs)
    check("ON 触发压缩（产出新列表）", isinstance(out, list) and out is not msgs)
    check("ON 保留首尾（长度不超原）", len(out) <= len(msgs))
    _set("CONTEXT_COMPRESSION_ENABLED", None)

    print("== 3. F 长期记忆注入（开关语义） ==")
    mh.user_memory_store = _FakeStore()
    _set("USER_MEMORY_ENABLED", None)
    check("OFF 返回空串", memory_block({"user_id": "u1"}) == "")
    _set("USER_MEMORY_ENABLED", "true")
    check("ON 但无 user_id → 空串", memory_block({}) == "")
    block = memory_block({"user_id": "u1"})
    check("ON 且有记忆 → 含画像", ("长期画像" in block) and ("成都" in block) and ("都江堰" in block))
    _set("USER_MEMORY_ENABLED", None)

    print("== 4. I Action 风控闸门（开关语义） ==")
    _set("ACTION_GUARD_ENABLED", None)
    ok = True
    try:
        require_action("book_ticket")  # 高危禁用动作
    except Exception:
        ok = False
    check("OFF 时禁用动作也不拦（零行为）", ok)
    _set("ACTION_GUARD_ENABLED", "true")
    passed = True
    try:
        require_action("weather_query")  # 低危已启用
    except Exception:
        passed = False
    check("ON 时低危动作放行", passed)
    denied = False
    try:
        require_action("book_ticket")
    except ActionDeniedError:
        denied = True
    except Exception:
        denied = False
    check("ON 时高危禁用动作被拒", denied)
    denied2 = False
    try:
        require_action("no_such_action")
    except ActionDeniedError:
        denied2 = True
    check("ON 时未登记动作被拒", denied2)
    _set("ACTION_GUARD_ENABLED", None)

    print("== 5. G/H 注册表可消费（暴露前置） ==")
    check("技能目录非空", len(skill_registry.list_all()) >= 1)
    check("数据源目录非空", len(data_registry.list_all()) >= 1)
    check("数据源时效报告可调用", isinstance(data_registry.staleness_report(), list))
    check("动作表含 stay_food_query", action_registry.get("stay_food_query") is not None)
    check("高危动作登记为禁用", action_registry.get("book_ticket") is not None
          and action_registry.get("book_ticket").enabled is False)

    print(f"\n{PASS}/{PASS + FAIL} 通过")
    if FAIL:
        print("全部通过 ❌" if False else f"存在 {FAIL} 项失败 ❌")
        sys.exit(1)
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
