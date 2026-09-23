# -*- coding: utf-8 -*-
"""规划槽位收集服务（第一步：对话内反问）。

背景（2026-09-14 主人拍板方案 A·第一步）：
    用户在对话里说"我想去杭州，帮我规划一下"时，系统应当先一次性问全
    「目的地 / 出发地 / 天数 / 偏好 / 预算」，而不是不问就硬生成一份行程。

边界（刻意为之）：
    本模块只做「判断缺什么 → 生成反问文案 → 记住已问过」三件事，
    **不碰行程引擎**（第二步才换成 trip_plan 智能体），因此刻意**不做 TripRequest 校验**——
    app/shared/schemas/trip_plan.py 里 start_date/end_date/travel_days 均为必填，
    直接复用会把"缺项场景"变成抛错，与"先反问"的目标正好相反。

规则（主人拍板）：
    1. 默认反问一次：缺项数 >= PLAN_SLOT_CLARIFY_MIN_MISSING 才反问；
       默认 1（五项未齐就问）；设 5 等于"仅五项全缺才反问"（保守门槛）。
    2. 同一会话最多反问一次：pending 标记存在 → 直接放行，绝不连环追问。
    3. 抽取失败一律放行（fail-open）：宁可漏问，不可误拦既有规划链路。
    4. 存储照抄 user_memory.py 的「Redis 优先 + 文件回退 + 全异常安全」范式。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm import llm_provider
from app.shared.config.common import env_bool, env_int, env_str
from app.shared.runtime.llm_cache import cached_invoke
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

# 槽位定义：(判定键, 中文标签, 反问文案里的提示语)
PLAN_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("destination", "目的地", "想去哪个城市/地区（例如：杭州）"),
    ("origin", "出发地", "从哪里出发（例如：成都；人已经在当地也可以直说）"),
    ("days", "游玩天数", "打算玩几天（例如：3 天）"),
    ("preference", "旅行偏好", "偏好或同行情况（例如：喜欢自然风光、带孩子、节奏别太赶）"),
    ("budget", "预算", "总预算大概多少（例如：预算 5000 元；没有就直说）"),
)

_PENDING_PREFIX = "pslot:"
_PENDING_TTL_SECONDS = 1800  # 30 分钟：一次规划信息收集窗口
_PROJ = Path(__file__).resolve().parents[3]
_FILE_DIR = _PROJ / "logs" / "plan_slots"


# ============================================================
# 开关（默认 OFF，遵守"主链路新模块特性开关默认 OFF"纪律）
# ============================================================
def plan_slot_clarify_enabled() -> bool:
    """规划槽位反问总开关。OFF 时闸门恒放行，行为与接线前完全一致。"""
    return env_bool("PLAN_SLOT_CLARIFY_ENABLED", False)


def plan_slot_min_missing() -> int:
    """反问阈值：缺项数 >= 该值才反问。

    1（默认）= 五项未齐就问 —— 主人要的"默认反问一次"；
    4        = 缺 4 项及以上才问 —— 保守门槛（评测友好，不动既有规划用例走向）。
    """
    value = env_int("PLAN_SLOT_CLARIFY_MIN_MISSING", 1)
    return min(max(value, 1), len(PLAN_SLOTS))


# ============================================================
# pending 标记存储（Redis 优先 + 文件回退；只存"已反问过"的时间戳）
# 说明：槽位本身**不单独存储**——下一轮的历史会由
# attraction_confirm_service.load_history 回填 state["history"]，
# 行程生成时自然能看到"目的地 + 补充信息"，避免双份存储导致不一致。
# ============================================================
class _RedisPendingBackend:
    def __init__(self, client: Any):
        self.c = client

    def get(self, session_id: str) -> Optional[float]:
        raw = self.c.get(_PENDING_PREFIX + session_id)
        return float(raw) if raw else None

    def set(self, session_id: str, ts: float) -> None:
        self.c.set(_PENDING_PREFIX + session_id, str(ts), ex=_PENDING_TTL_SECONDS)

    def delete(self, session_id: str) -> None:
        self.c.delete(_PENDING_PREFIX + session_id)


class _FilePendingBackend:
    def __init__(self, d: Path):
        self.d = d
        self.d.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_")[:64]
        return self.d / f"{safe}.json"

    def get(self, session_id: str) -> Optional[float]:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            ts = float(rec.get("ts", 0))
        except Exception:  # noqa: BLE001
            return None
        if time.time() - ts > _PENDING_TTL_SECONDS:
            try:
                p.unlink()
            except Exception:  # noqa: BLE001
                pass
            return None
        return ts

    def set(self, session_id: str, ts: float) -> None:
        self._path(session_id).write_text(
            json.dumps({"ts": ts}, ensure_ascii=False), encoding="utf-8"
        )

    def delete(self, session_id: str) -> None:
        p = self._path(session_id)
        if p.exists():
            try:
                p.unlink()
            except Exception:  # noqa: BLE001
                pass


def _init_pending_backend():
    if env_str("CACHE_BACKEND", "memory").lower() == "redis":
        try:
            import redis  # 延迟导入

            client = redis.Redis.from_url(
                env_str("REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True
            )
            client.ping()
            return _RedisPendingBackend(client)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"plan_slot_service: Redis 不可用，回退文件存储：{e}")
    return _FilePendingBackend(_FILE_DIR)


_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _init_pending_backend()
    return _BACKEND


def mark_plan_pending(session_id: str) -> None:
    """标记"本会话已反问过规划信息"，用于保证同一会话最多追问一次。"""
    if not session_id:
        return
    try:
        _backend().set(session_id, time.time())
    except Exception as e:  # noqa: BLE001
        logger.warning(f"plan_slot_service: 写 pending 标记失败（不影响主链路）：{e}")


def is_plan_pending(session_id: str) -> bool:
    """本会话是否正处于"刚问过规划信息、等用户补充"的状态。"""
    if not session_id:
        return False
    try:
        return _backend().get(session_id) is not None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"plan_slot_service: 读 pending 标记失败（按未挂起处理）：{e}")
        return False


def clear_plan_pending(session_id: str) -> None:
    """清除 pending 标记（用户在补信息的那一轮调用，保证不连环追问）。"""
    if not session_id:
        return
    try:
        _backend().delete(session_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"plan_slot_service: 清 pending 标记失败：{e}")


# ============================================================
# 抽取与判定
# ============================================================
def normalize_plan_flags(raw: Any) -> dict:
    """把模型返回规整为 {槽位键: bool}；缺字段/类型异常一律按 True（fail-open）。"""
    flags: dict[str, bool] = {}
    data = raw if isinstance(raw, dict) else {}
    for key, _label, _hint in PLAN_SLOTS:
        value = data.get(key, True)
        if isinstance(value, str):
            value = value.strip().lower() in ("true", "1", "yes", "y")
        flags[key] = bool(value)
    return flags


@step_log("extract_plan_slots")
def extract_plan_slots(query: str, history_text: str = "") -> dict:
    """判断五项规划信息是否已给出。

    Returns:
        dict: `{destination/origin/days/preference: bool}`。
        **任何异常都返回四项全 True**（视为信息齐全 → 闸门放行），
        保证抽取环节永不把既有规划链路拦死。
    """
    all_present = {key: True for key, _label, _hint in PLAN_SLOTS}
    query = (query or "").strip()
    if not query:
        return all_present

    try:
        def _call_model() -> dict:
            # 先渲染 Prompt 再建客户端：Prompt 缺失等配置问题可最快暴露，
            # 也让 fail-open 分支不产生任何多余的客户端构造开销。
            prompt = load_prompt(
                "tourism/plan_slot_extract",
                history_text=history_text or "（无）",
                query=query,
            )
            client = llm_provider.chat(json_mode=True)
            messages = [
                SystemMessage(content="你是旅行规划信息核对助手，只判断信息是否已给出，不做推测。"),
                HumanMessage(content=prompt),
            ]
            return (client | JsonOutputParser()).invoke(messages)

        raw = cached_invoke(
            namespace="plan_slot_extract",
            cache_parts=(query, history_text or ""),
            producer=_call_model,
            cache_label="规划槽位抽取",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"规划槽位抽取失败，按信息齐全放行（fail-open）：{e}")
        return all_present

    return normalize_plan_flags(raw)


def missing_plan_slots(flags: dict) -> list[str]:
    """返回缺失槽位键列表（保持 PLAN_SLOTS 定义顺序）。"""
    return [key for key, _label, _hint in PLAN_SLOTS if not flags.get(key, True)]


def plan_clarify_required(flags: dict) -> bool:
    """是否触发反问：缺项数达到阈值（默认 1，即五项未齐就问）。"""
    return len(missing_plan_slots(flags)) >= plan_slot_min_missing()


def build_plan_clarify_answer(missing_keys: list[str]) -> str:
    """确定性反问文案（不调模型：快、稳、零成本，也不受 429 影响）。"""
    labels = {key: label for key, label, _hint in PLAN_SLOTS}
    hints = {key: hint for key, _label, hint in PLAN_SLOTS}
    wanted = [k for k in missing_keys if k in labels] or [k for k, _l, _h in PLAN_SLOTS]
    bullets = "\n".join(f"- {labels[k]}：{hints[k]}" for k in wanted)
    missing_text = "、".join(labels[k] for k in wanted)
    return (
        "可以，这就为您安排。为了把行程排得更贴合，麻烦再补充以下信息：\n"
        f"{bullets}\n\n"
        "您可以直接一句话说清（例如“从成都出发，玩 3 天，喜欢自然风光，节奏别太赶”）。"
        f"如果暂时只确定了{missing_text}中的一部分，也可以先发来，我会按已有信息先出一版行程。"
    )
