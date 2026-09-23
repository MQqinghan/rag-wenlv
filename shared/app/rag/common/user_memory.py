# -*- coding: utf-8 -*-
"""F·Context 长期记忆：按 user_id 持久化跨会话画像与兴趣。

设计（对齐《框架》Context 维度「任务状态 ≠ 流程状态 ≠ 长期记忆」）：
- 长期记忆独立于单次对话 state，按 user_id 落库，跨会话/跨设备可恢复。
- 后端：CACHE_BACKEND=redis 时走 Redis（与 LLM 缓存同机）；否则回退本地 JSON 文件，零外部依赖。
- 全部异常安全：存储失败不影响主链路。

存储结构（单 key 聚合，避免散列膨胀）：
  umem:{user_id} -> {
    "profile": {"name":..., "city":..., "preferred_transport":..., "notes":...},
    "interests": {"tourism_dest": [...]},
    "history_itineraries": [{"city":"北京","days":3,"ts":"..."}],
    "updated_at": "..."
  }
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from app.shared.config.common import env_str
from app.shared.runtime.logger import logger

_UMEM_PREFIX = "umem:"
_REDIS_URL = env_str("REDIS_URL", default="redis://127.0.0.1:6379/0")
_PROJ = Path(__file__).resolve().parents[3]
_FILE_DIR = _PROJ / "logs" / "user_memory"


class _RedisBackend:
    def __init__(self, client):
        self.c = client

    def get(self, user_id: str) -> Optional[dict]:
        raw = self.c.get(_UMEM_PREFIX + user_id)
        return json.loads(raw) if raw else None

    def set(self, user_id: str, data: dict) -> None:
        self.c.set(_UMEM_PREFIX + user_id, json.dumps(data, ensure_ascii=False))


class _FileBackend:
    def __init__(self, d: Path):
        self.d = d
        self.d.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        safe = user_id.replace("/", "_").replace("\\", "_")[:64]
        return self.d / f"{safe}.json"

    def get(self, user_id: str) -> Optional[dict]:
        p = self._path(user_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, user_id: str, data: dict) -> None:
        self._path(user_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _init_backend():
    if env_str("CACHE_BACKEND", "memory").lower() == "redis":
        try:
            import redis  # 延迟导入

            client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
            client.ping()
            return _RedisBackend(client)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"user_memory: Redis 不可用，回退文件存储：{e}")
    return _FileBackend(_FILE_DIR)


class UserMemoryStore:
    """长期记忆仓库（按 user_id）。"""

    def __init__(self):
        self._backend = _init_backend()

    # ---- 底层聚合读写 ----
    def _load(self, user_id: str) -> dict:
        rec = self._backend.get(user_id)
        if not rec:
            rec = {"profile": {}, "interests": {}, "history_itineraries": [], "updated_at": ""}
        rec.setdefault("profile", {})
        rec.setdefault("interests", {})
        rec.setdefault("history_itineraries", [])
        return rec

    def _save(self, user_id: str, rec: dict) -> None:
        rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._backend.set(user_id, rec)

    # ---- 公开 API ----
    def get_profile(self, user_id: str) -> dict:
        return self._load(user_id)["profile"]

    def update_profile(self, user_id: str, **fields) -> dict:
        rec = self._load(user_id)
        rec["profile"].update({k: v for k, v in fields.items() if v is not None})
        self._save(user_id, rec)
        return rec["profile"]

    def add_interest(self, user_id: str, kind: str, value: str) -> None:
        if not value:
            return
        rec = self._load(user_id)
        lst = rec["interests"].setdefault(kind, [])
        if value not in lst:
            lst.append(value)
        self._save(user_id, rec)

    def get_interests(self, user_id: str, kind: Optional[str] = None) -> list:
        rec = self._load(user_id)
        if kind:
            return rec["interests"].get(kind, [])
        return rec["interests"]

    def record_itinerary(self, user_id: str, city: str, days: int, note: str = "") -> None:
        rec = self._load(user_id)
        rec["history_itineraries"].append({
            "city": city, "days": days, "note": note,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        # 仅保留最近 20 条
        rec["history_itineraries"] = rec["history_itineraries"][-20:]
        self._save(user_id, rec)

    def get_recent_itineraries(self, user_id: str, limit: int = 5) -> list:
        rec = self._load(user_id)
        return rec["history_itineraries"][-limit:]


# 进程内单例
user_memory_store = UserMemoryStore()
