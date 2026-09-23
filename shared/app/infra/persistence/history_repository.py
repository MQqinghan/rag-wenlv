"""
历史记录仓储模块，统一封装聊天记录的读写操作。
"""
from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    list_sessions,
    save_chat_message,
    update_message_item_names,
)


class HistoryRepository:
    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        return get_recent_messages(session_id, limit=limit)

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """按会话聚合的会话摘要列表（历史会话管理页用）"""
        return list_sessions(limit=limit)

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: list[str] | None = None,
        image_urls: list[str] | None = None,
        message_id: str | None = None,
        domain: str = "",
        is_plan: bool = False,
        map_data: dict | None = None,
    ) -> str:
        return save_chat_message(
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            item_names=item_names,
            image_urls=image_urls,
            message_id=message_id,
            domain=domain,
            is_plan=is_plan,
            map_data=map_data,
        )

    def clear_session(self, session_id: str) -> int:
        return clear_history(session_id)

    def update_item_names(self, ids: list[str], item_names: list[str]) -> int:
        return update_message_item_names(ids, item_names)


history_repository = HistoryRepository()
