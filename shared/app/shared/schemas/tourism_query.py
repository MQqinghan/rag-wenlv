"""
统一查询服务接口数据模型（文旅域视图）。
本模块直接定义统一查询服务所需的全部 schema。
"""
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求体"""
    query: str = Field(..., description="用户查询内容")
    is_stream: bool = Field(default=False, description="是否使用SSE流式输出")
    session_id: str | None = Field(default=None, description="会话ID，为空时自动生成")
    user_id: str | None = Field(default=None, description="用户ID，用于长期记忆(user_memory)，可选")


class QueryResponse(BaseModel):
    """查询响应体"""
    message: str = Field(..., description="处理状态说明")
    session_id: str = Field(..., description="当前会话ID")
    answer: str = Field(default="", description="最终答案（流式模式为空，通过SSE推送）")
    done_list: list[str] = Field(default_factory=list, description="已完成节点列表（中文展示名）")
    image_urls: list[str] = Field(default_factory=list, description="答案配图链接列表")
    domain: str = Field(default="tourism", description="本次路由到的领域（tourism/chitchat）")
    map_data: dict = Field(
        default_factory=dict,
        description="行程地图数据（行程类回答才有）：{ok,name,city,days:[{index,label,attractions:[{name,lon,lat,address}]}]}",
    )


class HistoryItem(BaseModel):
    """单条历史记录"""
    _id: str = ""
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    map_data: dict = Field(default_factory=dict, description="历史回放用的行程地图数据（非行程类为空）")
    ts: Any = None


class HistoryResponse(BaseModel):
    """历史记录响应体"""
    session_id: str
    items: list[HistoryItem] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """单个历史会话摘要（历史会话管理页用）"""
    session_id: str = Field(default="", description="会话ID")
    title: str = Field(default="", description="会话标题（首条用户提问，截断展示）")
    preview: str = Field(default="", description="最后一条消息预览")
    message_count: int = Field(default=0, description="会话内消息条数")
    last_ts: Any = Field(default=None, description="最后活跃时间戳（秒）")


class SessionListResponse(BaseModel):
    """历史会话列表响应体"""
    sessions: list[SessionSummary] = Field(default_factory=list)
