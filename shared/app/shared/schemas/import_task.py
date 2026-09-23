"""
应用主包 / 接口层 / 数据模型层中的 import_ 模块，负责承载导入接口的通用响应模型。
文旅走 import_server，共用本文件的两个响应模型。
"""
from pydantic import BaseModel


class UploadResponse(BaseModel):
    code: int = 200
    message: str
    task_ids: list[str]
    domain: str = ""  # 用户手动选择的导入领域（tourism）


class ImportStatusResponse(BaseModel):
    code: int = 200
    task_id: str
    status: str | None = None
    done_list: list[str]
    running_list: list[str]
    domain: str = ""  # 实际路由到的导入领域（由前端手动选择传入）
    message: str = ""  # blocked（疑似误导原因）/ failed（失败原因）等提示文本
