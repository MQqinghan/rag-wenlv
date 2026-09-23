"""
应用主包 / 基础设施层 / 持久化子模块的初始化文件，用于声明包边界与导出约定。
"""
from app.infra.persistence.history_repository import history_repository

__all__ = ["history_repository"]