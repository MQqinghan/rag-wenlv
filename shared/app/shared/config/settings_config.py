"""
应用基础配置模块，负责读取导入服务与查询服务的启动配置。
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# 显式从项目根目录加载 .env，避免因 PyCharm/终端工作目录不同导致环境变量缺失
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # app/shared/config -> 项目根
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass
class AppSettings:
    import_app_name: str = os.getenv("IMPORT_APP_NAME", "Enterprise RAG Import Service")
    query_app_name: str = os.getenv("QUERY_APP_NAME", "Enterprise RAG Query Service")
    app_env: str = os.getenv("APP_ENV", "dev")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    import_app_port: int = int(os.getenv("IMPORT_APP_PORT", "8000"))
    query_app_port: int = int(os.getenv("QUERY_APP_PORT", "8001"))
    cors_origins: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()
    )

settings = AppSettings()