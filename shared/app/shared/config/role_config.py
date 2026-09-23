"""
部署角色（APP_ROLE）与启动期配置自检 —— A5 部署切片（T8）。

背景
----
1. 导入端与查询端**代码层零交叉引用**（静态依赖图实测：导入端→查询侧 0 条、查询端→导入侧 0 条），
   本就是独立进程、独立端口、独立依赖组，因此部署切片的本质是**进程级隔离**而非文件级隔离。
2. 但两端共用 `app/infra/config` 配置聚合器（`providers.py` 一次实例化 8 个配置单例），
   而各配置读取用 `env_str`/`env_bool`，**缺 env 时静默回落空串/默认值不抛错**，
   导致"本端必需配置缺失"要等到运行时才炸，排查成本高。

本模块按角色声明必需/推荐配置，在进程启动时自检，把配置问题提前到启动期暴露。

角色
----
- `import`  导入服务（app/api/http/import_server.py）
- `query`   查询服务（app/api/http/query_server.py）
- `gateway` 移动端 BFF（app/api/http/mobile_gateway.py）
- `all`     本地开发（不做校验；未设 APP_ROLE 时的默认）

严格度
------
- `ROLE_STRICT_CHECK=true`：缺 required 配置直接抛 RuntimeError（容器/CI 部署用，fail-fast）。
- 默认 `false`：仅打 ERROR 日志并在 `/health` 暴露，**不阻断启动**（本地 PyCharm 启动不受影响）。

用法
----
    from app.shared.config.role_config import startup_role_check
    startup_role_check("query")     # 放在入口模块加载处
"""
from __future__ import annotations

import os

from app.shared.config.common import env_bool, env_str
from app.shared.runtime.logger import logger

ROLE_IMPORT = "import"
ROLE_QUERY = "query"
ROLE_GATEWAY = "gateway"
ROLE_ALL = "all"

_VALID_ROLES = (ROLE_IMPORT, ROLE_QUERY, ROLE_GATEWAY, ROLE_ALL)

# ---------------------------------------------------------------------------
# 各角色必需 / 推荐配置清单
# 键名严格取自 .env.example 与实际 .env，未凭印象补写。
# required   ：缺失即功能不可用（strict 模式下阻断启动）
# recommended：缺失则能力降级，仅告警
# ---------------------------------------------------------------------------
ROLE_REQUIRED: dict[str, tuple[str, ...]] = {
    # 导入端：写 Milvus + 传 MinIO + 外发总开关是核心
    ROLE_IMPORT: (
        "MILVUS_URL",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET_NAME",
        "IMPORT_EGRESS_MODE",
    ),
    # 查询端：读 Milvus + 调 LLM + 存历史
    ROLE_QUERY: (
        "MILVUS_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "MONGO_URL",
    ),
    # BFF：转发目标。均有代码内默认值，实际不会缺失，列此仅为显式声明部署契约
    ROLE_GATEWAY: (),
    ROLE_ALL: (),
}

# 只列「缺失即能力降级、且代码内无合理默认值」的键；
# 有默认值可静默兜底的（如 QUERY_HOST=127.0.0.1、IMAGE_SENSITIVE_GUARD=auto）不列，避免噪音告警。
ROLE_RECOMMENDED: dict[str, tuple[str, ...]] = {
    ROLE_IMPORT: (
        "OPENAI_API_KEY",       # 缺失→文旅元数据抽取（VL/LLM）不可用，质量降级
        "MINERU_BASE_URL",      # 缺失→PDF 无法走 MinerU 解析
        "MINERU_API_TOKEN",
    ),
    ROLE_QUERY: (
        "QWEATHER_API_KEY",     # 缺失→天气工具直接不可用
        "AMAP_API_KEY",         # 缺失→路线/POI 能力不可用
        "MCP_DASHSCOPE_BASE_URL",   # 缺失→联网检索（web 路）不可用
    ),
    # 网关当前全部配置均有代码内默认值，暂不列；
    # T12 鉴权落地后需补：WX_APPID / WX_SECRET / JWT_SECRET
    ROLE_GATEWAY: (),
    ROLE_ALL: (),
}


def get_role(default: str = ROLE_ALL) -> str:
    """
    读取部署角色。

    Args:
        default: APP_ROLE 未设置或非法时的回落值（各服务入口传入自己的角色）。

    Returns:
        str: import / query / gateway / all
    """
    role = (os.getenv("APP_ROLE") or "").strip().lower()
    if not role:
        return default
    if role not in _VALID_ROLES:
        logger.warning(
            f"APP_ROLE=[{role}]非法（可选 {_VALID_ROLES}），回落到 [{default}]"
        )
        return default
    return role


def _missing(keys: tuple[str, ...]) -> list[str]:
    """返回当前环境中取值为空的配置键名列表。"""
    return [k for k in keys if not (os.getenv(k) or "").strip()]


def check_role_config(role: str) -> dict:
    """
    按角色检查配置完整性（纯函数，不做日志与异常）。

    Args:
        role: import / query / gateway / all

    Returns:
        dict: {role, missing_required, missing_recommended, ok}
              ok = 无 required 缺失
    """
    if role not in _VALID_ROLES:
        role = ROLE_ALL
    miss_req = _missing(ROLE_REQUIRED.get(role, ()))
    miss_rec = _missing(ROLE_RECOMMENDED.get(role, ()))
    return {
        "role": role,
        "missing_required": miss_req,
        "missing_recommended": miss_rec,
        "ok": not miss_req,
    }


def startup_role_check(default_role: str = ROLE_ALL) -> dict:
    """
    进程启动期角色自检入口。各服务入口模块调用一次。

    - 角色为 all：直接返回，不做校验（本地开发）。
    - 缺 required：ERROR 日志；`ROLE_STRICT_CHECK=true` 时抛 RuntimeError（fail-fast）。
    - 缺 recommended：WARNING 日志，不阻断。

    Args:
        default_role: 本服务的默认角色（APP_ROLE 未设置时使用）。

    Returns:
        dict: check_role_config 的结果，供入口挂 /health 或日志使用。
    """
    role = get_role(default_role)
    result = check_role_config(role)

    if role == ROLE_ALL:
        logger.info("APP_ROLE 未设置，跳过启动期配置自检（本地开发模式）")
        return result

    strict = env_bool("ROLE_STRICT_CHECK", default=False)

    if result["missing_required"]:
        msg = (
            f"[{role}] 启动自检未通过，缺少必需配置: "
            f"{result['missing_required']}"
        )
        if strict:
            logger.error(msg + "（ROLE_STRICT_CHECK=true，阻断启动）")
            raise RuntimeError(msg)
        logger.error(msg + "（当前为告警模式，服务继续启动但功能不可用；"
                           "容器部署建议设 ROLE_STRICT_CHECK=true 做 fail-fast）")
    else:
        logger.info(f"[{role}] 启动自检通过：必需配置齐全")

    if result["missing_recommended"]:
        logger.warning(
            f"[{role}] 缺少推荐配置（对应能力将降级）: "
            f"{result['missing_recommended']}"
        )

    return result
