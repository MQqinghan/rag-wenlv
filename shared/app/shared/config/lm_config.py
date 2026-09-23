"""
LLM 配置模块，负责读取对话模型与视觉模型相关环境变量。

支持多供应商配置：
- DashScope（阿里云百炼）：默认配置，供 VL 视觉模型和 MCP 搜索使用
- 智谱（BigModel）：可选配置，用于替换默认聊天模型

切换聊天模型供应商只需改 .env 中的 LLM_PROVIDER：
  LLM_PROVIDER=zhipu    → 聊天用智谱 ZP_MODEL
  LLM_PROVIDER=dashscope → 聊天用千问 LLM_DEFAULT_MODEL
"""
from dataclasses import dataclass

from app.shared.config.common import env_float, env_str


@dataclass
class LLMConfig:
    # 供应商选择：zhipu / dashscope
    llm_provider: str

    # DashScope 配置（原配置，供 VL 模型和 MCP 使用）
    base_url: str
    api_key: str
    vl_model: str
    llm_model: str  # DashScope 聊天模型名
    llm_temperature: float

    # 智谱配置（新增，可选）
    zhipu_api_key: str = ""
    zhipu_base_url: str = ""  # 原始 URL（可能包含 /chat/completions）
    zhipu_llm_model: str = ""

    # E 模型档位路由（由 B1 输出 model_tier 驱动）。默认三档都落到当前默认模型；
    # 多模型就绪后改 .env 的 LLM_TIER_* 即可分流，业务代码无需改动。
    tier_lite: str = ""
    tier_standard: str = ""
    tier_longctx: str = ""


lm_config = LLMConfig(
    llm_provider=env_str("LLM_PROVIDER", default="dashscope").lower(),
    # DashScope 原配置
    base_url=env_str("OPENAI_BASE_URL"),
    api_key=env_str("OPENAI_API_KEY"),
    vl_model=env_str("VL_MODEL"),
    llm_model=env_str("LLM_DEFAULT_MODEL"),
    llm_temperature=env_float("LLM_DEFAULT_TEMPERATURE"),
    # 智谱新增配置
    zhipu_api_key=env_str("ZHIPUAI_API_KEY"),
    zhipu_base_url=env_str("ZHIPUAI_URL"),
    zhipu_llm_model=env_str("ZP_MODEL"),
    # E 模型档位（默认空串=沿用当前默认模型；多模型就绪后在 .env 填 LLM_TIER_*）
    tier_lite=env_str("LLM_TIER_LITE", default=""),
    tier_standard=env_str("LLM_TIER_STANDARD", default=""),
    tier_longctx=env_str("LLM_TIER_LONGCTX", default=""),
)
