"""
工具模块，负责提供 lm 相关的辅助能力。

支持多供应商路由：
- 有参调用：按模型名前缀判断（glm*→智谱，其他→DashScope）
- 无参调用：按 LLM_PROVIDER 开关选（zhipu→ZP_MODEL，dashscope→LLM_DEFAULT_MODEL）
"""
import re

from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI

from app.shared.config.lm_config import lm_config
from app.shared.runtime.logger import logger

_DEFAULT_LLM_MODEL = "qwen3-32b"
_DEFAULT_TEMPERATURE = 0.1
_llm_client_cache: dict[tuple[str, bool], ChatOpenAI] = {}


def _normalize_base_url(base_url: str) -> str:
    """规范化 base_url，去掉 LangChain 会自动拼接的路径后缀"""
    return re.sub(r"/(chat/completions|embeddings)/?$", "", base_url)


def _resolve_provider_and_model(
    model: str | None,
) -> tuple[str, str, str, str]:
    """
    解析要使用的 (供应商, 模型名, api_key, base_url)。

    有参调用：model 以 glm 开头 → 智谱，否则 → DashScope
    无参调用：按 LLM_PROVIDER 开关选
    """
    if model:
        # 显式指定了模型名 → 按前缀判断供应商
        is_zhipu = model.lower().startswith("glm")
        target_model = model
    else:
        # 无参 → 按 LLM_PROVIDER 开关选
        is_zhipu = lm_config.llm_provider == "zhipu"
        if is_zhipu:
            target_model = lm_config.zhipu_llm_model or _DEFAULT_LLM_MODEL
        else:
            target_model = lm_config.llm_model or _DEFAULT_LLM_MODEL

    if is_zhipu:
        provider_name = "智谱"
        api_key = lm_config.zhipu_api_key
        base_url = _normalize_base_url(lm_config.zhipu_base_url)
    else:
        provider_name = "DashScope"
        api_key = lm_config.api_key
        base_url = _normalize_base_url(lm_config.base_url)

    return provider_name, target_model, api_key, base_url, is_zhipu


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取带全局缓存的LangChain ChatOpenAI客户端实例。

    :param model: 模型名，有参调用时按前缀自动路由供应商
    :param json_mode: 是否开启JSON输出模式
    """
    provider_name, target_model, api_key, base_url, is_zhipu = _resolve_provider_and_model(model)
    cache_key = (target_model, json_mode)

    # 缓存命中
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    logger.info(f"[LLM客户端] 供应商路由：模型={target_model} → {provider_name}")

    # 配置校验
    if not api_key:
        raise ValueError(f"[LLM客户端] 配置缺失：{provider_name} API Key 未配置")
    if not base_url:
        raise ValueError(f"[LLM客户端] 配置缺失：{provider_name} Base URL 未配置")

    # 组装参数
    extra_body = {} if is_zhipu else {"enable_thinking": False}
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[LLM客户端] 已开启JSON输出模式")

    # 初始化
    try:
        llm_client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or _DEFAULT_TEMPERATURE,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败：{e}") from e

    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，供应商={provider_name}")

    return llm_client
