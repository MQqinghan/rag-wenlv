"""
模型提供者模块，统一封装聊天模型、Embedding 与 Reranker 的访问方式。
"""
import time

from langchain_openai import ChatOpenAI

from app.infra.config import infra_config
from app.shared.model import generate_embeddings, get_bge_m3_ef, get_llm_client, get_reranker_model
from app.shared.runtime.logger import logger

# 限流（429）重试的退避间隔（秒）：批量导入多任务并发时，LLM API 可能触发账户速率限制
_RATE_LIMIT_RETRY_DELAYS = (2.0, 4.0, 8.0)


def _is_rate_limit_error(e: Exception) -> bool:
    """判断异常是否为 LLM API 限流（429 / 账户速率限制）。"""
    text = f"{type(e).__name__} {e}"
    return "RateLimit" in type(e).__name__ or "429" in text or "速率限制" in text or "1302" in text


def invoke_llm_with_retry(chain, messages, max_retries: int = 3):
    """
    调用 LLM 链，遇限流（429）按指数退避重试。

    :param chain: langchain 可调用链（如 llm | StrOutputParser()）
    :param messages: 输入消息
    :param max_retries: 限流时的最大重试次数
    :return: LLM 输出结果
    """
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(messages)
        except Exception as e:
            last_err = e
            if attempt >= max_retries or not _is_rate_limit_error(e):
                raise
            delay = _RATE_LIMIT_RETRY_DELAYS[min(attempt, len(_RATE_LIMIT_RETRY_DELAYS) - 1)]
            logger.warning(f"LLM 调用触发限流（第{attempt + 1}次），{delay}s 后重试：{type(e).__name__}")
            time.sleep(delay)
    raise last_err


class LLMProvider:
    def chat(self, model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
        """
        获取聊天模型客户端。

        Args:
            model: 可选模型名；为空时使用默认聊天模型。
            json_mode: 是否启用 JSON 输出模式，适用于结构化抽取场景。

        Returns:
            ChatOpenAI: 可直接调用或流式调用的聊天模型客户端。
        """
        return get_llm_client(model=model, json_mode=json_mode)

    def chat_by_tier(self, tier: str, json_mode: bool = False) -> ChatOpenAI:
        """
        E 模型档位路由：由 B1 输出的 model_tier 选择具体模型。

        tier ∈ {lite, standard, longctx}；未配置具体模型时回落到默认模型，
        因此不配置 LLM_TIER_* 时行为与 chat() 完全一致（零风险）。
        """
        tier_models = {
            "lite": infra_config.llm.tier_lite,
            "standard": infra_config.llm.tier_standard,
            "longctx": infra_config.llm.tier_longctx,
        }
        model = tier_models.get(tier, infra_config.llm.tier_standard) or None
        return get_llm_client(model=model, json_mode=json_mode)

    def vision_chat(self) -> ChatOpenAI:
        """
        获取视觉模型客户端。

        Returns:
            ChatOpenAI: 面向图片理解场景的视觉模型客户端。
        """
        return get_llm_client(model=infra_config.llm.vl_model)

    def embedding_model(self):
        """
        获取 Embedding 模型对象。

        Returns:
            Any: BGE-M3 Embedding 模型实例。
        """
        return get_bge_m3_ef()

    def reranker_model(self):
        """
        获取重排模型对象。
        Returns:
            Any: 可对问答对进行相关性打分的重排模型实例。
        """
        return get_reranker_model()

    def embed_documents(self, texts: list[str]) -> dict:
        """
        为文本列表生成向量表示。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            dict: 同时包含稠密向量与稀疏向量的结果字典。
        """
        return generate_embeddings(texts)


llm_provider = LLMProvider()
