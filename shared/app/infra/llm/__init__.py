# app/infra/llm/__init__.py
"""
LLM 基础设施模块

统一对外暴露大模型相关能力，使用方式：
    from app.infra.llm import llm_provider

    llm_provider.chat()
    llm_provider.vision_chat()
    llm_provider.embedding_model()
    llm_provider.reranker_model()
    llm_provider.embed_documents(texts)
"""
from app.infra.llm.providers import LLMProvider, llm_provider, invoke_llm_with_retry

__all__ = [
    "LLMProvider",
    "llm_provider",
    "invoke_llm_with_retry",
]