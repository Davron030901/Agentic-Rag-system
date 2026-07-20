"""Provider-agnostic factories for chat, vision, and embedding models.

Every other module imports models from here so provider auto-detection lives in
exactly one place. Models are cached so the whole process shares one client.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .config import Provider, get_settings


@lru_cache(maxsize=1)
def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Return the chat/reasoning model for the active provider.

    Temperature defaults to 0 for deterministic graders; the generation node
    may request a slightly higher temperature on regeneration attempts.
    """
    settings = get_settings()
    if settings.provider is Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.profile.chat_model, temperature=temperature)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.profile.chat_model, temperature=temperature
    )


def build_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Uncached variant used when a distinct temperature is required."""
    settings = get_settings()
    if settings.provider is Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.profile.chat_model, temperature=temperature)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.profile.chat_model, temperature=temperature
    )


@lru_cache(maxsize=1)
def get_vision_model() -> BaseChatModel:
    """Return a vision-capable chat model for image captioning."""
    settings = get_settings()
    if settings.provider is Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.profile.vision_model, temperature=0.0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.profile.vision_model, temperature=0.0
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the embedding model for the active provider.

    NOTE: the embedding dimension is fixed per provider and MUST match the
    Qdrant collection; see ``config.ProviderProfile``.
    """
    settings = get_settings()
    if settings.provider is Provider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.profile.embedding_model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.profile.embedding_model}"
    )
