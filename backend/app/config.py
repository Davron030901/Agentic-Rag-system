"""Central configuration and LLM/provider auto-detection.

Design decisions (senior-engineer assumptions, stated inline):

* Provider is auto-detected from the environment. OpenAI is the documented
  default when both keys are present (see PROVIDER tie-break) because the user
  requested OpenAI as the default path; Gemini remains the free-tier option and
  is auto-selected whenever only ``GOOGLE_API_KEY`` is set.
* Embedding dimension MUST match the Qdrant collection, so we key the collection
  name off the provider (``docs_openai`` vs ``docs_gemini``) and never mix them.
* Qdrant runs embedded/local by default (``QDRANT_URL`` blank). Set it to use a
  hosted instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ProviderProfile:
    """Static facts about a provider that the rest of the app depends on."""

    provider: Provider
    chat_model: str
    vision_model: str
    embedding_model: str
    embedding_dim: int
    collection: str


def _detect_provider() -> Provider:
    """Resolve the active provider from environment variables.

    Rules:
        * ``PROVIDER`` env var, if set, wins (explicit override).
        * else if only one key is present, use that provider.
        * else if both keys are present, prefer OpenAI (documented default).
        * else fail fast with a clear error.
    """
    explicit = os.getenv("PROVIDER", "").strip().lower()
    if explicit:
        try:
            return Provider(explicit)
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"PROVIDER={explicit!r} is invalid; use 'openai' or 'gemini'."
            ) from exc

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))

    if has_openai and has_gemini:
        # Both set, no override -> documented default preference.
        return Provider.OPENAI
    if has_openai:
        return Provider.OPENAI
    if has_gemini:
        return Provider.GEMINI

    raise RuntimeError(
        "No LLM provider configured. Set OPENAI_API_KEY (OpenAI) or "
        "GOOGLE_API_KEY (Gemini free-tier) in your environment / .env file."
    )


def _profile_for(provider: Provider) -> ProviderProfile:
    if provider is Provider.OPENAI:
        return ProviderProfile(
            provider=Provider.OPENAI,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=1536,
            collection="docs_openai",
        )
    return ProviderProfile(
        provider=Provider.GEMINI,
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash"),
        vision_model=os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash"),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
        embedding_dim=768,
        collection="docs_gemini",
    )


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for a single process."""

    profile: ProviderProfile

    # Vector DB
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]
    qdrant_path: str

    # Retrieval / chunking
    chunk_size: int
    chunk_overlap: int
    top_k: int

    # Web search
    tavily_api_key: Optional[str]

    # Agent retry caps (guarantee termination)
    max_regenerations: int
    max_web_escalations: int

    # CORS
    cors_allow_origins: list[str]

    @property
    def provider(self) -> Provider:
        return self.profile.provider

    @property
    def collection(self) -> str:
        return self.profile.collection

    @property
    def embedding_dim(self) -> int:
        return self.profile.embedding_dim

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings, resolved once and cached."""
    profile = _profile_for(_detect_provider())

    origins = [
        o.strip()
        for o in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    return Settings(
        profile=profile,
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_path=os.getenv("QDRANT_PATH", "./data/qdrant"),
        chunk_size=_int_env("CHUNK_SIZE", 1000),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 150),
        top_k=_int_env("TOP_K", 4),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        max_regenerations=_int_env("MAX_REGENERATIONS", 2),
        max_web_escalations=_int_env("MAX_WEB_ESCALATIONS", 1),
        cors_allow_origins=origins,
    )
