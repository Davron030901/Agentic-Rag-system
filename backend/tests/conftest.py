"""Test configuration.

Sets fake env vars BEFORE importing app modules so provider auto-detection and
settings resolve deterministically. No real network/LLM calls are made — all
LLM/tool functions are monkeypatched in the tests themselves.
"""

from __future__ import annotations

import os

# Deterministic settings for the whole test session.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("PROVIDER", "openai")
os.environ.setdefault("MAX_REGENERATIONS", "2")
os.environ.setdefault("MAX_WEB_ESCALATIONS", "1")
os.environ.setdefault("QDRANT_PATH", "./data/qdrant_test")

import pytest
from langchain_core.documents import Document


def doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=meta)


@pytest.fixture
def good_docs():
    return [
        doc("Paris is the capital of France.", source="geo.pdf", page=1, modality="text"),
        doc("France is in western Europe.", source="geo.pdf", page=2, modality="text"),
    ]
