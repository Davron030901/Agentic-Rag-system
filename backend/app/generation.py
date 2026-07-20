"""Answer generation grounded in context, plus context/source formatting."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .llms import build_chat_model
from .prompts import GENERATE_SYSTEM, GENERATE_USER
from .schemas import Source


def format_context(documents: List[Document]) -> str:
    """Render documents as numbered context blocks for inline [n] citations."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        if meta.get("modality") == "web":
            label = meta.get("url", "web")
        else:
            src = meta.get("source", "doc")
            page = meta.get("page")
            label = f"{src}" + (f" p.{page}" if page else "")
        blocks.append(f"[{i}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_sources(documents: List[Document]) -> List[Source]:
    """Convert documents into citation Source objects for the API/frontend."""
    sources: List[Source] = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        snippet = doc.page_content[:280]
        if meta.get("modality") == "web":
            url = meta.get("url") or meta.get("source", "")
            sources.append(
                Source(
                    type="web",
                    id=url or f"web-{i}",
                    url=url or None,
                    title=meta.get("title") or url,
                    snippet=snippet,
                )
            )
        else:
            src = meta.get("source", f"doc-{i}")
            page = meta.get("page")
            title = f"{src}" + (f" (p.{page})" if page else "")
            sources.append(
                Source(
                    type="doc",
                    id=f"{src}#{i}",
                    url=None,
                    title=title,
                    snippet=snippet,
                )
            )
    return sources


def generate_answer(question: str, documents: List[Document], temperature: float = 0.0) -> str:
    """Generate an answer strictly grounded in the provided documents.

    Temperature is nudged up on regeneration attempts so a stuck, ungrounded
    answer has a chance to change.
    """
    context = format_context(documents)
    model = build_chat_model(temperature=temperature)
    resp = model.invoke(
        [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=GENERATE_USER.format(context=context, question=question)),
        ]
    )
    return (resp.content or "").strip()
