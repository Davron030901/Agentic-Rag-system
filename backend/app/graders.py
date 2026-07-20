"""LLM graders that drive control flow — all use structured output (YesNo).

Never parse free-text yes/no. Each grader binds the chat model to the ``YesNo``
schema via ``with_structured_output`` and returns a plain bool.

These are thin wrappers so nodes can be unit-tested by monkeypatching the
grader functions with deterministic fakes.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .llms import get_chat_model
from .prompts import (
    ANSWER_GRADER_SYSTEM,
    ANSWER_GRADER_USER,
    DOC_GRADER_SYSTEM,
    DOC_GRADER_USER,
    GROUNDED_GRADER_SYSTEM,
    GROUNDED_GRADER_USER,
)
from .schemas import YesNo


def _yes(system: str, user: str) -> YesNo:
    """Invoke the chat model with structured YesNo output."""
    model = get_chat_model(temperature=0.0).with_structured_output(YesNo)
    result = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    # with_structured_output returns a YesNo instance; guard for dict fallbacks.
    if isinstance(result, YesNo):
        return result
    return YesNo(**result)  # type: ignore[arg-type]


def grade_document_relevance(question: str, document: Document) -> bool:
    """True if the document is relevant to the question."""
    res = _yes(
        DOC_GRADER_SYSTEM,
        DOC_GRADER_USER.format(document=document.page_content, question=question),
    )
    return res.binary_score == "yes"


def grade_groundedness(documents: List[Document], generation: str) -> bool:
    """True if the generation is supported by (grounded in) the documents."""
    joined = "\n\n---\n\n".join(d.page_content for d in documents)
    res = _yes(
        GROUNDED_GRADER_SYSTEM,
        GROUNDED_GRADER_USER.format(documents=joined, generation=generation),
    )
    return res.binary_score == "yes"


def grade_answer_relevance(question: str, generation: str) -> bool:
    """True if the generation actually resolves the question."""
    res = _yes(
        ANSWER_GRADER_SYSTEM,
        ANSWER_GRADER_USER.format(question=question, generation=generation),
    )
    return res.binary_score == "yes"
