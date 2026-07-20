"""Integration tests: the full compiled graph across the three scenarios.

All LLM/tool boundaries are monkeypatched so routing is exercised deterministically
with zero network calls.
"""

from __future__ import annotations

from app import graders, generation as gen, graph, tools
from tests.conftest import doc


def _patch_common(monkeypatch, *, docs, web_docs=None):
    monkeypatch.setattr(graph, "retrieve_documents", lambda q: docs)
    monkeypatch.setattr(gen, "generate_answer", lambda q, d, temperature=0.0: "answer [1]")
    monkeypatch.setattr(
        tools,
        "web_search",
        lambda q, max_results=4: (web_docs or [doc("web fact", url="http://x", modality="web")]),
    )


def test_happy_path(monkeypatch):
    """Good docs -> generate -> grounded & useful -> END."""
    _patch_common(monkeypatch, docs=[doc("Paris is the capital of France.", source="g.pdf", page=1)])
    monkeypatch.setattr(graders, "grade_document_relevance", lambda q, d: True)
    monkeypatch.setattr(graders, "grade_groundedness", lambda d, g: True)
    monkeypatch.setattr(graders, "grade_answer_relevance", lambda q, g: True)

    result = graph.run_agent("What is the capital of France?")

    assert result["steps"] == ["retrieve", "grade_documents", "generate", "grade_generation"]
    assert result["web_search_used"] is False
    assert result["low_confidence"] is False
    assert result["answer"] == "answer [1]"


def test_web_fallback_path(monkeypatch):
    """Docs graded weak -> web_search -> generate -> grounded & useful -> END."""
    _patch_common(monkeypatch, docs=[doc("irrelevant text", source="g.pdf", page=1)])
    monkeypatch.setattr(graders, "grade_document_relevance", lambda q, d: False)  # all dropped
    monkeypatch.setattr(graders, "grade_groundedness", lambda d, g: True)
    monkeypatch.setattr(graders, "grade_answer_relevance", lambda q, g: True)

    result = graph.run_agent("Out-of-document question?")

    assert "web_search" in result["steps"]
    assert result["web_search_used"] is True
    assert result["steps"].index("web_search") < result["steps"].index("generate")
    assert result["low_confidence"] is False


def test_self_correction_path(monkeypatch):
    """generate -> not grounded -> regenerate -> eventually grounded -> END."""
    _patch_common(monkeypatch, docs=[doc("Paris is the capital of France.", source="g.pdf", page=1)])
    monkeypatch.setattr(graders, "grade_document_relevance", lambda q, d: True)

    calls = {"n": 0}

    def flaky_grounded(d, g):
        calls["n"] += 1
        return calls["n"] >= 2  # first generation ungrounded, second grounded

    monkeypatch.setattr(graders, "grade_groundedness", flaky_grounded)
    monkeypatch.setattr(graders, "grade_answer_relevance", lambda q, g: True)

    result = graph.run_agent("What is the capital of France?")

    # generate appears at least twice (original + one regeneration).
    assert result["steps"].count("generate") >= 2
    assert result["low_confidence"] is False


def test_retry_cap_prevents_infinite_loop(monkeypatch):
    """Force grading to never pass -> graph still terminates, flagged low confidence."""
    _patch_common(monkeypatch, docs=[doc("Paris is the capital of France.", source="g.pdf", page=1)])
    monkeypatch.setattr(graders, "grade_document_relevance", lambda q, d: True)
    monkeypatch.setattr(graders, "grade_groundedness", lambda d, g: False)  # never grounded
    monkeypatch.setattr(graders, "grade_answer_relevance", lambda q, g: True)

    result = graph.run_agent("What is the capital of France?")

    # Terminates (does not raise recursion error) and is flagged.
    assert result["low_confidence"] is True
    # Bounded number of regenerations: original + max_regenerations (2) = 3 generates.
    assert result["steps"].count("generate") == 3
