"""Unit tests for each graph node in isolation (mocked LLM/tools)."""

from __future__ import annotations

from app import graders, generation as gen, graph, tools
from app.graph import (
    grade_documents,
    grade_generation,
    generate,
    retrieve,
    web_search,
)
from tests.conftest import doc


def test_retrieve_populates_documents_and_step(monkeypatch):
    monkeypatch.setattr(graph, "retrieve_documents", lambda q: [doc("hello", source="a.pdf")])
    out = retrieve({"question": "hi", "steps": []})
    assert out["steps"] == ["retrieve"]
    assert out["documents"][0]["page_content"] == "hello"


def test_grade_documents_drops_irrelevant(monkeypatch):
    # Keep only documents containing "keep".
    monkeypatch.setattr(
        graders,
        "grade_document_relevance",
        lambda q, d: "keep" in d.page_content,
    )
    state = {
        "question": "q",
        "steps": ["retrieve"],
        "documents": [
            {"page_content": "keep this", "metadata": {}},
            {"page_content": "drop that", "metadata": {}},
        ],
    }
    out = grade_documents(state)
    assert len(out["documents"]) == 1
    assert out["documents"][0]["page_content"] == "keep this"
    assert out["steps"] == ["retrieve", "grade_documents"]


def test_web_search_merges_and_flags(monkeypatch):
    monkeypatch.setattr(
        tools, "web_search", lambda q, max_results=4: [doc("web fact", url="http://x", modality="web")]
    )
    state = {
        "question": "q",
        "steps": ["retrieve", "grade_documents"],
        "documents": [{"page_content": "existing", "metadata": {}}],
        "web_escalation_count": 0,
    }
    out = web_search(state)
    assert out["web_search_used"] is True
    assert out["web_escalation_count"] == 1
    assert len(out["documents"]) == 2
    assert out["steps"][-1] == "web_search"


def test_generate_sets_generation_and_sources(monkeypatch):
    monkeypatch.setattr(gen, "generate_answer", lambda q, docs, temperature=0.0: "the answer [1]")
    state = {
        "question": "q",
        "steps": [],
        "documents": [{"page_content": "ctx", "metadata": {"source": "a.pdf", "page": 1}}],
        "regenerate_count": 0,
    }
    out = generate(state)
    assert out["generation"] == "the answer [1]"
    assert out["sources"][0]["type"] == "doc"
    assert out["steps"][-1] == "generate"


def test_grade_generation_grounded_and_useful_routes_end(monkeypatch):
    monkeypatch.setattr(graders, "grade_groundedness", lambda docs, g: True)
    monkeypatch.setattr(graders, "grade_answer_relevance", lambda q, g: True)
    out = grade_generation(
        {"question": "q", "generation": "a", "documents": [], "steps": []}
    )
    assert out["_route"] == graph.ROUTE_END
    assert out["low_confidence"] is False


def test_grade_generation_not_grounded_regenerates_then_caps(monkeypatch):
    monkeypatch.setattr(graders, "grade_groundedness", lambda docs, g: False)
    # Below cap -> regenerate.
    out1 = grade_generation(
        {"question": "q", "generation": "a", "documents": [], "steps": [], "regenerate_count": 0}
    )
    assert out1["_route"] == graph.ROUTE_GENERATE
    assert out1["regenerate_count"] == 1
    # At cap (max=2) -> end + low confidence.
    out2 = grade_generation(
        {"question": "q", "generation": "a", "documents": [], "steps": [], "regenerate_count": 2}
    )
    assert out2["_route"] == graph.ROUTE_END
    assert out2["low_confidence"] is True


def test_grade_generation_grounded_not_useful_escalates_then_caps(monkeypatch):
    monkeypatch.setattr(graders, "grade_groundedness", lambda docs, g: True)
    monkeypatch.setattr(graders, "grade_answer_relevance", lambda q, g: False)
    # Web budget available -> web_search.
    out1 = grade_generation(
        {"question": "q", "generation": "a", "documents": [], "steps": [], "web_escalation_count": 0}
    )
    assert out1["_route"] == graph.ROUTE_WEB
    # Web budget exhausted (max=1) -> end + low confidence.
    out2 = grade_generation(
        {"question": "q", "generation": "a", "documents": [], "steps": [], "web_escalation_count": 1}
    )
    assert out2["_route"] == graph.ROUTE_END
    assert out2["low_confidence"] is True
