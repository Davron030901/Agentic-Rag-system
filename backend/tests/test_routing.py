"""Unit tests for the pure conditional-edge routing functions."""

from __future__ import annotations

from langgraph.graph import END

from app import graph
from app.graph import route_after_grade_documents, route_after_grade_generation


def test_route_after_grade_documents_relevant_goes_generate():
    state = {"documents": [{"page_content": "x", "metadata": {}}], "web_escalation_count": 0}
    assert route_after_grade_documents(state) == "generate"


def test_route_after_grade_documents_empty_goes_web_when_allowed():
    state = {"documents": [], "web_escalation_count": 0}
    assert route_after_grade_documents(state) == "web_search"


def test_route_after_grade_documents_empty_caps_to_generate():
    # Web escalation already used up (max=1) -> best-effort generate.
    state = {"documents": [], "web_escalation_count": 1}
    assert route_after_grade_documents(state) == "generate"


def test_route_after_grade_generation_reads_route():
    assert route_after_grade_generation({"_route": graph.ROUTE_GENERATE}) == "generate"
    assert route_after_grade_generation({"_route": graph.ROUTE_WEB}) == "web_search"
    assert route_after_grade_generation({"_route": graph.ROUTE_END}) == END
    # Missing route defaults to END (safe termination).
    assert route_after_grade_generation({}) == END
