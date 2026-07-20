"""The LangGraph adaptive-RAG state machine.

Control flow (exactly as specified):

    retrieve --> grade_documents
    grade_documents --(relevant)--> generate
    grade_documents --(weak/none)--> web_search
    web_search --> generate
    generate --> grade_generation
    grade_generation --(grounded + useful)--> END
    grade_generation --(not grounded)--> generate      # regenerate, retry-capped
    grade_generation --(grounded, not useful)--> web_search  # more evidence, retry-capped

``grade_generation`` is a real node (so it shows up as a step pill) that makes
TWO distinct structured judgments (groundedness and answer-relevance) and then
picks the next route, incrementing the bounded retry counters. Hard caps on
``regenerate_count`` and ``web_escalation_count`` guarantee termination: every
back-edge increments a bounded counter, so the graph always reaches END and
falls back to a best-effort answer flagged ``low_confidence``.

The functions ``grade_document_relevance``, ``grade_groundedness``,
``grade_answer_relevance``, ``generate_answer`` and ``web_search`` are imported
at module import; tests monkeypatch them on this module to exercise routing
without any real LLM/tool calls.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from . import graders, generation as gen, tools
from .config import get_settings
from .retriever import retrieve_documents
from .schemas import DocDict, GraphState

# Sentinel route values written by grade_generation and read by its edge.
ROUTE_END = "end"
ROUTE_GENERATE = "generate"
ROUTE_WEB = "web_search"


# ---------------------------------------------------------------------------
# Document <-> dict helpers (state stays JSON-serialisable).
# ---------------------------------------------------------------------------
def _to_dict(doc: Document) -> DocDict:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}


def _to_docs(items: List[DocDict]) -> List[Document]:
    return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in items]


def _append_step(state: GraphState, name: str) -> List[str]:
    return list(state.get("steps", [])) + [name]


# ---------------------------------------------------------------------------
# Nodes.
# ---------------------------------------------------------------------------
def retrieve(state: GraphState) -> dict:
    """Retrieve top-K candidate chunks for the question."""
    docs = retrieve_documents(state["question"])
    return {
        "documents": [_to_dict(d) for d in docs],
        "steps": _append_step(state, "retrieve"),
    }


def grade_documents(state: GraphState) -> dict:
    """LLM-grade each retrieved chunk; drop the irrelevant ones."""
    question = state["question"]
    docs = _to_docs(state.get("documents", []))
    kept = [d for d in docs if graders.grade_document_relevance(question, d)]
    return {
        "documents": [_to_dict(d) for d in kept],
        "steps": _append_step(state, "grade_documents"),
    }


def web_search(state: GraphState) -> dict:
    """Fallback to live web search and merge results into the context."""
    question = state["question"]
    results = tools.web_search(question)
    merged = _to_docs(state.get("documents", [])) + results
    return {
        "documents": [_to_dict(d) for d in merged],
        "web_search_used": True,
        "web_escalation_count": state.get("web_escalation_count", 0) + 1,
        "steps": _append_step(state, "web_search"),
    }


def generate(state: GraphState) -> dict:
    """Generate an answer strictly grounded in the surviving context."""
    question = state["question"]
    docs = _to_docs(state.get("documents", []))
    # Nudge temperature up on later regeneration attempts to escape a bad answer.
    regen = state.get("regenerate_count", 0)
    temperature = 0.0 if regen == 0 else min(0.3 * regen, 0.6)
    answer = gen.generate_answer(question, docs, temperature=temperature)
    sources = [s.model_dump() for s in gen.build_sources(docs)]
    return {
        "generation": answer,
        "sources": sources,
        "steps": _append_step(state, "generate"),
    }


def grade_generation(state: GraphState) -> dict:
    """Two structured judgments -> next route, with retry caps."""
    settings = get_settings()
    question = state["question"]
    docs = _to_docs(state.get("documents", []))
    generation = state.get("generation", "")

    grounded = graders.grade_groundedness(docs, generation)

    regen = state.get("regenerate_count", 0)
    web_esc = state.get("web_escalation_count", 0)
    low_confidence = state.get("low_confidence", False)

    if not grounded:
        # Hallucination -> regenerate from the same context, if budget remains.
        if regen < settings.max_regenerations:
            route = ROUTE_GENERATE
            regen += 1
        else:
            route = ROUTE_END
            low_confidence = True
        useful = None  # not evaluated on this branch
    else:
        useful = graders.grade_answer_relevance(question, generation)
        if useful:
            route = ROUTE_END
        else:
            # Grounded but does not resolve the question -> need more evidence.
            if web_esc < settings.max_web_escalations and settings.web_search_enabled:
                route = ROUTE_WEB
            else:
                route = ROUTE_END
                low_confidence = True

    return {
        "regenerate_count": regen,
        "low_confidence": low_confidence,
        "steps": _append_step(state, "grade_generation"),
        "_route": route,
    }


# ---------------------------------------------------------------------------
# Conditional edges.
# ---------------------------------------------------------------------------
def route_after_grade_documents(state: GraphState) -> str:
    """Relevant docs -> generate; weak/empty -> web_search (if allowed)."""
    settings = get_settings()
    docs = state.get("documents", [])
    web_esc = state.get("web_escalation_count", 0)
    if docs:
        return "generate"
    # No relevant docs survived. Escalate to web search if we still can.
    if web_esc < settings.max_web_escalations and settings.web_search_enabled:
        return "web_search"
    # Otherwise generate a best-effort (will honestly say "I don't know").
    return "generate"


def route_after_grade_generation(state: GraphState) -> str:
    """Read the route computed by the grade_generation node."""
    route = state.get("_route", ROUTE_END)
    if route == ROUTE_GENERATE:
        return "generate"
    if route == ROUTE_WEB:
        return "web_search"
    return END


# ---------------------------------------------------------------------------
# Graph builder.
# ---------------------------------------------------------------------------
def build_graph():
    """Compile and return the adaptive-RAG graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {"generate": "generate", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_grade_generation,
        {"generate": "generate", "web_search": "web_search", END: END},
    )

    return workflow.compile()


def initial_state(question: str) -> GraphState:
    """Build a fresh state for a question with all counters zeroed."""
    return {
        "question": question,
        "documents": [],
        "generation": "",
        "web_search_used": False,
        "steps": [],
        "sources": [],
        "regenerate_count": 0,
        "web_escalation_count": 0,
        "low_confidence": False,
    }


def run_agent(question: str) -> dict:
    """Invoke the graph for a question and return a clean result dict.

    A generous ``recursion_limit`` is set as a belt-and-braces safety net on top
    of the explicit retry caps; the caps should always terminate first.
    """
    graph = build_graph()
    final = graph.invoke(initial_state(question), config={"recursion_limit": 50})
    return {
        "answer": final.get("generation", ""),
        "steps": final.get("steps", []),
        "sources": final.get("sources", []),
        "web_search_used": final.get("web_search_used", False),
        "low_confidence": final.get("low_confidence", False),
    }
