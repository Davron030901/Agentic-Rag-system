"""Evaluation harness for the Adaptive Multimodal Agentic RAG Assistant.

Run AFTER ingesting a corpus (see README). It reports the required metrics and
runs the four experiments. This is a REAL harness: it calls the live agent and
LLM-judges, so it needs valid API keys and an ingested collection.

Usage:
    python -m eval.evaluate --questions eval/questions.json --out eval/results.json

Metrics reported:
    * Retrieval hit rate  — % of in-document questions with >=1 relevant top-K chunk
    * Groundedness        — % of answers judged grounded in cited context
    * Answer relevance    — % of answers that resolve the question
    * Refusal correctness — % of out-of-document questions correctly refused/flagged
    * Latency / cost proxy— avg seconds and approx chars (token proxy) per query

Experiments:
    1. Chunk size 500 / 1000 / 2000  -> retrieval hit rate (requires re-ingest)
    2. Top-K 2 / 4 / 8               -> precision vs noise
    3. With vs without doc grading   -> groundedness / relevance delta
    4. With vs without web fallback   -> coverage on out-of-document questions
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from app import graders
from app.config import get_settings
from app.generation import generate_answer
from app.graph import run_agent
from app.retriever import retrieve_documents


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------
def load_questions(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["questions"]


# ---------------------------------------------------------------------------
# Core metrics over the full agent.
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    id: int
    question: str
    in_document: bool
    answer: str
    steps: List[str]
    web_used: bool
    low_confidence: bool
    grounded: bool
    relevant: bool
    refused: bool
    latency_s: float
    answer_chars: int


def _is_refusal(answer: str) -> bool:
    a = answer.lower()
    return "i don't know" in a or "i do not know" in a or "cannot answer" in a


def evaluate_full(questions: List[dict]) -> Dict:
    """Run the full agent on each question and compute headline metrics."""
    records: List[RunRecord] = []
    for q in questions:
        start = time.time()
        result = run_agent(q["question"])
        latency = time.time() - start
        answer = result["answer"]

        # Re-use the LLM graders as judges for the metrics.
        docs = retrieve_documents(q["question"])
        grounded = graders.grade_groundedness(docs, answer) if answer else False
        relevant = graders.grade_answer_relevance(q["question"], answer) if answer else False
        refused = _is_refusal(answer)

        records.append(
            RunRecord(
                id=q["id"],
                question=q["question"],
                in_document=q.get("in_document", True),
                answer=answer,
                steps=result["steps"],
                web_used=result["web_search_used"],
                low_confidence=result["low_confidence"],
                grounded=grounded,
                relevant=relevant,
                refused=refused,
                latency_s=round(latency, 3),
                answer_chars=len(answer),
            )
        )

    in_doc = [r for r in records if r.in_document]
    out_doc = [r for r in records if not r.in_document]

    def pct(sub: List[bool]) -> float:
        return round(100.0 * sum(sub) / len(sub), 1) if sub else 0.0

    metrics = {
        "n_questions": len(records),
        "groundedness_pct": pct([r.grounded for r in records]),
        "answer_relevance_pct": pct([r.relevant for r in records]),
        # For out-of-document questions, "correct" = refused OR answered via web.
        "refusal_correctness_pct": pct(
            [(r.refused or r.web_used) for r in out_doc]
        ),
        "avg_latency_s": round(statistics.mean([r.latency_s for r in records]), 3),
        "avg_answer_chars": round(statistics.mean([r.answer_chars for r in records]), 1),
        "web_fallback_rate_pct": pct([r.web_used for r in records]),
    }
    return {"metrics": metrics, "records": [r.__dict__ for r in records]}


# ---------------------------------------------------------------------------
# Retrieval hit rate (grader-judged relevance of top-K).
# ---------------------------------------------------------------------------
def retrieval_hit_rate(questions: List[dict], k: int) -> float:
    """% of in-document questions with >=1 grader-relevant chunk in top-K."""
    in_doc = [q for q in questions if q.get("in_document", True)]
    hits = 0
    for q in in_doc:
        docs = retrieve_documents(q["question"], k=k)
        if any(graders.grade_document_relevance(q["question"], d) for d in docs):
            hits += 1
    return round(100.0 * hits / len(in_doc), 1) if in_doc else 0.0


# ---------------------------------------------------------------------------
# Experiments.
# ---------------------------------------------------------------------------
def experiment_topk(questions: List[dict], ks=(2, 4, 8)) -> Dict[str, float]:
    """Top-K sweep on retrieval hit rate (no re-ingest needed)."""
    return {f"top_k={k}": retrieval_hit_rate(questions, k) for k in ks}


def experiment_grading_ablation(questions: List[dict]) -> Dict:
    """With vs without document grading -> groundedness/relevance.

    'Without grading' = generate directly from raw top-K (no relevance filter).
    """
    in_doc = [q for q in questions if q.get("in_document", True)]

    def run(with_grading: bool) -> Dict[str, float]:
        grounded, relevant = [], []
        for q in in_doc:
            docs = retrieve_documents(q["question"])
            if with_grading:
                docs = [d for d in docs if graders.grade_document_relevance(q["question"], d)]
            answer = generate_answer(q["question"], docs)
            grounded.append(graders.grade_groundedness(docs, answer) if docs else False)
            relevant.append(graders.grade_answer_relevance(q["question"], answer))
        n = len(in_doc) or 1
        return {
            "groundedness_pct": round(100.0 * sum(grounded) / n, 1),
            "answer_relevance_pct": round(100.0 * sum(relevant) / n, 1),
        }

    return {"with_grading": run(True), "without_grading": run(False)}


def experiment_web_ablation(questions: List[dict]) -> Dict:
    """With vs without web fallback -> coverage on out-of-document questions.

    Coverage = % of out-of-document questions that get a non-refusal answer.
    Without-web coverage is estimated by generating from local docs only.
    """
    out_doc = [q for q in questions if not q.get("in_document", True)]

    # With web fallback: use the full agent.
    with_web = 0
    for q in out_doc:
        r = run_agent(q["question"])
        if not _is_refusal(r["answer"]):
            with_web += 1

    # Without web fallback: local docs only.
    without_web = 0
    for q in out_doc:
        docs = retrieve_documents(q["question"])
        answer = generate_answer(q["question"], docs)
        if not _is_refusal(answer):
            without_web += 1

    n = len(out_doc) or 1
    return {
        "with_web_coverage_pct": round(100.0 * with_web / n, 1),
        "without_web_coverage_pct": round(100.0 * without_web / n, 1),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG evaluation harness")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument(
        "--skip-chunk-experiment",
        action="store_true",
        help="Skip the chunk-size sweep (it requires re-ingesting the corpus).",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"Provider={settings.provider.value} collection={settings.collection} k={settings.top_k}")

    questions = load_questions(args.questions)

    print("\n== Full-agent evaluation ==")
    full = evaluate_full(questions)
    print(json.dumps(full["metrics"], indent=2))

    print("\n== Experiment: retrieval hit rate @ current chunking ==")
    full["metrics"]["retrieval_hit_rate_pct"] = retrieval_hit_rate(questions, settings.top_k)
    print(f"hit_rate={full['metrics']['retrieval_hit_rate_pct']}%")

    print("\n== Experiment 2: Top-K sweep ==")
    topk = experiment_topk(questions)
    print(json.dumps(topk, indent=2))

    print("\n== Experiment 3: document-grading ablation ==")
    grading = experiment_grading_ablation(questions)
    print(json.dumps(grading, indent=2))

    print("\n== Experiment 4: web-fallback ablation ==")
    web = experiment_web_ablation(questions)
    print(json.dumps(web, indent=2))

    report = {
        "provider": settings.provider.value,
        "collection": settings.collection,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "headline_metrics": full["metrics"],
        "experiment_topk": topk,
        "experiment_grading_ablation": grading,
        "experiment_web_ablation": web,
        "records": full["records"],
        "note_chunk_experiment": (
            "To run the chunk-size 500/1000/2000 experiment, re-ingest the corpus "
            "with CHUNK_SIZE set to each value and re-run retrieval_hit_rate; results "
            "recorded manually in README (requires re-embedding)."
        ),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
