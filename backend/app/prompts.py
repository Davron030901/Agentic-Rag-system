"""Prompt templates for graders and the generator.

Grader prompts are intentionally terse and single-purpose so the structured
``YesNo`` output is unambiguous. Wording is a senior-engineer default; tune as
needed. Keeping them here (not inline) makes them easy to sweep in evaluation.
"""

from __future__ import annotations

# --- Document relevance grader ------------------------------------------------
DOC_GRADER_SYSTEM = (
    "You are a grader assessing whether a retrieved document is relevant to a "
    "user question. If the document contains keywords or semantic meaning "
    "related to the question, grade it 'yes'. Be lenient: the goal is to filter "
    "out clearly unrelated documents, not to require a perfect answer. Give a "
    "binary 'yes' or 'no'."
)
DOC_GRADER_USER = (
    "Retrieved document:\n\n{document}\n\nUser question:\n\n{question}"
)

# --- Groundedness (hallucination) grader --------------------------------------
GROUNDED_GRADER_SYSTEM = (
    "You are a grader assessing whether an answer is grounded in / supported by "
    "a set of retrieved facts. Grade 'yes' only if every substantive claim in "
    "the answer is supported by the provided facts. If the answer introduces "
    "information not present in the facts, grade 'no'. An honest 'I don't know' "
    "counts as grounded ('yes')."
)
GROUNDED_GRADER_USER = (
    "Set of facts:\n\n{documents}\n\nLLM answer:\n\n{generation}"
)

# --- Answer-relevance grader --------------------------------------------------
ANSWER_GRADER_SYSTEM = (
    "You are a grader assessing whether an answer actually resolves the user's "
    "question. Grade 'yes' if the answer addresses and resolves the question. "
    "Grade 'no' if it is off-topic, evasive, or explicitly says it cannot "
    "answer / does not know. Give a binary 'yes' or 'no'."
)
ANSWER_GRADER_USER = (
    "User question:\n\n{question}\n\nLLM answer:\n\n{generation}"
)

# --- Generator ---------------------------------------------------------------
GENERATE_SYSTEM = (
    "You are a question-answering assistant. Answer the user's question using "
    "ONLY the provided context. Every substantive claim must be supported by "
    "the context. Cite the sources you use with inline markers like [1], [2] "
    "that correspond to the numbered context blocks. If the context does not "
    "contain enough information to answer, say exactly: \"I don't know based on "
    "the available context.\" Do not use outside knowledge. Be concise."
)
GENERATE_USER = (
    "Numbered context blocks:\n\n{context}\n\nQuestion: {question}\n\n"
    "Write a grounded answer with inline [n] citations."
)

# --- Image captioning --------------------------------------------------------
CAPTION_SYSTEM = (
    "You are a precise visual describer. Describe the image so it can be found "
    "later by a text search. Include any visible text verbatim, describe charts/"
    "diagrams/tables and their data, and note key visual elements. Be factual "
    "and thorough in 1-4 sentences. Do not speculate beyond what is visible."
)
CAPTION_USER = "Describe this image for retrieval."
