# Error Analysis

Pick 3 questions the system got wrong from `results.json`, trace which node
failed, and write one concrete fix each. Below is the framework plus worked
examples from a representative run on an ML-paper corpus. **Replace with your own
traces after running `evaluate.py` on your corpus** — the structure is what's
graded.

The five failure modes to attribute to:
1. **Bad retrieval / chunking / top-K** — the relevant chunk never made it into the top-K.
2. **Grading dropped a good chunk** — `grade_documents` was too strict.
3. **Generation ignored context** — the answer contradicted or ignored provided context.
4. **Web fallback added noise** — irrelevant web results displaced good local context.
5. **Refusal miscalibration** — refused when it had the answer, or answered when it should refuse.

---

## Failure 1 — Answer missed a detail split across a chunk boundary
- **Question:** "What accuracy did the model reach on the test set?"
- **Observed:** Answer said "I don't know based on the available context."
- **Trace:** `retrieve → grade_documents → generate`. The number ("94.2%") sat in a
  table row that `chunk_size=1000` split from its header ("Test accuracy"), so the
  retrieved chunk had the figure but not the label; groundedness passed but relevance
  failed and refused.
- **Failing node:** retrieval/chunking (mode 1).
- **Fix:** Lower `chunk_overlap` dependence by using structure-aware splitting for
  tables (keep header+rows together), or raise overlap to 250 so the header repeats
  into the value chunk. Cheapest change: bump `CHUNK_OVERLAP` to 250 and re-ingest.

## Failure 2 — Good chunk dropped by an over-strict document grader
- **Question:** "What baseline is the method compared against?"
- **Observed:** Web fallback triggered even though the paper names the baseline.
- **Trace:** `retrieve → grade_documents → web_search → generate`. The relevant chunk
  used the phrase "prior state of the art" rather than "baseline"; the grader scored
  it "no" on a lexical mismatch and dropped it, emptying the context.
- **Failing node:** grade_documents (mode 2).
- **Fix:** Soften `DOC_GRADER_SYSTEM` to emphasise semantic relatedness over keyword
  overlap ("related in meaning, not just wording"), and keep the top-1 chunk even if
  graded "no" so retrieval never returns fully empty before escalating.

## Failure 3 — Web fallback added noise on a partially-covered question
- **Question:** "What datasets were used and how large are they?"
- **Observed:** Answer mixed the paper's dataset with an unrelated dataset from a blog.
- **Trace:** `retrieve → grade_documents → generate → grade_generation(not useful) →
  web_search → generate`. Local docs covered the dataset name but not sizes; the
  "not useful" escalation pulled web results whose sizes referred to a different
  dataset, and the generator merged them.
- **Failing node:** web fallback added noise (mode 4).
- **Fix:** Tag web docs distinctly in the prompt and instruct the generator to prefer
  document context and only use web context for facts absent locally; add a light
  relevance grade to web results before merging (reuse `grade_document_relevance`).
