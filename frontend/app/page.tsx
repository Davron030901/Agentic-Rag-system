"use client";

import { useState } from "react";
import { askQuestion, type ChatResponse, type IngestResult } from "../lib/api";
import StepPills from "../components/StepPills";
import Sources from "../components/Sources";
import Uploader from "../components/Uploader";
import IngestedList from "../components/IngestedList";

interface Turn {
  question: string;
  response?: ChatResponse;
  error?: string;
  loading?: boolean;
}

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [docs, setDocs] = useState<IngestResult[]>([]);

  const hasContext = docs.length > 0;

  function onIngested(r: IngestResult) {
    // Accumulate across the session; keep previously ingested docs.
    setDocs((prev) => [...prev, r]);
  }

  async function submit() {
    const q = input.trim();
    if (!q || busy || !hasContext) return;
    setInput("");
    setBusy(true);
    const idx = turns.length;
    setTurns((t) => [...t, { question: q, loading: true }]);
    try {
      const response = await askQuestion(q);
      setTurns((t) => t.map((x, i) => (i === idx ? { question: q, response } : x)));
    } catch (e: any) {
      setTurns((t) =>
        t.map((x, i) => (i === idx ? { question: q, error: e.message ?? "Request failed" } : x)),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap">
      <div className="header">
        <h1>Adaptive Agentic RAG Assistant</h1>
        <p>
          Self-grading retrieval with web-search fallback. Upload documents, then
          ask questions — each answer shows the agent&apos;s reasoning path and cited
          sources.
        </p>
      </div>

      <Uploader onIngested={onIngested} />
      <IngestedList docs={docs} />

      {turns.length === 0 && (
        <div className="empty">
          {hasContext
            ? "Ask a question about your indexed documents."
            : "Upload a document to get started."}
        </div>
      )}

      {turns.map((t, i) => (
        <div key={i}>
          <div className="msg user">
            <div className="role">You</div>
            <div className="answer">{t.question}</div>
          </div>
          <div className="msg">
            <div className="role">Assistant</div>
            {t.loading && <div className="answer">Thinking…</div>}
            {t.error && <div className="answer">⚠️ {t.error}</div>}
            {t.response && (
              <>
                <StepPills steps={t.response.steps} />
                {t.response.web_search_used && <span className="pill web">web fallback used</span>}
                {t.response.low_confidence && (
                  <span className="flag">low confidence (retry cap reached)</span>
                )}
                <div className="answer" style={{ marginTop: 10 }}>
                  {t.response.answer}
                </div>
                <Sources sources={t.response.sources} />
              </>
            )}
          </div>
        </div>
      ))}

      <div className="composer">
        <div className="inner">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder={hasContext ? "Ask a question…" : "Upload a document to get started"}
            disabled={busy || !hasContext}
          />
          <button onClick={submit} disabled={busy || !hasContext || !input.trim()}>
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
