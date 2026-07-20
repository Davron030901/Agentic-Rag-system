// Renders citations, visually distinguishing document chunks from web results.

import type { Source } from "../lib/api";

export default function Sources({ sources }: { sources: Source[] }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <h4>Sources</h4>
      {sources.map((s, i) => {
        const inner = (
          <>
            <span className={`tag ${s.type}`}>{s.type === "web" ? "WEB" : "DOC"}</span>
            <span>
              [{i + 1}] {s.title ?? s.id}
            </span>
            {s.snippet ? <span className="snippet">{s.snippet}</span> : null}
          </>
        );
        return s.type === "web" && s.url ? (
          <a key={s.id} className="source" href={s.url} target="_blank" rel="noreferrer">
            {inner}
          </a>
        ) : (
          <div key={`${s.id}-${i}`} className="source">
            {inner}
          </div>
        );
      })}
    </div>
  );
}
