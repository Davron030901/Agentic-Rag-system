"use client";

import { useRef, useState } from "react";
import {
  ACCEPTED_EXTENSIONS,
  confirmationMessage,
  extensionOf,
  ingestFile,
  isSupportedFile,
  type IngestResult,
} from "../lib/api";

interface Status {
  kind: "success" | "error" | "info";
  text: string;
}

export default function Uploader({
  onIngested,
}: {
  onIngested: (r: IngestResult) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);
  const [statuses, setStatuses] = useState<Status[]>([]);

  function pushStatus(s: Status) {
    setStatuses((prev) => [...prev, s]);
  }

  async function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);

    // Client-side validation BEFORE upload — reject unsupported clearly.
    const supported = files.filter((f) => isSupportedFile(f.name));
    const rejected = files.filter((f) => !isSupportedFile(f.name));
    for (const f of rejected) {
      pushStatus({
        kind: "error",
        text: `${f.name} — unsupported format "${extensionOf(f.name) || "unknown"}". Allowed: ${ACCEPTED_EXTENSIONS.join(", ")}.`,
      });
    }
    if (supported.length === 0) return;

    setBusy(true);
    for (const f of supported) {
      setCurrent(f.name);
      try {
        const result = await ingestFile(f);
        pushStatus({ kind: "success", text: confirmationMessage(result) });
        onIngested(result);
      } catch (e: any) {
        pushStatus({ kind: "error", text: `${f.name} — ${e?.message ?? "upload failed"}` });
      }
    }
    setCurrent(null);
    setBusy(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="uploader">
      <div
        className={`dropzone ${dragging ? "drag" : ""} ${busy ? "busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!busy) handleFiles(e.dataTransfer.files);
        }}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        aria-label="Upload document"
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS.join(",")}
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />
        {busy ? (
          <div className="dz-inner">
            <span className="spinner" aria-hidden />
            <span>Ingesting {current}…</span>
          </div>
        ) : (
          <div className="dz-inner">
            <button
              type="button"
              className="upload-btn"
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
            >
              Upload document
            </button>
            <span className="dz-hint">
              or drag &amp; drop — PDF, TXT, MD, DOCX, PNG/JPG/WEBP (multiple allowed)
            </span>
          </div>
        )}
      </div>

      {statuses.length > 0 && (
        <div className="statuses">
          {statuses.map((s, i) => (
            <div key={i} className={`status ${s.kind}`}>
              {s.kind === "success" ? "✓ " : s.kind === "error" ? "⚠️ " : ""}
              {s.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
