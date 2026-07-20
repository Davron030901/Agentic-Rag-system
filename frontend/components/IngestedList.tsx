// Shows the documents the assistant currently has access to (this session).

import type { FileType, IngestResult } from "../lib/api";

const ICON: Record<FileType, string> = {
  pdf: "📄",
  docx: "📝",
  text: "🗒️",
  image: "🖼️",
};

const TYPE_LABEL: Record<FileType, string> = {
  pdf: "PDF",
  docx: "DOCX",
  text: "Text",
  image: "Image",
};

export default function IngestedList({ docs }: { docs: IngestResult[] }) {
  if (docs.length === 0) return null;
  return (
    <div className="ingested">
      <h4>Indexed documents ({docs.length})</h4>
      <ul>
        {docs.map((d, i) => (
          <li key={`${d.filename}-${i}`} className="ingested-item">
            <span className="ic" aria-hidden>
              {ICON[d.file_type]}
            </span>
            <span className="fn" title={d.filename}>
              {d.filename}
            </span>
            <span className="ft">{TYPE_LABEL[d.file_type]}</span>
            <span className="ck">
              {d.chunks} chunk{d.chunks === 1 ? "" : "s"}
              {d.images_captioned > 0 ? ` · ${d.images_captioned} img` : ""}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
