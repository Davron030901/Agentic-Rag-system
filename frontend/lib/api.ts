// Typed client for the backend /chat endpoint.

export interface Source {
  type: "doc" | "web";
  id: string;
  url?: string | null;
  title?: string | null;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  steps: string[];
  sources: Source[];
  web_search_used: boolean;
  low_confidence: boolean;
}

export type FileType = "pdf" | "docx" | "text" | "image";

export interface IngestResult {
  filename: string;
  file_type: FileType;
  chunks: number;
  images_captioned: number;
  pages?: number | null;
  status: "indexed" | "error";
  detail?: string;
  collection: string;
  provider: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:7860";

// Formats the frontend accepts; kept in sync with the backend SUPPORTED_EXTS.
export const ACCEPTED_EXTENSIONS = [
  ".pdf",
  ".txt",
  ".md",
  ".markdown",
  ".docx",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
];

export function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export function isSupportedFile(filename: string): boolean {
  return ACCEPTED_EXTENSIONS.includes(extensionOf(filename));
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through to text */
  }
  try {
    return await res.text();
  } catch {
    return `HTTP ${res.status}`;
  }
}

export async function askQuestion(question: string): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    throw new Error(await readError(res));
  }
  return (await res.json()) as ChatResponse;
}

export async function ingestFile(file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(await readError(res));
  }
  return (await res.json()) as IngestResult;
}

// Human-readable confirmation string, tailored per file type.
export function confirmationMessage(r: IngestResult): string {
  if (r.file_type === "image") {
    return `${r.filename} — image captioned and indexed successfully.`;
  }
  if (r.file_type === "text") {
    return `${r.filename} — ${r.chunks} chunks — indexed successfully.`;
  }
  // pdf / docx
  const pagePart = r.pages != null ? `${r.pages} pages, ` : "";
  return `${r.filename} — ${pagePart}${r.chunks} chunks, ${r.images_captioned} images captioned — indexed successfully.`;
}
