// Renders the ordered list of graph nodes the agent actually visited.

const LABELS: Record<string, string> = {
  retrieve: "retrieve",
  grade_documents: "grade docs",
  web_search: "web search",
  generate: "generate",
  grade_generation: "grade generation",
};

export default function StepPills({ steps }: { steps: string[] }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="pills">
      {steps.map((s, i) => (
        <span key={`${s}-${i}`} className={`pill ${s === "web_search" ? "web" : s}`}>
          {i + 1}. {LABELS[s] ?? s}
        </span>
      ))}
    </div>
  );
}
