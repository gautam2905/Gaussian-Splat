import { useState } from "react";
import { useNavigate } from "react-router-dom";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

const STAGE_LABELS: Record<string, string> = {
  orchestrating: "Planning your exhibit…",
  scene_gen: "Building the room…",
  objects_gen: "Generating objects…",
  object_ready: "Crafted an object…",
  ready: "Done!",
  refused: "This prompt couldn't be exhibited",
  failed: "Something went wrong",
};

export default function Generate() {
  const [prompt, setPrompt] = useState("");
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const submit = () => {
    if (!prompt.trim()) return;
    setStage("connecting");
    setError(null);
    const url = `${API_BASE}/api/exhibits/events?prompt=${encodeURIComponent(prompt)}`;
    const es = new EventSource(url);

    const knownEvents = [
      "stage",
      "object_ready",
      "object_failed",
      "ready",
      "refused",
      "failed",
    ] as const;

    const handlers = new Map<string, (e: MessageEvent) => void>();

    knownEvents.forEach((name) => {
      const h = (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data as string) as Record<string, unknown>;
          if (name === "stage") {
            setStage(data.stage as string);
          } else if (name === "object_ready") {
            setStage("object_ready");
          } else if (name === "ready") {
            setStage("ready");
            es.close();
            void navigate(`/exhibit/${data.exhibit_id as string}`);
          } else if (name === "refused") {
            setError(`Refused: ${data.reason as string}`);
            es.close();
          } else if (name === "failed") {
            setError(`Failed: ${data.detail as string}`);
            es.close();
          }
        } catch {
          /* ignore malformed event */
        }
      };
      handlers.set(name, h);
      es.addEventListener(name, h as EventListener);
    });

    es.onerror = () => {
      setError("Connection error");
      es.close();
    };
  };

  const busy = stage !== null && !["ready", "refused", "failed"].includes(stage);

  return (
    <main className="p-8 max-w-2xl mx-auto">
      <h1 className="text-3xl font-serif">Generate an exhibit</h1>
      <textarea
        className="mt-4 w-full bg-black/40 border border-white/20 rounded p-3"
        rows={3}
        placeholder="An anatomy exhibit / A horology workshop / A botany garden"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={busy}
      />
      <button
        className="mt-4 bg-amber-500 text-black px-6 py-2 rounded font-semibold disabled:opacity-40"
        onClick={submit}
        disabled={busy || !prompt.trim()}
      >
        Generate
      </button>
      {stage && <p className="mt-6 text-lg">{STAGE_LABELS[stage] ?? stage}</p>}
      {error && <p className="mt-6 text-red-400">{error}</p>}
    </main>
  );
}
