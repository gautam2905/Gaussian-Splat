import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "http://localhost:8000";

const STAGE_LABELS: Record<string, string> = {
  connecting: "Connecting…",
  orchestrating: "Planning your exhibit…",
  scene_gen: "Building the room…",
  objects_gen: "Generating objects…",
  object_ready: "Crafted an object…",
  ready: "Done — entering exhibit",
  refused: "This prompt couldn't be exhibited",
  failed: "Something went wrong",
};

const TERMINAL_STAGES = new Set(["refused", "failed"]);

export default function Generate() {
  const [searchParams] = useSearchParams();
  const [prompt, setPrompt] = useState(searchParams.get("prompt") ?? "");
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const firedRef = useRef(false);

  const submit = (overridePrompt?: string) => {
    const p = overridePrompt ?? prompt;
    if (!p.trim()) return;
    if (firedRef.current) return;
    firedRef.current = true;
    setStage("connecting");
    setError(null);

    const url = `${API_BASE}/api/exhibits/events?prompt=${encodeURIComponent(p)}`;
    const es = new EventSource(url);

    const knownEvents = [
      "stage",
      "object_ready",
      "object_failed",
      "ready",
      "refused",
      "failed",
    ] as const;

    for (const name of knownEvents) {
      es.addEventListener(name, (e: MessageEvent) => {
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
            setStage("refused");
            es.close();
            firedRef.current = false;
          } else if (name === "failed") {
            setError(`Failed: ${data.detail as string}`);
            setStage("failed");
            es.close();
            firedRef.current = false;
          }
        } catch {
          /* ignore malformed event */
        }
      });
    }

    es.onerror = () => {
      setError("Connection error");
      es.close();
      firedRef.current = false;
    };
  };

  // Auto-fire if landed via a gallery link with ?auto=true&prompt=...
  useEffect(() => {
    const auto = searchParams.get("auto") === "true";
    const promptParam = searchParams.get("prompt");
    if (auto && promptParam && !firedRef.current) {
      submit(promptParam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount

  const busy = stage !== null && !TERMINAL_STAGES.has(stage);

  return (
    <main className="p-8 max-w-2xl mx-auto min-h-screen flex flex-col">
      <h1 className="text-3xl font-serif">Generate an exhibit</h1>
      <p className="mt-2 text-sm opacity-60 font-serif italic">
        A few sample seeds: <em>anatomy</em>, <em>horology</em>,{" "}
        <em>botany</em>. Or type anything you want to see in a museum.
      </p>
      <textarea
        className="mt-6 w-full bg-black/40 border border-white/20 focus:border-amber-400 rounded p-3 font-serif"
        rows={3}
        placeholder="An anatomy exhibit / A horologist's workshop / A garden of strange seeds"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={busy}
      />
      <div className="mt-4 flex items-center gap-3">
        <button
          className="bg-amber-500 hover:bg-amber-400 text-black px-6 py-2 rounded font-semibold disabled:opacity-40"
          onClick={() => submit()}
          disabled={busy || !prompt.trim()}
        >
          Generate
        </button>
        <span className="text-xs opacity-60">
          ~10s in stub mode (no API cost)
        </span>
      </div>
      {stage && (
        <div className="mt-8 font-serif">
          <div className="flex items-center gap-3">
            {busy && (
              <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            )}
            <span className="text-lg">{STAGE_LABELS[stage] ?? stage}</span>
          </div>
        </div>
      )}
      {error && <p className="mt-6 text-red-400">{error}</p>}
    </main>
  );
}
