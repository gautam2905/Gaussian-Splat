import { useEffect } from "react";
import { useParams } from "react-router-dom";

import ExhibitViewer from "../viewer/ExhibitViewer";
import HintChip from "../viewer/hud/HintChip";
import ProgressBar from "../viewer/hud/ProgressBar";
import StationIndicator from "../viewer/hud/StationIndicator";
import DissectionPanel from "../viewer/dissection/DissectionPanel";
import { useExhibitStore } from "../viewer/store/exhibitStore";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

export default function Exhibit() {
  const { id } = useParams<{ id: string }>();
  const manifest = useExhibitStore((s) => s.manifest);
  const loading = useExhibitStore((s) => s.loading);
  const error = useExhibitStore((s) => s.error);
  const load = useExhibitStore((s) => s.load);

  useEffect(() => {
    if (id) void load(id);
  }, [id, load]);

  if (loading) return <main className="p-8">Loading exhibit…</main>;
  if (error) return <main className="p-8 text-red-400">Error loading exhibit: {error}</main>;
  if (!manifest) return <main className="p-8">No exhibit data.</main>;

  return (
    <>
      <ExhibitViewer manifest={manifest} apiBase={API_BASE} />
      <StationIndicator />
      <ProgressBar />
      <HintChip />
      <DissectionPanel
        onClose={() => window.dispatchEvent(new CustomEvent("viewer:exit-focus"))}
      />
    </>
  );
}
