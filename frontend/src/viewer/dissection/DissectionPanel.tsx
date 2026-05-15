import { useEffect } from "react";

import { useExhibitStore } from "../store/exhibitStore";
import { useLocomotionStore } from "../store/locomotionStore";
import { type ClipAxis, useDissectionStore } from "./DissectionStore";

interface Props {
  onClose: () => void;
}

export default function DissectionPanel({ onClose }: Props) {
  const manifest = useExhibitStore((s) => s.manifest);
  const mode = useLocomotionStore((s) => s.mode);
  const focusedObjectId = useLocomotionStore((s) => s.focusedObjectId);

  const clipAxis = useDissectionStore((s) => s.clipAxis);
  const clipT = useDissectionStore((s) => s.clipT);
  const explodeT = useDissectionStore((s) => s.explodeT);
  const isExplodeAvailable = useDissectionStore((s) => s.isExplodeAvailable);
  const setClip = useDissectionStore((s) => s.setClip);
  const setExplode = useDissectionStore((s) => s.setExplode);
  const reset = useDissectionStore((s) => s.reset);

  // Sync DissectionStore's focusedObjectId and defaults whenever the
  // locomotion store transitions into/out of focus mode.
  useEffect(() => {
    if (focusedObjectId && manifest) {
      const obj = manifest.objects.find((o) => o.id === focusedObjectId);
      useDissectionStore.getState().setFocused(focusedObjectId, {
        axis: (obj?.preferred_clip_axis ?? "y") as ClipAxis,
        t: obj?.preferred_clip_t ?? 0.5,
      });
    } else {
      useDissectionStore.getState().setFocused(null);
    }
  }, [focusedObjectId, manifest]);

  if (mode !== "focused" || !manifest || !focusedObjectId) return null;
  const obj = manifest.objects.find((o) => o.id === focusedObjectId);
  if (!obj) return null;

  return (
    <aside
      className="absolute top-4 right-4 w-80 bg-black/70 backdrop-blur-sm p-5 rounded-lg text-white font-serif"
      style={{ borderRight: "2px solid #8a6b3a" }}
    >
      <h2 className="text-xl font-bold">{obj.label}</h2>
      <p className="mt-2 text-sm opacity-80 leading-relaxed">{obj.placard_description}</p>

      <hr className="my-4 border-white/10" />

      <div className="space-y-4 font-sans text-sm">
        <div>
          <label className="block text-xs uppercase tracking-wide opacity-70 mb-1">
            Cut along
          </label>
          <select
            className="bg-black/60 border border-white/20 px-2 py-1 rounded text-sm"
            value={clipAxis}
            onChange={(e) => setClip(e.target.value as ClipAxis, clipT)}
          >
            <option value="x">X — left/right</option>
            <option value="y">Y — top/bottom</option>
            <option value="z">Z — front/back</option>
          </select>
        </div>

        <div>
          <label className="block text-xs uppercase tracking-wide opacity-70 mb-1">
            Cross-section position
            <span className="ml-2 opacity-60">{Math.round(clipT * 100)}%</span>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={clipT}
            onChange={(e) => setClip(clipAxis, parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-xs uppercase tracking-wide opacity-70 mb-1">
            Exploded view
            {!isExplodeAvailable && (
              <span className="ml-2 text-amber-400 text-xs">(single component)</span>
            )}
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={explodeT}
            disabled={!isExplodeAvailable}
            onChange={(e) => setExplode(parseFloat(e.target.value))}
            className="w-full disabled:opacity-40"
          />
        </div>

        <div className="flex gap-2 pt-2">
          <button
            className="px-3 py-1 bg-white/10 hover:bg-white/20 rounded text-sm"
            onClick={reset}
          >
            Reset
          </button>
          <button
            className="px-3 py-1 bg-amber-500 hover:bg-amber-400 text-black rounded text-sm font-semibold"
            onClick={onClose}
          >
            Back to exhibit
          </button>
        </div>
      </div>
    </aside>
  );
}
