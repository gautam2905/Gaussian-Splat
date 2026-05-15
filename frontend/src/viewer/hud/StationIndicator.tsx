import { useExhibitStore } from "../store/exhibitStore";
import { useLocomotionStore } from "../store/locomotionStore";

export default function StationIndicator() {
  const manifest = useExhibitStore((s) => s.manifest);
  const currentStation = useLocomotionStore((s) => s.currentStation);
  const mode = useLocomotionStore((s) => s.mode);
  if (!manifest || mode !== "walking") return null;
  const station = manifest.spline.stations[currentStation];
  if (!station) return null;
  return (
    <div
      className="absolute top-4 left-1/2 -translate-x-1/2 bg-black/40 backdrop-blur-sm px-4 py-2 rounded text-sm font-serif italic"
      style={{ borderBottom: "1px solid #8a6b3a" }}
    >
      Station {currentStation + 1} of {manifest.spline.stations.length} — {station.label}
    </div>
  );
}
