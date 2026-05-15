import { useLocomotionStore } from "../store/locomotionStore";

export default function ProgressBar() {
  const t = useLocomotionStore((s) => s.splineT);
  const mode = useLocomotionStore((s) => s.mode);
  if (mode !== "walking") return null;
  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-64 h-1 bg-white/10 rounded-full overflow-hidden">
      <div
        className="h-full bg-amber-400 transition-all duration-150"
        style={{ width: `${Math.max(0, Math.min(1, t)) * 100}%` }}
      />
    </div>
  );
}
