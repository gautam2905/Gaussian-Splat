import { useEffect, useState } from "react";

import { useLocomotionStore } from "../store/locomotionStore";

export default function HintChip() {
  const [visible, setVisible] = useState(true);
  const mode = useLocomotionStore((s) => s.mode);
  useEffect(() => {
    const onAnyInput = () => setVisible(false);
    window.addEventListener("keydown", onAnyInput, { once: true });
    return () => window.removeEventListener("keydown", onAnyInput);
  }, []);
  if (!visible || mode !== "walking") return null;
  return (
    <div className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-sm px-3 py-2 rounded text-xs text-white/80 font-serif">
      → next &nbsp;|&nbsp; ← back &nbsp;|&nbsp; click to inspect &nbsp;|&nbsp; Esc to exit
    </div>
  );
}
