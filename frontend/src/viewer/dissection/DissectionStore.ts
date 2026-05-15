import { create } from "zustand";

export type ClipAxis = "x" | "y" | "z";

interface DissectionState {
  focusedObjectId: string | null;
  clipAxis: ClipAxis;
  /** [0, 1] — 0 = no cut (plane at far side of bbox), 1 = fully clipped (plane at near side). */
  clipT: number;
  /** [0, 1] — exploded radius factor. */
  explodeT: number;
  isExplodeAvailable: boolean;
  setFocused: (id: string | null, defaults?: { axis?: ClipAxis; t?: number }) => void;
  setClip: (axis: ClipAxis, t: number) => void;
  setExplode: (t: number) => void;
  setExplodeAvailable: (b: boolean) => void;
  reset: () => void;
}

export const useDissectionStore = create<DissectionState>((set) => ({
  focusedObjectId: null,
  clipAxis: "y",
  clipT: 0,
  explodeT: 0,
  isExplodeAvailable: false,
  setFocused: (id, defaults) =>
    set({
      focusedObjectId: id,
      clipAxis: defaults?.axis ?? "y",
      clipT: defaults?.t ?? 0,
      explodeT: 0,
    }),
  setClip: (axis, t) => set({ clipAxis: axis, clipT: t }),
  setExplode: (t) => set({ explodeT: t }),
  setExplodeAvailable: (b) => set({ isExplodeAvailable: b }),
  reset: () => set({ clipT: 0, explodeT: 0 }),
}));
