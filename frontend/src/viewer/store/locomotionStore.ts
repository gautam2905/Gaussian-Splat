import { create } from "zustand";

export type CameraMode = "walking" | "focused";

interface LocomotionState {
  mode: CameraMode;
  currentStation: number;
  inTransition: boolean;
  splineT: number;               // 0..1 along spline; updated each frame in walking
  focusedObjectId: string | null; // set when in focused mode
  setMode: (m: CameraMode) => void;
  setStation: (i: number) => void;
  setSplineT: (t: number) => void;
  setTransitioning: (b: boolean) => void;
  setFocusedObjectId: (id: string | null) => void;
}

export const useLocomotionStore = create<LocomotionState>((set) => ({
  mode: "walking",
  currentStation: 0,
  inTransition: false,
  splineT: 0,
  focusedObjectId: null,
  setMode: (m) => set({ mode: m }),
  setStation: (i) => set({ currentStation: i }),
  setSplineT: (t) => set({ splineT: t }),
  setTransitioning: (b) => set({ inTransition: b }),
  setFocusedObjectId: (id) => set({ focusedObjectId: id }),
}));
