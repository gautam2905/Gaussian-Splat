import { create } from "zustand";

import { getExhibit } from "../../api/client";
import type { Manifest } from "../../api/types";

interface ExhibitState {
  manifest: Manifest | null;
  loading: boolean;
  error: string | null;
  load: (id: string) => Promise<void>;
}

export const useExhibitStore = create<ExhibitState>((set) => ({
  manifest: null,
  loading: false,
  error: null,
  load: async (id) => {
    set({ loading: true, error: null, manifest: null });
    try {
      const m = await getExhibit(id);
      set({ manifest: m, loading: false });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ error: msg, loading: false });
    }
  },
}));
