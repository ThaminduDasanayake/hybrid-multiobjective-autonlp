/**
 * Global client-side state via Zustand.
 *
 * Keeps cross-page state (active job, last result) out of individual
 * component state so that navigating between pages doesn't lose it.
 */
import { create } from "zustand";

export const useStore = create((set) => ({
  /** ID of the job currently being monitored, or null. */
  activeJobId: null,

  /** Cached result payload for the most recently completed job. */
  lastResult: null,

  setActiveJobId: (id) => set({ activeJobId: id }),
  setLastResult: (result) => set({ lastResult: result }),

  /** Clear both when the user starts a brand-new run. */
  resetJob: () => set({ activeJobId: null, lastResult: null }),
}));
