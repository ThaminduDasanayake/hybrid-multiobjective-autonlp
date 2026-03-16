/**
 * Global client-side state via Zustand.
 *
 * Keeps cross-page state (active job) out of individual component state
 * so that navigating between pages doesn't lose it.
 */
import { create } from "zustand";

export const useStore = create((set) => ({
  /** ID of the job currently being monitored, or null. */
  activeJobId: null,

  setActiveJobId: (id) => set({ activeJobId: id }),

  /** Clear when the user starts a brand-new run. */
  resetJob: () => set({ activeJobId: null }),
}));
