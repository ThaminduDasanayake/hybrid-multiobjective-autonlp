// Global Zustand store — keeps job state alive across page navigation.
import { create } from "zustand";
import { ACTIVE_JOB_KEY } from "./constants";

export const useStore = create((set) => ({
  // The job currently being watched, or null.
  activeJobId: localStorage.getItem(ACTIVE_JOB_KEY) || null,

  setActiveJobId: (id) => {
    if (id) localStorage.setItem(ACTIVE_JOB_KEY, id);
    else localStorage.removeItem(ACTIVE_JOB_KEY);
    set({ activeJobId: id });
  },

  // Clear the active job when starting a fresh run.
  resetJob: () => {
    localStorage.removeItem(ACTIVE_JOB_KEY);
    set({ activeJobId: null });
  },

  // Tracks which ablations have been submitted but haven't finished yet (survives page navigation).
  queuedAblations: {},

  // Mark an ablation as pending or clear it once it completes.
  setAblationQueued: (key, status) =>
    set((state) => ({
      queuedAblations: { ...state.queuedAblations, [key]: status },
    })),
}));
