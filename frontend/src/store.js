/**
 * Global client-side state via Zustand.
 *
 * Keeps cross-page state (active job) out of individual component state
 * so that navigating between pages doesn't lose it.
 */
import { create } from "zustand";

const ACTIVE_JOB_KEY = "t_autonlp_active_job";

export const useStore = create((set) => ({
  /** ID of the job currently being monitored, or null. */
  activeJobId: localStorage.getItem(ACTIVE_JOB_KEY) || null,

  setActiveJobId: (id) => {
    if (id) localStorage.setItem(ACTIVE_JOB_KEY, id);
    else localStorage.removeItem(ACTIVE_JOB_KEY);
    set({ activeJobId: id });
  },

  /** Clear when the user starts a brand-new run. */
  resetJob: () => {
    localStorage.removeItem(ACTIVE_JOB_KEY);
    set({ activeJobId: null });
  },

  /**
   * Tracks ablation keys (e.g. "single_f1_job_20250318_a1b2c3d4") that have
   * been submitted but haven't completed yet.  Keys are scoped to the parent
   * job ID.  Lives in the global store so that the state survives page
   * navigation (component remounts).
   */
  queuedAblations: {},

  /** Mark an ablation key as queued (true) or clear it (false). */
  setAblationQueued: (key, status) =>
    set((state) => ({
      queuedAblations: { ...state.queuedAblations, [key]: status },
    })),
}));
