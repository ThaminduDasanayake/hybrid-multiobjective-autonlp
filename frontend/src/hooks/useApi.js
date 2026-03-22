/**
 * React Query hooks for the T-AutoNLP API.
 *
 * Query key hierarchy
 *   ["jobs"]                  — full job list (shared between History and Experiments pages)
 *   ["job-result", jobId]     — result.json for one completed job
 *   ["ablations"]             — all ablation study results
 *
 * Mutation hooks invalidate the relevant query keys on success so that
 * dependent components re-render automatically without manual state updates.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  cancelJob,
  deleteJob,
  getAblations,
  getHypervolumeHistory,
  getJobResult,
  getJobs,
  runAblation,
  startJob,
} from "@/api.js";

// query hooks

/**
 * Fetches all jobs and filters to completed ones only.
 * Both HistoryAnalysis and Experiments share this cache entry.
 *
 * @returns {import("@tanstack/react-query").UseQueryResult<Record<string, object>>}
 */
export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: async () => {
      const data = await getJobs();
      // Keep only completed jobs — incomplete runs have no result.json.
      return Object.fromEntries(Object.entries(data).filter(([, s]) => s.status === "completed"));
    },
    // 15 s is long enough that navigating between pages doesn't re-fetch,
    // but short enough that a job completing in the background is seen soon.
    staleTime: 15_000,
  });
}

/**
 * Fetches the result.json for a single completed job.
 * The query is disabled until a jobId is provided.
 *
 * @param {string | null | undefined} jobId
 * @returns {import("@tanstack/react-query").UseQueryResult<object>}
 */
export function useJobResult(jobId) {
  return useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => getJobResult(jobId),
    // Only run when a job is actually selected.
    enabled: !!jobId,
    // Completed job results are immutable — never re-fetch unless explicitly
    // invalidated (e.g. after a cache-clear).
    staleTime: Infinity,
  });
}

/**
 * Fetches all ablation study results.
 *
 * Pass `refetchInterval` to enable automatic polling while ablation jobs are
 * still running (Experiments uses this while queued jobs are pending).
 *
 * @param {{ refetchInterval?: number | false }} [opts]
 * @returns {import("@tanstack/react-query").UseQueryResult<Record<string, object>>}
 */
export function useAblations({ refetchInterval = false } = {}) {
  return useQuery({
    queryKey: ["ablations"],
    queryFn: getAblations,
    staleTime: 30_000,
    refetchInterval,
  });
}

/**
 * Fetches per-generation hypervolume history for a completed job.
 *
 * @param {string | null | undefined} jobId
 * @returns {import("@tanstack/react-query").UseQueryResult<Array<{ generation: number, hypervolume: number }>>}
 */
export function useHypervolumeHistory(jobId) {
  return useQuery({
    queryKey: ["hv-history", jobId],
    queryFn: () => getHypervolumeHistory(jobId),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

// mutation hooks

/**
 * Starts a new AutoML job.
 * Invalidates the job list on success so the Run page can redirect and
 * History can pick up the new entry.
 *
 * @returns {import("@tanstack/react-query").UseMutationResult}
 */
export function useStartJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: startJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

/**
 * Terminates a running job.
 * Invalidates the job list so status badges update immediately.
 *
 * @returns {import("@tanstack/react-query").UseMutationResult}
 */
export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId) => cancelJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

/**
 * Permanently deletes a completed job and its data.
 * Invalidates the job list so the table updates immediately.
 *
 * @returns {import("@tanstack/react-query").UseMutationResult}
 */
export function useDeleteJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId) => deleteJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

/**
 * Queues an ablation study run.
 * Invalidates the ablations cache on success so the table refreshes.
 *
 * @returns {import("@tanstack/react-query").UseMutationResult}
 */
export function useRunAblation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: runAblation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ablations"] });
    },
  });
}
