// React Query hooks for all API calls — mutations auto-invalidate the relevant cache keys on success.

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
  submitFeedback,
} from "@/api.js";

// query hooks

// Fetches completed jobs only (shared cache between History and Experiments).
export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: async () => {
      const data = await getJobs();
        // Only include completed jobs — others don't have a result yet.
      return Object.fromEntries(Object.entries(data).filter(([, s]) => s.status === "completed"));
    },
    // 15s keeps navigation snappy but still picks up background completions quickly.
    staleTime: 15_000,
  });
}

// Fetches the result for a single completed job (disabled until a jobId is given).
export function useJobResult(jobId) {
  return useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => getJobResult(jobId),
    enabled: !!jobId,
    staleTime: Infinity, // Results are immutable once complete, so never re-fetch.
  });
}

// Fetches ablation results. Pass refetchInterval to poll while jobs are still running.
export function useAblations({ refetchInterval = false, parentJobId } = {}) {
  return useQuery({
    queryKey: ["ablations", parentJobId ?? "all"],
    queryFn: () => getAblations(parentJobId),
    staleTime: 30_000,
    refetchInterval,
  });
}

// Fetches per-generation hypervolume history for the convergence chart.
export function useHypervolumeHistory(jobId) {
  return useQuery({
    queryKey: ["hv-history", jobId],
    queryFn: () => getHypervolumeHistory(jobId),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

// mutation hooks

// Starts a new AutoML job and refreshes the job list on success.
export function useStartJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: startJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// Terminates a running job and refreshes the job list.
export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId) => cancelJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// Permanently deletes a job and clears its cached result.
export function useDeleteJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId) => deleteJob(jobId),
    onSuccess: (_data, jobId) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.removeQueries({ queryKey: ["job-result", jobId] });
      queryClient.removeQueries({ queryKey: ["hv-history", jobId] });
    },
  });
}

// Queues an ablation study and refreshes the ablations list on success.
export function useRunAblation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: runAblation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ablations"] });
    },
  });
}

// Submits user feedback.
export function useSubmitFeedback() {
  return useMutation({
    mutationFn: submitFeedback,
  });
}
