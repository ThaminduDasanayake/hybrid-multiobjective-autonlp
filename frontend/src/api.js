/**
 * API client for the T-AutoNLP FastAPI backend.
 *
 * Base URL strategy
 * -----------------
 * In development Vite proxies /api/* → localhost:8000 (see vite.config.js),
 * so VITE_API_BASE_URL is intentionally left empty in .env and relative paths
 * work without CORS overhead.
 *
 * In production (Vercel) the proxy does not exist.  Set VITE_API_BASE_URL to
 * your backend's public URL (Ngrok tunnel or cloud VM) in the Vercel dashboard
 * or in .env.production, and every fetch/EventSource will use it automatically.
 *
 * SSE streams
 * -----------
 * Use the exported `streamUrl(jobId)` helper when constructing an EventSource
 * so the base URL is applied consistently:
 *
 *   const es = new EventSource(streamUrl(jobId));
 */

import { BASE_URL } from "@/constants.js";

/**
 * Returns the full SSE stream URL for a given job.
 * Use this instead of hardcoding the path in components.
 *
 * @param {string} jobId
 * @returns {string}
 */
export function streamUrl(jobId) {
  return `${BASE_URL}/api/jobs/${jobId}/stream`;
}

/**
 * Internal helper — sends a fetch request and throws a descriptive Error if
 * the response is not 2xx.  Returns parsed JSON on success.
 *
 * @param {string} path
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
async function _request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    // FastAPI returns { detail: "..." } for errors — surface that message.
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      const detail = body?.detail;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        // Pydantic validation errors: [{ msg, loc, type }, ...]
        message = detail.map((e) => e.msg ?? JSON.stringify(e)).join("; ");
      } else {
        message = JSON.stringify(body);
      }
    } catch {
      message = await res.text().catch(() => message);
    }
    throw new Error(message);
  }

  return res.json();
}

/**
 * Start a new AutoML optimisation job.
 *
 * @param {{
 *   dataset_name: string,
 *   max_samples?: number,
 *   population_size?: number,
 *   n_generations?: number,
 *   bo_calls?: number,
 *   optimization_mode?: string,
 *   disable_bo?: boolean,
 * }} config
 * @returns {Promise<{ job_id: string }>}
 */
export function startJob(config) {
  return _request("/api/jobs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Fetch all jobs sorted by start time (newest first).
 *
 * @returns {Promise<Record<string, object>>}
 */
export function getJobs() {
  return _request("/api/jobs");
}

/**
 * Fetch the result.json for a completed job.
 * Throws if the job has not finished or the result file is missing.
 *
 * @param {string} jobId
 * @returns {Promise<object>}
 */
export function getJobResult(jobId) {
  return _request(`/api/jobs/${jobId}/result`);
}

/**
 * Terminate a running or queued job.
 *
 * @param {string} jobId
 * @returns {Promise<{ message: string }>}
 */
export function cancelJob(jobId) {
  return _request(`/api/jobs/${jobId}`, { method: "DELETE" });
}

/**
 * Fetch metrics from all completed ablation studies.
 *
 * @returns {Promise<Record<string, object>>}
 */
export function getAblations() {
  return _request("/api/ablations");
}

/**
 * Queue an ablation study run (fire-and-forget, server returns HTTP 202).
 *
 * @param {{ mode: string, disable_bo: boolean, parent_job_id: string }} config
 * @returns {Promise<{ status: string, mode: string, parent_job_id: string, disable_bo: boolean }>}
 */
export function runAblation(config) {
  return _request("/api/ablations", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Fetch per-generation hypervolume history for convergence plotting.
 *
 * @param {string} jobId
 * @returns {Promise<Array<{ generation: number, hypervolume: number }>>}
 */
export function getHypervolumeHistory(jobId) {
  return _request(`/api/jobs/${jobId}/hypervolume-history`);
}
