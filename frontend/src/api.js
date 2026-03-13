/**
 * API client for the T-AutoNLP FastAPI backend.
 *
 * During development, Vite proxies all /api/* requests to http://localhost:8000,
 * so no base URL or CORS headers are needed here.
 *
 * NOTE: The SSE /stream endpoint is intentionally NOT wrapped here.
 * Use the browser's native EventSource API directly inside your components:
 *
 *   const es = new EventSource(`/api/jobs/${jobId}/stream`);
 *   es.onmessage = (e) => { const { status, logs } = JSON.parse(e.data); };
 *   es.onerror   = ()  => es.close();
 */

/**
 * Internal helper — sends a fetch request and throws a descriptive Error if
 * the response is not 2xx.  Returns parsed JSON on success.
 *
 * @param {string} path
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
async function _request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    // FastAPI returns { detail: "..." } for errors — surface that message.
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body?.detail ?? JSON.stringify(body);
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
 * Resume a terminated or failed job from its last checkpoint.
 *
 * @param {string} jobId
 * @returns {Promise<{ message: string }>}
 */
export function resumeJob(jobId) {
  return _request(`/api/jobs/${jobId}/resume`, { method: "POST" });
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
 * @param {{ mode: string, disable_bo: boolean, dataset: string }} config
 * @returns {Promise<{ status: string, mode: string, dataset: string, disable_bo: boolean }>}
 */
export function runAblation(config) {
  return _request("/api/ablations", {
    method: "POST",
    body: JSON.stringify(config),
  });
}
