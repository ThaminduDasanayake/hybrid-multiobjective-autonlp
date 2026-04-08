// All API calls to the backend. In dev, Vite proxies /api/* to localhost:8000. In prod, set VITE_API_BASE_URL.

import { BASE_URL } from "@/constants.js";

// Builds the full SSE stream URL for a job.
export function streamUrl(jobId) {
  return `${BASE_URL}/api/jobs/${jobId}/stream`;
}

// Fetch wrapper — throws a readable error message on non-2xx responses.
async function _request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!res.ok) {
    // Pull the error message out of FastAPI's { detail: "..." } response.
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      const detail = body?.detail;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        // Pydantic validation errors come as an array — join them into one message.
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

// Start a new AutoML optimization job.
export function startJob(config) {
  return _request("/api/jobs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

// Fetch all jobs, newest first.
export function getJobs() {
  return _request("/api/jobs");
}

// Fetch the result for a completed job.
export function getJobResult(jobId) {
  return _request(`/api/jobs/${jobId}/result`);
}

// Terminate a running or queued job.
export function cancelJob(jobId) {
  return _request(`/api/jobs/${jobId}`, { method: "DELETE" });
}

// Permanently delete a job and all its stored data.
export function deleteJob(jobId) {
  return _request(`/api/jobs/${jobId}/data`, { method: "DELETE" });
}

// Fetch ablation study results, optionally filtered to a single parent job.
export function getAblations(parentJobId) {
  const params = parentJobId ? `?parent_job_id=${encodeURIComponent(parentJobId)}` : "";
  return _request(`/api/ablations${params}`);
}

// Queue an ablation study run (fire-and-forget).
export function runAblation(config) {
  return _request("/api/ablations", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

// Fetch per-generation hypervolume data for the convergence chart.
export function getHypervolumeHistory(jobId) {
  return _request(`/api/jobs/${jobId}/hypervolume-history`);
}

// Submit user feedback.
export function submitFeedback(feedback) {
  return _request("/api/feedback", {
    method: "POST",
    body: JSON.stringify(feedback),
  });
}
