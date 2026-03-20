import { lazy, Suspense, useEffect } from "react";
import { AlertCircle, BarChart3, FlaskConical, Loader2, RefreshCw } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { useAblations, useJobResult, useJobs, useRunAblation } from "../hooks/useApi";
import { useStore } from "../store";
import DropdownSelector from "../components/DropdownSelector";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { fmt } from "../utils/formatters";
import JobConfigCard from "@/components/JobConfigCard.jsx";
import RunButton from "@/components/experiments/RunButton.jsx";
import ComparisonTable from "@/components/experiments/ComparisonTable.jsx";

const AblationBarChart = lazy(() => import("@/components/experiments/AblationBarChart.jsx"));

function f4(v) {
  return v != null ? Number(v).toFixed(4) : null;
}
function fRuntime(secs) {
  if (secs == null) return null;
  return secs < 60 ? `${Number(secs).toFixed(1)} s` : `${(secs / 60).toFixed(1)} m`;
}
function fSize(v) {
  return v != null ? String(v) : null;
}

const Experiments = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobIdFromUrl = searchParams.get("job");

  // Ablation queue state lives in the global Zustand store so it survives
  // page navigation (component remounts).  Keys stay queued until the
  // ablation result appears in GET /api/ablations, not just until HTTP 202.
  const queuedAblations = useStore((s) => s.queuedAblations);
  const setAblationQueued = useStore((s) => s.setAblationQueued);

  // Derived: true while at least one run is still queued
  const isPolling = Object.values(queuedAblations).some(Boolean);

  // server data
  const { data: jobMap = {}, isLoading: jobsLoading, error: jobsError } = useJobs();

  // Derivation chain: URL → localStorage → newest job
  // Each candidate is validated against the current jobMap to avoid stale IDs.
  const completedEntries = Object.entries(jobMap).filter(
    ([, job]) => job?.status === "completed",
  );
  const completedIds = completedEntries.map(([id]) => id);
  const lastSaved = localStorage.getItem("t_autonlp_last_ablation_job");
  const completedSet = new Set(completedIds);
  const selectedJobId =
    (jobIdFromUrl && completedSet.has(jobIdFromUrl) ? jobIdFromUrl : null) ??
    (lastSaved && completedSet.has(lastSaved) ? lastSaved : null) ??
    completedIds[0] ??
    null;

  useEffect(() => {
    // Don't modify localStorage until jobs have loaded — otherwise the
    // transient selectedJobId=null state clears the saved preference.
    if (jobsLoading) return;

    if (selectedJobId) {
      localStorage.setItem("t_autonlp_last_ablation_job", selectedJobId);
      if (jobIdFromUrl !== selectedJobId) {
        setSearchParams({ job: selectedJobId }, { replace: true });
      }
    } else {
      // No valid job after load — clear stale localStorage entry
      localStorage.removeItem("t_autonlp_last_ablation_job");
    }
  }, [jobsLoading, selectedJobId, jobIdFromUrl, setSearchParams]);

  const { data: jobResult } = useJobResult(selectedJobId);

  // Ablations: refetchInterval activates React Query's built-in polling while
  // any ablation is queued, replacing the manual setInterval approach.
  const {
    data: ablationsData,
    isLoading: ablationsLoading,
    error: ablationsError,
    refetch: refetchAblations,
  } = useAblations({ refetchInterval: isPolling ? 10_000 : false });

  const runAblationMutation = useRunAblation();

  // derived values
  const masterMetrics = jobResult?.metrics ?? null;
  const masterRuntime = jobResult?.runtime_seconds ?? null;
  const masterConfig = jobResult?.config ?? null;

  const dataset =
    jobMap[selectedJobId]?._config?.dataset_name ?? jobMap[selectedJobId]?.dataset_name ?? "";

  const d = ablationsData ?? {};
  const single = d[`single_f1_${selectedJobId}`]?.metrics ?? null;
  const singleRT = d[`single_f1_${selectedJobId}`]?.runtime_seconds ?? null;
  const two = d[`multi_2d_${selectedJobId}`]?.metrics ?? null;
  const twoRT = d[`multi_2d_${selectedJobId}`]?.runtime_seconds ?? null;
  const gaOnly = d[`ga_only_${selectedJobId}`]?.metrics ?? null;
  const gaOnlyRT = d[`ga_only_${selectedJobId}`]?.runtime_seconds ?? null;

  // Auto-clear queued keys once their results appear in the ablations response.
  // This replaces the old onSuccess callback which cleared too early (on HTTP 202).
  useEffect(() => {
    if (!ablationsData) return;
    const current = useStore.getState().queuedAblations;
    for (const key of Object.keys(current)) {
      if (current[key] && ablationsData[key]) {
        setAblationQueued(key, false);
      }
    }
  }, [ablationsData, setAblationQueued]);

  // handlers
  const handleRun = (mode, disableBo) => {
    const key = `${disableBo ? "ga_only" : mode}_${selectedJobId}`;
    // Synchronous guard via Zustand's getState() — no ref needed.
    if (useStore.getState().queuedAblations[key]) return;
    setAblationQueued(key, true);
    runAblationMutation.mutate(
      { mode, disable_bo: disableBo, parent_job_id: selectedJobId },
      {
        // Only clear on error — success (HTTP 202) just means the job was
        // accepted.  The key stays queued until the result appears.
        onError: () => setAblationQueued(key, false),
      },
    );
  };

  return (
    <div className="p-8">
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold text-foreground">
            Experiments
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Ablation study results and system comparison tables
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={() => refetchAblations()}
          disabled={ablationsLoading}
        >
          <RefreshCw size={12} className={ablationsLoading || isPolling ? "animate-spin" : ""} />
          {isPolling ? "Polling…" : "Refresh"}
        </Button>
      </div>

      {/* Job selector (sets dataset context for ablation lookup) */}
      <div className="mb-6">
        {jobsLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 size={15} className="animate-spin" />
            Loading jobs…
          </div>
        ) : jobsError ? (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertDescription>{jobsError.message}</AlertDescription>
          </Alert>
        ) : completedIds.length === 0 ? (
          <div className="rounded-xl border-2 border-dashed border-border bg-card p-12 text-center">
            <BarChart3
              className="mx-auto mb-3 text-muted-foreground/30"
              size={36}
              strokeWidth={1}
            />
            <p className="text-sm font-medium text-muted-foreground">No completed runs yet</p>
            <p className="mt-1 text-xs text-muted-foreground/60">
              Run an AutoML experiment first, then return here.
            </p>
          </div>
        ) : (
          <DropdownSelector
            label="Select a completed run"
            options={completedEntries.map(([id, job]) => ({
              value: id,
              label: `${id} — ${fmt.date(job?.start_time)}`,
            }))}
            value={selectedJobId}
            onChange={(newId) => setSearchParams({ job: newId })}
          />
        )}
      </div>

      {masterConfig && (
        <div className="mb-6">
          <JobConfigCard config={masterConfig} />
        </div>
      )}

      {/* Only render tables once a job is selected */}
      {selectedJobId && (
        <div className="space-y-8">
          {ablationsError && (
            <Alert variant="destructive">
              <AlertCircle />
              <AlertDescription>{ablationsError.message}</AlertDescription>
            </Alert>
          )}

          {/* Table 1: Single-Objective vs Multi-Objective */}
          <ComparisonTable
            title="Table 1: Single-Objective vs. Multi-Objective"
            subtitle="Compares optimising F1 alone against the full 3-objective formulation."
            headers={["Method", "Best F1", "Pareto Size", "Hypervolume", "Runtime"]}
            rows={[
              {
                label: "Single-Objective (F1 Only)",
                cells: [
                  { value: f4(single?.best_f1) },
                  { value: fSize(single?.pareto_front_size) },
                  { value: f4(single?.hypervolume) },
                  { value: fRuntime(singleRT) },
                ],
              },
              {
                label: "Multi-Objective (F1 + Latency + Interp)",
                sub: `Dataset: ${dataset}`,
                even: true,
                cells: [
                  { value: f4(masterMetrics?.best_f1), best: true },
                  { value: fSize(masterMetrics?.pareto_front_size), best: true },
                  { value: f4(masterMetrics?.hypervolume), best: true },
                  { value: fRuntime(masterRuntime) },
                ],
              },
            ]}
          />

          {/* Table 2: Ablation Studies */}
          <ComparisonTable
            title="Table 2: Ablation Studies"
            subtitle="Proves the contribution of BO and interpretability to overall quality."
            headers={["Configuration", "Best F1", "Hypervolume", "Runtime"]}
            rows={[
              {
                label: "Full GA + BO (3-Objective)",
                cells: [
                  { value: f4(masterMetrics?.best_f1), best: true },
                  { value: f4(masterMetrics?.hypervolume), best: true },
                  { value: fRuntime(masterRuntime) },
                ],
              },
              {
                label: "GA-Only — Random Hyperparams",
                sub: "Bayesian optimization disabled",
                even: true,
                cells: [
                  { value: f4(gaOnly?.best_f1) },
                  { value: f4(gaOnly?.hypervolume) },
                  { value: fRuntime(gaOnlyRT) },
                ],
              },
              {
                label: "2-Objective (No Interpretability)",
                cells: [
                  { value: f4(two?.best_f1) },
                  { value: f4(two?.hypervolume) },
                  { value: fRuntime(twoRT) },
                ],
              },
            ]}
          />

          {/* Ablation Bar Chart */}
          <section>
            <div className="mb-3">
              <h2 className="section-title">Ablation Comparison</h2>
              <p className="section-subtitle">
                Visual comparison of F1 and Hypervolume across configurations
              </p>
            </div>
            <div className="card-section">
              <Suspense
                fallback={
                  <div className="chart-empty h-90 gap-2">
                    <Loader2 size={16} className="animate-spin" />
                    Loading chart…
                  </div>
                }
              >
                <AblationBarChart
                  masterMetrics={masterMetrics}
                  single={single}
                  two={two}
                  gaOnly={gaOnly}
                />
              </Suspense>
            </div>
          </section>

          {/* Run missing ablation experiments */}
          <section>
            <div className="mb-3">
              <h2 className="flex items-center gap-2 section-title">
                <FlaskConical size={16} className="text-primary" />
                Run Missing Ablations
              </h2>
              <p className="section-subtitle">
                Jobs run in the background. Click{" "}
                <span className="font-medium text-foreground">Refresh</span> (or wait — it
                auto-polls) to see results appear in the tables above.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              {!single && (
                <RunButton
                  label="Run Single-Objective Baseline"
                  queued={!!queuedAblations[`single_f1_${selectedJobId}`]}
                  onClick={() => handleRun("single_f1", false)}
                />
              )}
              {!two && (
                <RunButton
                  label="Run 2-Objective (No Interp)"
                  queued={!!queuedAblations[`multi_2d_${selectedJobId}`]}
                  onClick={() => handleRun("multi_2d", false)}
                />
              )}
              {!gaOnly && (
                <RunButton
                  label="Run GA-Only (No BO)"
                  queued={!!queuedAblations[`ga_only_${selectedJobId}`]}
                  onClick={() => handleRun("multi_3d", true)}
                />
              )}

              {single && two && gaOnly && (
                <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm text-muted-foreground">
                  <span className="text-accent-foreground">✓</span>
                  All ablation runs complete for{" "}
                  <span className="font-medium text-foreground">{dataset}</span>.
                </div>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
};

export default Experiments;
