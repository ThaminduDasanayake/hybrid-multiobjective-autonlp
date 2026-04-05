import { lazy, Suspense, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  AlertCircle,
  ArrowLeft,
  Box,
  Clock,
  Download,
  FlaskConical,
  Layers,
  Loader2,
  RefreshCw,
  Star,
} from "lucide-react";
import {
  useAblations,
  useHypervolumeHistory,
  useJobResult,
  useJobs,
  useRunAblation,
} from "../hooks/useApi";
import { useStore } from "../store";
import DecisionSupport from "../components/job-details/DecisionSupport.jsx";
import JobConfigCard from "../components/JobConfigCard";
import SolutionsTable from "../components/job-details/SolutionsTable.jsx";
import MetricCard from "@/components/job-details/MetricCard.jsx";
import ComparisonTable from "@/components/job-details/ComparisonTable.jsx";
import RunButton from "@/components/job-details/RunButton.jsx";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { fmt } from "../utils/formatters";

// Lazy-load Plotly chart components so plotly.js (~3 MB) is only fetched when
// the user navigates to this page — not included in the initial app bundle.
const ParetoFront3D = lazy(() => import("../components/job-details/ParetoFront3D.jsx"));
const ParetoFront2D = lazy(() => import("../components/job-details/ParetoFront2D.jsx"));
const ConvergenceChart = lazy(() => import("../components/job-details/ConvergenceChart.jsx"));
const HypervolumeConvergenceChart = lazy(
  () => import("../components/job-details/HypervolumeConvergenceChart.jsx"),
);
const ParetoHeatmap = lazy(() => import("../components/job-details/ParetoHeatmap.jsx"));
const PipelineBreakdownChart = lazy(
  () => import("../components/job-details/PipelineBreakdownChart.jsx"),
);
const ModelDistributionChart = lazy(
  () => import("../components/job-details/ModelDistributionChart.jsx"),
);
const AblationBarChart = lazy(() => import("@/components/job-details/AblationBarChart.jsx"));

function f4(v) {
  return v != null ? Number(v).toFixed(4) : null;
}
function fRuntime(secs) {
  if (secs == null) return null;
  return secs < 60 ? `${Number(secs).toFixed(1)} s` : `${(secs / 60).toFixed(1)} m`;
}
function fLatency(v) {
  return v != null ? Number(v).toFixed(4) : null;
}
function fSize(v) {
  return v != null ? String(v) : null;
}

const JobDetail = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();

  // Job metadata (for dataset name)
  const { data: jobMap = {} } = useJobs();
  const jobMeta = jobMap[jobId];
  const dataset = jobMeta?.dataset_name ?? "";

  // Result data (charts + metrics)
  const { data: jobData, isLoading: resultLoading, error: resultError } = useJobResult(jobId);
  const { data: hvHistory = [], isLoading: hvLoading } = useHypervolumeHistory(jobId);
  const metrics = jobData?.metrics ?? null;
  // Unified pipelines array — derive the three previous arrays from it so all
  // chart components below continue to receive the same prop names unchanged.
  const pipelines = jobData?.pipelines ?? [];
  const allSolutions = pipelines;
  const paretoFront = pipelines.filter((p) => p.is_pareto_optimal);
  const searchHistory = pipelines;
  const runtimeSecs = jobData?.runtime_seconds ?? null;

  // Ablation state
  const queuedAblations = useStore((s) => s.queuedAblations);
  const setAblationQueued = useStore((s) => s.setAblationQueued);
  const isPolling = Object.values(queuedAblations).some(Boolean);

  const {
    data: ablationsData,
    isLoading: ablationsLoading,
    error: ablationsError,
    refetch: refetchAblations,
  } = useAblations({ refetchInterval: isPolling ? 10_000 : false, parentJobId: jobId });

  const runAblationMutation = useRunAblation();

  const masterMetrics = jobData?.metrics ?? null;
  const masterRuntime = jobData?.runtime_seconds ?? null;

  const d = ablationsData ?? {};
  const singleData = d[`single_f1_${jobId}`];
  const single = singleData?.status === "completed" ? singleData.metrics : null;
  const singleRT = singleData?.status === "completed" ? singleData.runtime_seconds : null;
  const twoData = d[`multi_2d_${jobId}`];
  const two = twoData?.status === "completed" ? twoData.metrics : null;
  const twoRT = twoData?.status === "completed" ? twoData.runtime_seconds : null;
  const gaOnlyData = d[`ga_only_${jobId}`];
  const gaOnly = gaOnlyData?.status === "completed" ? gaOnlyData.metrics : null;
  const gaOnlyRT = gaOnlyData?.status === "completed" ? gaOnlyData.runtime_seconds : null;
  const randomSearchData = d[`random_search_${jobId}`];
  const randomSearch = randomSearchData?.status === "completed" ? randomSearchData.metrics : null;
  const randomSearchRT =
    randomSearchData?.status === "completed" ? randomSearchData.runtime_seconds : null;

  // Scroll to top when navigating to a new job
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [jobId]);

  // Auto-clear queued keys once their results appear in the ablations response.
  useEffect(() => {
    if (!ablationsData) return;
    const current = useStore.getState().queuedAblations;
    for (const key of Object.keys(current)) {
      if (current[key] && ablationsData[key]) {
        setAblationQueued(key, false);
      }
    }
  }, [ablationsData, setAblationQueued]);

  const handleRun = (mode, disableBo) => {
    const key = `${disableBo ? "ga_only" : mode}_${jobId}`;
    if (useStore.getState().queuedAblations[key]) return;
    setAblationQueued(key, true);
    runAblationMutation.mutate(
      { mode, disable_bo: disableBo, parent_job_id: jobId },
      { onError: () => setAblationQueued(key, false) },
    );
  };

  const handleExport = () => {
    const bundle = {
      exported_at: new Date().toISOString(),
      job_id: jobId,
      dataset,
      config: jobData.config,
      metrics: jobData.metrics,
      runtime_seconds: jobData.runtime_seconds,
      pipelines: jobData.pipelines,
      hypervolume_history: hvHistory,
      ablations: {
        single_objective: ablationsData?.[`single_f1_${jobId}`] ?? null,
        two_objective: ablationsData?.[`multi_2d_${jobId}`] ?? null,
        ga_only: ablationsData?.[`ga_only_${jobId}`] ?? null,
        random_search: ablationsData?.[`random_search_${jobId}`] ?? null,
      },
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research_bundle_${jobId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/")}
            className="mb-3 -ml-2 text-muted-foreground"
          >
            <ArrowLeft size={14} />
            Back to Home
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleExport}
            disabled={resultLoading || !jobData || hvLoading || ablationsLoading}
          >
            <Download size={14} />
            Export Research Bundle
          </Button>
        </div>
        <h1 className="text-2xl font-bold text-foreground">Job Details</h1>
        <p className="mt-1 font-mono text-sm text-muted-foreground">
          {jobId} — {fmt.date(jobMeta?.start_time)}
        </p>
      </div>

      {resultLoading && (
        <div className="flex items-center gap-2 py-16 text-sm text-muted-foreground">
          <Loader2 size={16} className="animate-spin" />
          Loading results…
        </div>
      )}

      {resultError && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertDescription>{resultError.message}</AlertDescription>
        </Alert>
      )}

      <JobConfigCard config={jobData?.config} />

      {jobData && !resultLoading && (
        <div className="space-y-8 mt-4">
          {/* Metric Cards */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard
              icon={Layers}
              label="Pipelines Explored"
              value={metrics?.total_solutions ?? allSolutions.length}
              sub="total configurations evaluated"
            />
            <MetricCard
              icon={Box}
              label="Hypervolume"
              value={metrics?.hypervolume ? Number(metrics.hypervolume).toFixed(4) : "—"}
              sub="objective space coverage"
            />
            <MetricCard
              icon={Star}
              label="Pareto Front"
              value={metrics?.pareto_front_size ?? paretoFront.length}
              sub={`of ${metrics?.total_solutions ?? allSolutions.length} total`}
            />
            <MetricCard
              icon={Clock}
              label="Runtime"
              value={
                runtimeSecs != null
                  ? runtimeSecs < 60
                    ? `${runtimeSecs.toFixed(0)}s`
                    : `${(runtimeSecs / 60).toFixed(1)}m`
                  : "—"
              }
              sub="wall-clock time"
            />
          </div>

          <DecisionSupport paretoFront={paretoFront} kneePoint={metrics?.knee_point} />

          {/* 3D Pareto Front */}
          <section>
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h2 className="section-title">3D Pareto Front</h2>
                <ul className="section-subtitle">
                  <li>Orange = Pareto-optimal</li>
                  <li>Grey = dominated solutions</li>
                </ul>
              </div>
              <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
                {allSolutions.length} solutions
              </span>
            </div>
            <div className="card-section">
              <Suspense
                fallback={
                  <div className="chart-empty h-130 gap-2">
                    <Loader2 size={16} className="animate-spin" />
                    Loading chart…
                  </div>
                }
              >
                <ParetoFront3D allSolutions={allSolutions} paretoFront={paretoFront} />
              </Suspense>
            </div>
          </section>

          {/* 2D Pareto Projections */}
          <section>
            <div className="mb-3">
              <h2 className="section-title">2D Pareto Projections</h2>
              <ul className="section-subtitle">
                <li>Orange = Pareto-optimal (connected)</li>
                <li>Grey = dominated solutions</li>
              </ul>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="card-section">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  F1 Score vs. Latency
                </p>
                <Suspense
                  fallback={
                    <div className="chart-empty h-90 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <ParetoFront2D
                    allSolutions={allSolutions}
                    paretoFront={paretoFront}
                    xKey="latency"
                    yKey="f1_score"
                    xLabel="Latency (ms) ↓"
                    yLabel="F1 Score ↑"
                    xScale={1000}
                  />
                </Suspense>
              </div>
              <div className="card-section">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  F1 Score vs. Interpretability
                </p>
                <Suspense
                  fallback={
                    <div className="chart-empty h-90 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <ParetoFront2D
                    allSolutions={allSolutions}
                    paretoFront={paretoFront}
                    xKey="interpretability"
                    yKey="f1_score"
                    xLabel="Interpretability ↑"
                    yLabel="F1 Score ↑"
                  />
                </Suspense>
              </div>
              <div className="card-section">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  Latency vs. Interpretability
                </p>
                <Suspense
                  fallback={
                    <div className="chart-empty h-90 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <ParetoFront2D
                    allSolutions={allSolutions}
                    paretoFront={paretoFront}
                    xKey="latency"
                    yKey="interpretability"
                    xLabel="Latency (ms) ↓"
                    yLabel="Interpretability ↑"
                    xScale={1000}
                  />
                </Suspense>
              </div>
            </div>
          </section>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
            {/* Solution Analysis */}
            <section className="flex flex-col lg:col-span-2">
              <div className="mb-3">
                <h2 className="section-title">Solution Analysis</h2>
                <p className="section-subtitle">
                  Distribution of models and objective values across the search
                </p>
              </div>
              <div className="card-section flex-1">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  Model Frequency
                </p>
                <Suspense
                  fallback={
                    <div className="chart-empty h-90 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <ModelDistributionChart allSolutions={allSolutions} paretoFront={paretoFront} />
                </Suspense>
              </div>
            </section>

            {/* Pipeline Component Breakdown */}
            <section className="flex flex-col lg:col-span-3">
              <div className="mb-3">
                <h2 className="section-title">Pipeline Component Breakdown</h2>
                <p className="section-subtitle">
                  Frequency of vectorizer, scaler, and dimensionality reduction choices — all
                  solutions vs Pareto front
                </p>
              </div>
              <div className="card-section flex-1">
                <Suspense
                  fallback={
                    <div className="chart-empty h-40 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <PipelineBreakdownChart allSolutions={allSolutions} paretoFront={paretoFront} />
                </Suspense>
              </div>
            </section>
          </div>

          {/* Pareto Front Trade-off Heatmap */}
          <section>
            <div className="mb-3">
              <h2 className="section-title">Pareto Front Trade-off Analysis</h2>
              <p className="section-subtitle">
                Each row is one Pareto-optimal pipeline — orange = high value (good), dark = low
                value. Latency is inverted so orange always means better.
              </p>
            </div>
            <div className="card-section">
              <Suspense
                fallback={
                  <div className="chart-empty h-40 gap-2">
                    <Loader2 size={16} className="animate-spin" />
                    Loading chart…
                  </div>
                }
              >
                <ParetoHeatmap paretoFront={paretoFront} />
              </Suspense>
            </div>
          </section>

          {/* Convergence Analysis */}
          <section>
            <div className="mb-3">
              <h2 className="section-title">Convergence Analysis</h2>
              <p className="section-subtitle">
                F1 score and hypervolume progression across GA generations
              </p>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="card-section">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  Max F1 per Generation
                </p>
                <Suspense
                  fallback={
                    <div className="chart-empty h-90 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <ConvergenceChart searchHistory={searchHistory} />
                </Suspense>
              </div>
              <div className="card-section">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  Hypervolume per Generation
                </p>
                <Suspense
                  fallback={
                    <div className="chart-empty h-90 gap-2">
                      <Loader2 size={16} className="animate-spin" />
                      Loading chart…
                    </div>
                  }
                >
                  <HypervolumeConvergenceChart hvHistory={hvHistory} />
                </Suspense>
              </div>
            </div>
          </section>

          {/* Pareto-Optimal Pipelines */}
          <section>
            <div className="mb-3">
              <h2 className="section-title">Pareto-Optimal Pipelines</h2>
              <p className="section-subtitle">
                Ranked by F1 score — all pipelines below are non-dominated
              </p>
            </div>
            <SolutionsTable solutions={paretoFront} kneePoint={metrics?.knee_point} />
          </section>

          {/* ── Experiments Section ── */}
          <div className="border-t border-border pt-8">
            <div className="mb-6 flex items-start justify-between">
              <div>
                <h2 className="flex items-center gap-2 text-xl font-bold text-foreground">
                  <FlaskConical size={18} className="text-primary" />
                  Ablation Experiments
                </h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Compare optimization modes and validate each component's contribution
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchAblations()}
                disabled={ablationsLoading}
              >
                <RefreshCw
                  size={12}
                  className={ablationsLoading || isPolling ? "animate-spin" : ""}
                />
                {isPolling ? "Polling…" : "Refresh"}
              </Button>
            </div>

            {ablationsError && (
              <Alert variant="destructive" className="mb-6">
                <AlertCircle />
                <AlertDescription>{ablationsError.message}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-8">
              {/* Table 1: Comprehensive Evaluation & Ablation Studies */}
              <ComparisonTable
                title="Table 1: Comprehensive Evaluation & Ablation Studies"
                subtitle="Comparison of the proposed 3-objective architecture against single-objective baselines and component ablations."
                headers={[
                  "Configuration",
                  "Best F1 ↑",
                  "Latency (ms) ↓",
                  "Interp. ↑",
                  "Pareto Size",
                  "Hypervolume ↑",
                  "Runtime",
                ]}
                rows={[
                  {
                    label: "Full GA + BO (Proposed 3-Objective)",
                    cells: [
                      { value: f4(masterMetrics?.best_f1), best: true },
                      { value: fLatency(masterMetrics?.best_latency_ms), best: true },
                      { value: f4(masterMetrics?.best_interpretability), best: true },
                      { value: fSize(masterMetrics?.pareto_front_size), best: true },
                      { value: f4(masterMetrics?.hypervolume), best: true },
                      { value: fRuntime(masterRuntime), best: true },
                    ],
                  },
                  {
                    label: "2-Objective (No Interpretability)",
                    even: true,
                    cells: [
                      { value: f4(two?.best_f1) },
                      { value: fLatency(two?.best_latency_ms) },
                      { value: f4(two?.best_interpretability), passive: true },
                      { value: fSize(two?.pareto_front_size), globalMetric: true },
                      { value: f4(two?.hypervolume), globalMetric: true },
                      { value: fRuntime(twoRT) },
                    ],
                  },
                  {
                    label: "GA-Only — Random Hyperparams",
                    sub: "Bayesian optimization disabled",
                    cells: [
                      { value: f4(gaOnly?.best_f1) },
                      { value: fLatency(gaOnly?.best_latency_ms) },
                      { value: f4(gaOnly?.best_interpretability) },
                      { value: fSize(gaOnly?.pareto_front_size) },
                      { value: f4(gaOnly?.hypervolume) },
                      { value: fRuntime(gaOnlyRT) },
                    ],
                  },
                  {
                    label: "Single-Objective (F1 Only)",
                    even: true,
                    cells: [
                      { value: f4(single?.best_f1) },
                      { value: fLatency(single?.best_latency_ms), passive: true },
                      { value: f4(single?.best_interpretability), passive: true },
                      { value: fSize(single?.pareto_front_size), globalMetric: true },
                      { value: f4(single?.hypervolume), globalMetric: true },
                      { value: fRuntime(singleRT) },
                    ],
                  },
                  {
                    label: "Random Search Baseline",
                    sub: "No GA, no BO — pure random sampling",
                    cells: [
                      { value: f4(randomSearch?.best_f1) },
                      { value: fLatency(randomSearch?.best_latency_ms) },
                      { value: f4(randomSearch?.best_interpretability) },
                      { value: fSize(randomSearch?.pareto_front_size ?? 1) },
                      { value: f4(randomSearch?.hypervolume) },
                      { value: fRuntime(randomSearchRT) },
                    ],
                  },
                ]}
              />

              {/* Ablation Bar Chart */}
              <section>
                <div className="mb-3">
                  <h2 className="section-title">Ablation Comparison</h2>
                  <p className="section-subtitle">
                    Visual comparison of all objectives and hypervolume across configurations
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
                      randomSearch={randomSearch}
                    />
                  </Suspense>
                </div>
              </section>

              {/* Run Missing Ablations */}
              <section>
                <div className="mb-3">
                  <h2 className="section-title">Run Missing Ablations</h2>
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
                      queued={!!queuedAblations[`single_f1_${jobId}`]}
                      onClick={() => handleRun("single_f1", false)}
                    />
                  )}
                  {!two && (
                    <RunButton
                      label="Run 2-Objective (No Interp)"
                      queued={!!queuedAblations[`multi_2d_${jobId}`]}
                      onClick={() => handleRun("multi_2d", false)}
                    />
                  )}
                  {!gaOnly && (
                    <RunButton
                      label="Run GA-Only (No BO)"
                      queued={!!queuedAblations[`ga_only_${jobId}`]}
                      onClick={() => handleRun("multi_3d", true)}
                    />
                  )}
                  {!randomSearch && (
                    <RunButton
                      label="Run Random Search Baseline"
                      queued={!!queuedAblations[`random_search_${jobId}`]}
                      onClick={() => handleRun("random_search", false)}
                    />
                  )}
                  {single && two && gaOnly && randomSearch && (
                    <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm text-muted-foreground">
                      <span className="text-accent-foreground">✓</span>
                      All ablation runs complete for{" "}
                      <span className="font-medium text-foreground">{dataset}</span>.
                    </div>
                  )}
                </div>
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default JobDetail;
