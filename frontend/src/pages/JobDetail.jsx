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
import DecisionSupport from "../components/history-analysis/DecisionSupport.jsx";
import JobConfigCard from "../components/JobConfigCard";
import SolutionsTable from "../components/history-analysis/SolutionsTable.jsx";
import MetricCard from "@/components/history-analysis/MetricCard.jsx";
import ComparisonTable from "@/components/experiments/ComparisonTable.jsx";
import RunButton from "@/components/experiments/RunButton.jsx";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { fmt } from "../utils/formatters";

// Lazy-load Plotly chart components so plotly.js (~3 MB) is only fetched when
// the user navigates to this page — not included in the initial app bundle.
const ParetoFront3D = lazy(() => import("../components/history-analysis/ParetoFront3D.jsx"));
const ParetoFront2D = lazy(() => import("../components/history-analysis/ParetoFront2D.jsx"));
const ConvergenceChart = lazy(() => import("../components/history-analysis/ConvergenceChart.jsx"));
const HypervolumeConvergenceChart = lazy(
  () => import("../components/history-analysis/HypervolumeConvergenceChart.jsx"),
);
const ParetoHeatmap = lazy(() => import("../components/history-analysis/ParetoHeatmap.jsx"));
const PipelineBreakdownChart = lazy(
  () => import("../components/history-analysis/PipelineBreakdownChart.jsx"),
);
const ModelDistributionChart = lazy(
  () => import("../components/history-analysis/ModelDistributionChart.jsx"),
);
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
  const allSolutions = jobData?.all_solutions ?? [];
  const paretoFront = jobData?.pareto_front ?? [];
  const searchHistory = jobData?.search_history ?? [];
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
  } = useAblations({ refetchInterval: isPolling ? 10_000 : false });

  const runAblationMutation = useRunAblation();

  const masterMetrics = jobData?.metrics ?? null;
  const masterRuntime = jobData?.runtime_seconds ?? null;

  const d = ablationsData ?? {};
  const single = d[`single_f1_${jobId}`]?.metrics ?? null;
  const singleRT = d[`single_f1_${jobId}`]?.runtime_seconds ?? null;
  const two = d[`multi_2d_${jobId}`]?.metrics ?? null;
  const twoRT = d[`multi_2d_${jobId}`]?.runtime_seconds ?? null;
  const gaOnly = d[`ga_only_${jobId}`]?.metrics ?? null;
  const gaOnlyRT = d[`ga_only_${jobId}`]?.runtime_seconds ?? null;

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
      pareto_front: jobData.pareto_front,
      all_solutions: jobData.all_solutions,
      search_history: jobData.search_history,
      hypervolume_history: hvHistory,
      ablations: {
        single_objective: ablationsData?.[`single_f1_${jobId}`] ?? null,
        two_objective: ablationsData?.[`multi_2d_${jobId}`] ?? null,
        ga_only: ablationsData?.[`ga_only_${jobId}`] ?? null,
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
            onClick={() => navigate("/history")}
            className="mb-3 -ml-2 text-muted-foreground"
          >
            <ArrowLeft size={14} />
            Back to Run History
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={resultLoading || !jobData || hvLoading || ablationsLoading}
          >
            <Download size={14} />
            Export Research Bundle
          </Button>
        </div>
        <h1 className="text-2xl font-bold text-foreground">
          {dataset || "Job Detail"}
        </h1>
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
        <div className="space-y-6 mt-4">
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
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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

          {/* Solution Analysis */}
          <section>
            <div className="mb-3">
              <h2 className="section-title">Solution Analysis</h2>
              <p className="section-subtitle">
                Distribution of models and objective values across the search
              </p>
            </div>
            <div className="card-section">
              <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">Model Frequency</p>
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
          <section>
            <div className="mb-3">
              <h2 className="section-title">Pipeline Component Breakdown</h2>
              <p className="section-subtitle">
                Frequency of vectorizer, scaler, and dimensionality reduction choices — all
                solutions vs Pareto front
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
                <PipelineBreakdownChart allSolutions={allSolutions} paretoFront={paretoFront} />
              </Suspense>
            </div>
          </section>

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
          </div>
        </div>
      )}
    </div>
  );
};

export default JobDetail;
