import { lazy, Suspense, useEffect } from "react";
import { AlertCircle, BarChart3, Box, Clock, Layers, Loader2, Star } from "lucide-react";
import { useJobResult, useJobs } from "../hooks/useApi";
import DecisionSupport from "../components/DecisionSupport";
import DropdownSelector from "../components/DropdownSelector";
import JobConfigCard from "../components/JobConfigCard";
import SolutionsTable from "../components/SolutionsTable";
import { Alert, AlertDescription } from "../components/ui/alert";
import { fmt } from "../utils/formatters";
import MetricCard from "@/components/MetricCard.jsx";
import { useSearchParams } from "react-router-dom";

// Lazy-load Plotly chart components so plotly.js (~3 MB) is only fetched when
// the user navigates to this page — not included in the initial app bundle.
const ParetoFront3D = lazy(() => import("../components/ParetoFront3D"));
const ParetoFront2D = lazy(() => import("../components/ParetoFront2D"));
const ConvergenceChart = lazy(() => import("../components/ConvergenceChart"));

const HistoryAnalysis = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobIdFromUrl = searchParams.get("job");

  const { data: jobMap = {}, isLoading: jobsLoading, error: jobsError } = useJobs();
  const completedIds = Object.keys(jobMap);

  // Derivation chain: URL → localStorage → newest job
  const lastSaved = localStorage.getItem("t_autonlp_last_history_job");
  const activeJobId = jobIdFromUrl ?? lastSaved ?? completedIds[0];

  useEffect(() => {
    if (activeJobId) {
      localStorage.setItem("t_autonlp_last_history_job", activeJobId);
      if (jobIdFromUrl !== activeJobId) {
        setSearchParams({ job: activeJobId }, { replace: true });
      }
    }
  }, [activeJobId, jobIdFromUrl, setSearchParams]);

  const handleJobSelect = (id) => setSearchParams({ job: id });

  const { data: jobData, isLoading: resultLoading, error: resultError } = useJobResult(activeJobId);
  const metrics = jobData?.metrics ?? null;
  const allSolutions = jobData?.all_solutions ?? [];
  const paretoFront = jobData?.pareto_front ?? [];
  const searchHistory = jobData?.search_history ?? [];
  const runtimeSecs = jobData?.runtime_seconds ?? null;

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground">History &amp; Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Explore completed runs, inspect the 3D Pareto front, and compare pipeline configurations.
        </p>
      </div>

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
              Run an AutoML experiment first, then return here to explore results.
            </p>
          </div>
        ) : (
          <DropdownSelector
            label="Select a completed run"
            options={completedIds.map((id) => ({
              value: id,
              label: `${id} — ${fmt.date(jobMap[id]?.start_time)}`,
            }))}
            value={activeJobId}
            onChange={handleJobSelect}
          />
        )}
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

          <section>
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-foreground">3D Pareto Front</h2>
                <p className="text-xs text-muted-foreground accent-teal-600">
                  Orange diamonds = Pareto-optimal · Grey = dominated solutions
                </p>
              </div>
              <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
                {allSolutions.length} solutions
              </span>
            </div>

            <div className="overflow-hidden rounded-xl border border-border bg-card p-2 shadow-sm">
              <Suspense
                fallback={
                  <div className="flex h-130 items-center justify-center gap-2 text-sm text-muted-foreground">
                    <Loader2 size={16} className="animate-spin" />
                    Loading chart…
                  </div>
                }
              >
                <ParetoFront3D allSolutions={allSolutions} paretoFront={paretoFront} />
              </Suspense>
            </div>
          </section>

          {/* ── 2D Pareto projections ────────────────────────────────── */}
          <section>
            <div className="mb-3">
              <h2 className="text-base font-semibold text-foreground">2D Pareto Projections</h2>
              <p className="text-xs text-muted-foreground">
                Orange = Pareto-optimal (connected) · Grey = dominated solutions
              </p>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="overflow-hidden rounded-xl border border-border bg-card p-2 shadow-sm">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  F1 Score vs. Latency
                </p>
                <Suspense
                  fallback={
                    <div className="flex h-90 items-center justify-center gap-2 text-sm text-muted-foreground">
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
              <div className="overflow-hidden rounded-xl border border-border bg-card p-2 shadow-sm">
                <p className="mb-1 px-2 text-xs font-medium text-muted-foreground">
                  F1 Score vs. Interpretability
                </p>
                <Suspense
                  fallback={
                    <div className="flex h-90 items-center justify-center gap-2 text-sm text-muted-foreground">
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
            </div>
          </section>

          <section>
            <div className="mb-3">
              <h2 className="text-base font-semibold text-foreground">
                Search History &amp; Convergence
              </h2>
              <p className="text-xs text-muted-foreground">
                Maximum F1 score found by the GA at each generation
              </p>
            </div>
            <div className="overflow-hidden rounded-xl border border-border bg-card p-2 shadow-sm">
              <Suspense
                fallback={
                  <div className="flex h-90 items-center justify-center gap-2 text-sm text-muted-foreground">
                    <Loader2 size={16} className="animate-spin" />
                    Loading chart…
                  </div>
                }
              >
                <ConvergenceChart searchHistory={searchHistory} />
              </Suspense>
            </div>
          </section>

          <section>
            <div className="mb-3">
              <h2 className="text-base font-semibold text-foreground">Pareto-Optimal Pipelines</h2>
              <p className="text-xs text-muted-foreground">
                Ranked by F1 score — all pipelines below are non-dominated
              </p>
            </div>

            <SolutionsTable solutions={paretoFront} kneePoint={metrics?.knee_point} />
          </section>
        </div>
      )}
    </div>
  );
};

export default HistoryAnalysis;
