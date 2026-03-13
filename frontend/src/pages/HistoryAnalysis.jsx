import { lazy, Suspense, useEffect, useState } from "react";
import { AlertCircle, Box, Clock, Layers, Loader2, Star, TrendingUp, Zap } from "lucide-react";
import { getJobResult, getJobs } from "../api";
import DecisionSupport from "../components/DecisionSupport";
import DropdownSelector from "../components/DropdownSelector";
import JobConfigCard from "../components/JobConfigCard";
import SolutionsTable from "../components/SolutionsTable";
import { Alert, AlertDescription } from "../components/ui/alert";
import { fmt } from "../utils/formatters";
import MetricCard from "@/components/MetricCard.jsx";

// Lazy-load Plotly chart components so plotly.js (~3 MB) is only fetched when
// the user navigates to this page — not included in the initial app bundle.
const ParetoFront3D = lazy(() => import("../components/ParetoFront3D"));
const ParetoFront2D = lazy(() => import("../components/ParetoFront2D"));
const ConvergenceChart = lazy(() => import("../components/ConvergenceChart"));

const HistoryAnalysis = () => {
  // ── job list ────────────────────────────────────────────────────────────
  const [jobMap, setJobMap] = useState({}); // { job_id: statusObj }
  const [jobsLoading, setJobsLoading] = useState(true);
  const [jobsError, setJobsError] = useState(null);

  // ── selected job ────────────────────────────────────────────────────────
  const [selectedId, setSelectedId] = useState("");
  const [jobData, setJobData] = useState(null); // result.json payload
  const [resultLoading, setResultLoading] = useState(false);
  const [resultError, setResultError] = useState(null);

  // Fetch the job list once on mount.
  useEffect(() => {
    getJobs()
      .then((data) => {
        // Keep only completed jobs — incomplete runs have no result.json.
        const completed = Object.fromEntries(
          Object.entries(data).filter(([, s]) => s.status === "completed"),
        );
        setJobMap(completed);

        // Auto-select the most recent run.
        const ids = Object.keys(completed);
        if (ids.length > 0) setSelectedId(ids[0]);
      })
      .catch((err) => setJobsError(err.message))
      .finally(() => setJobsLoading(false));
  }, []);

  // Fetch result whenever the selected job changes.
  useEffect(() => {
    if (!selectedId) {
      setJobData(null);
      return;
    }
    setResultLoading(true);
    setResultError(null);
    setJobData(null);

    getJobResult(selectedId)
      .then(setJobData)
      .catch((err) => setResultError(err.message))
      .finally(() => setResultLoading(false));
  }, [selectedId]);

  // ── derived values ───────────────────────────────────────────────────────
  const completedIds = Object.keys(jobMap);
  const metrics = jobData?.metrics ?? null;
  const allSolutions = jobData?.all_solutions ?? [];
  const paretoFront = jobData?.pareto_front ?? [];
  const searchHistory = jobData?.search_history ?? [];
  const runtimeSecs = jobData?.runtime_seconds ?? null;

  // ── render ───────────────────────────────────────────────────────────────
  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-foreground">History &amp; Analysis</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Explore completed runs, inspect the 3D Pareto front, and compare pipeline configurations.
        </p>
      </div>

      {/* ── Job selector ────────────────────────────────────────────────── */}
      <div className="mb-6">
        {jobsLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 size={15} className="animate-spin" />
            Loading jobs…
          </div>
        ) : jobsError ? (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertDescription>{jobsError}</AlertDescription>
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
            value={selectedId}
            onChange={setSelectedId}
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
          <AlertDescription>{resultError}</AlertDescription>
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

          <DecisionSupportOld kneePoint={metrics?.knee_point} paretoFront={paretoFront} />

          {/* ── Decision support cards ───────────────────────────────── */}
          <DecisionSupport paretoFront={paretoFront} kneePoint={metrics?.knee_point} />

          {/* ── 3D Pareto chart ──────────────────────────────────────── */}
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
                  <div className="flex h-[520px] items-center justify-center gap-2 text-sm text-muted-foreground">
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
                    <div className="flex h-[360px] items-center justify-center gap-2 text-sm text-muted-foreground">
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
                    <div className="flex h-[360px] items-center justify-center gap-2 text-sm text-muted-foreground">
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

          {/* ── Convergence chart ────────────────────────────────────── */}
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
                  <div className="flex h-[360px] items-center justify-center gap-2 text-sm text-muted-foreground">
                    <Loader2 size={16} className="animate-spin" />
                    Loading chart…
                  </div>
                }
              >
                <ConvergenceChart searchHistory={searchHistory} />
              </Suspense>
            </div>
          </section>

          {/* ── Pareto solutions table ───────────────────────────────── */}
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
import { GitMerge } from "lucide-react";
import { computeKnee } from "../utils/knee";
import { Card, CardContent } from "@/components/ui/card.jsx";

const pick = (arr, key, dir) =>
  arr.reduce((best, sol) =>
    dir === 1 ? (sol[key] > best[key] ? sol : best) : sol[key] < best[key] ? sol : best,
  );

const RecommendCard = ({ title, icon: Icon, metricLabel, metricValue, pipeline }) => (
  <Card className="flex-1 bg-card shadow-sm border-border">
    <CardContent className="p-5">
      <div className="mb-3 flex flex-row items-center justify-between">
        <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
        <Icon size={18} className="text-muted-foreground" />
      </div>
      <div>
        <p className="text-3xl font-bold tracking-tight text-foreground">{metricValue}</p>
        <p className="mt-1 text-xs text-muted-foreground">{metricLabel}</p>
      </div>
      <div className="mt-4 space-y-1.5 border-t border-border pt-3">
        {[
          ["Model", "model"],
          ["Vectorizer", "vectorizer"],
          ["Scaler", "scaler"],
        ].map(([label, key]) => (
          <div key={key} className="flex justify-between text-xs">
            <span className="text-muted-foreground">{label}</span>
            <span className="font-medium text-foreground capitalize">
              {pipeline[key] ?? "none"}
            </span>
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
);

const DecisionSupportOld = ({ paretoFront, kneePoint }) => {
  if (!paretoFront || paretoFront.length === 0) return null;

  const bestAccuracy = pick(paretoFront, "f1_score", 1);
  const bestSpeed = pick(paretoFront, "latency", -1);
  const bestInterp = pick(paretoFront, "interpretability", 1);
  const knee = kneePoint || computeKnee(paretoFront);

  return (
    <section>
      <div className="mb-3">
        <h2 className="text-base font-semibold text-foreground">Decision Support</h2>
        <p className="text-xs text-muted-foreground">
          Four recommended pipelines from the Pareto front
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <RecommendCard
          title="Knee Point ★"
          icon={GitMerge}
          metricLabel="Balanced trade-off"
          metricValue={knee.f1_score.toFixed(4)}
          pipeline={knee}
        />
        <RecommendCard
          title="Best Accuracy"
          icon={TrendingUp}
          metricLabel="F1 Score"
          metricValue={bestAccuracy.f1_score.toFixed(4)}
          pipeline={bestAccuracy}
        />
        <RecommendCard
          title="Best Speed"
          icon={Zap}
          metricLabel="Latency"
          metricValue={`${(bestSpeed.latency * 1000).toFixed(4)} ms`}
          pipeline={bestSpeed}
        />
        <RecommendCard
          title="Best Interp."
          icon={Star}
          metricLabel="Interpretability Score"
          metricValue={bestInterp.interpretability.toFixed(4)}
          pipeline={bestInterp}
        />
      </div>
    </section>
  );
};
