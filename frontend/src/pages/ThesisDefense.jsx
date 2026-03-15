import { useEffect, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  FlaskConical,
  Loader2,
  Play,
  RefreshCw,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { useAblations, useJobResult, useJobs, useRunAblation } from "../hooks/useApi";
import DropdownSelector from "../components/DropdownSelector";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { fmt } from "../utils/formatters";
import JobConfigCard from "@/components/JobConfigCard.jsx";

function f4(v) {
  return v != null ? Number(v).toFixed(4) : null;
}
function fRuntime(secs) {
  if (secs == null) return null;
  return secs < 60 ? `${Number(secs).toFixed(1)} s` : `${(secs / 60).toFixed(1)} m`;
}

/** Styled pending pill shown in table cells when data is not yet available. */
function Pending() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
      <Loader2 size={9} className="animate-spin" />
      Run pending
    </span>
  );
}

/** A single data cell — value is a pre-formatted string or null (= pending). */
function Cell({ value, best = false }) {
  return (
    <td className="px-4 py-3 text-right">
      {value == null ? (
        <Pending />
      ) : (
        <span
          className={`font-mono tabular-nums text-sm ${
            best ? "font-semibold text-primary" : "text-foreground/80"
          }`}
        >
          {value}
        </span>
      )}
    </td>
  );
}

function Th({ children, right = false }) {
  return (
    <th
      className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground ${
        right ? "text-right" : "text-left"
      }`}
    >
      {children}
    </th>
  );
}

function Row({ label, sub, even, children }) {
  return (
    <tr className={`border-b border-border last:border-0 ${even ? "bg-muted/20" : ""}`}>
      <td className="px-4 py-3">
        <p className="text-sm font-medium text-foreground">{label}</p>
        {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      </td>
      {children}
    </tr>
  );
}

function ComparisonTable({ title, subtitle, children }) {
  return (
    <section>
      <div className="mb-3">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      <div className="overflow-hidden rounded-xl border border-border bg-card">
        <div className="overflow-x-auto">{children}</div>
      </div>
    </section>
  );
}

/** Button to queue a missing ablation. Shows "Queued…" spinner once pressed. */
function RunButton({ label, queued, onClick }) {
  return (
    <Button
      variant="outline"
      onClick={onClick}
      disabled={queued}
      className="border-primary/30 bg-primary/5 text-primary hover:bg-primary/10 hover:text-primary"
    >
      {queued ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
      {queued ? "Queued…" : label}
    </Button>
  );
}

const ThesisDefense = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobIdFromUrl = searchParams.get("job");

  // Tracks which ablation keys have been queued this session. Kept as local
  // state because one mutation can have multiple concurrent pending keys.
  const [queued, setQueued] = useState({});

  // Derived: true while at least one run is still queued
  const isPolling = Object.values(queued).some(Boolean);

  // ── server data ────────────────────────────────────────────────────────────
  const {
    data: jobMap = {},
    isLoading: jobsLoading,
    error: jobsError,
  } = useJobs();

  // Derivation chain: URL → localStorage → newest job
  const completedIds = Object.keys(jobMap);
  const lastSaved = localStorage.getItem("t_autonlp_last_ablation_job");
  const selectedJobId = jobIdFromUrl ?? lastSaved ?? completedIds[0];

  useEffect(() => {
    if (selectedJobId) {
      localStorage.setItem("t_autonlp_last_ablation_job", selectedJobId);
      if (jobIdFromUrl !== selectedJobId) {
        setSearchParams({ job: selectedJobId }, { replace: true });
      }
    }
  }, [selectedJobId, jobIdFromUrl, setSearchParams]);

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

  // ── derived values ─────────────────────────────────────────────────────────
  const masterMetrics = jobResult?.metrics ?? null;
  const masterRuntime = jobResult?.runtime_seconds ?? null;
  const masterConfig = jobResult?.config ?? null;

  const dataset =
    jobMap[selectedJobId]?._config?.dataset_name ?? jobMap[selectedJobId]?.dataset_name ?? "";

  const d = ablationsData ?? {};
  const single = d[`single_f1_${dataset}`]?.metrics ?? null;
  const singleRT = d[`single_f1_${dataset}`]?.runtime_seconds ?? null;
  const two = d[`multi_2d_${dataset}`]?.metrics ?? null;
  const twoRT = d[`multi_2d_${dataset}`]?.runtime_seconds ?? null;
  const gaOnly = d[`ga_only_${dataset}`]?.metrics ?? null;
  const gaOnlyRT = d[`ga_only_${dataset}`]?.runtime_seconds ?? null;

  // ── handlers ───────────────────────────────────────────────────────────────
  const handleRun = (mode, disableBo) => {
    const key = `${disableBo ? "ga_only" : mode}_${dataset}`;
    setQueued((q) => ({ ...q, [key]: true }));
    runAblationMutation.mutate(
      { mode, disable_bo: disableBo, dataset },
      { onError: () => setQueued((q) => ({ ...q, [key]: false })) },
    );
  };

  return (
    <div className="p-8">
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold text-foreground">
            Thesis Defense
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

      {/* ── Job selector (sets dataset context for ablation lookup) ──────── */}
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
            options={completedIds.map((id) => ({
              value: id,
              label: `${id} — ${fmt.date(jobMap[id]?.start_time)}`,
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
      {selectedJobId && dataset && (
        <div className="space-y-8">
          {ablationsError && (
            <Alert variant="destructive">
              <AlertCircle />
              <AlertDescription>{ablationsError.message}</AlertDescription>
            </Alert>
          )}

          {/* ── Table 1: Single-Objective vs Multi-Objective ─────────────── */}
          <ComparisonTable
            title="Table 1: Single-Objective vs. Multi-Objective"
            subtitle="Compares optimising F1 alone against the full 3-objective formulation."
          >
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-border bg-muted/60">
                  <Th>Method</Th>
                  <Th right>Best F1</Th>
                  <Th right>Pareto Size</Th>
                  <Th right>Hypervolume</Th>
                  <Th right>Runtime</Th>
                </tr>
              </thead>
              <tbody>
                <Row label="Single-Objective (F1 Only)" even={false}>
                  <Cell value={f4(single?.best_f1)} />
                  <Cell
                    value={
                      single?.pareto_front_size != null ? String(single.pareto_front_size) : null
                    }
                  />
                  <Cell value={f4(single?.hypervolume)} />
                  <Cell value={fRuntime(singleRT)} />
                </Row>
                <Row
                  label="Multi-Objective (F1 + Latency + Interp)"
                  sub={`Dataset: ${dataset}`}
                  even
                >
                  <Cell value={f4(masterMetrics?.best_f1)} best />
                  <Cell
                    value={
                      masterMetrics?.pareto_front_size != null
                        ? String(masterMetrics.pareto_front_size)
                        : null
                    }
                    best
                  />
                  <Cell value={f4(masterMetrics?.hypervolume)} best />
                  <Cell value={fRuntime(masterRuntime)} />
                </Row>
              </tbody>
            </table>
          </ComparisonTable>

          <ComparisonTable
            title="Table 2: Ablation Studies"
            subtitle="Proves the contribution of BO and interpretability to overall quality."
          >
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-border bg-muted/60">
                  <Th>Configuration</Th>
                  <Th right>Best F1</Th>
                  <Th right>Hypervolume</Th>
                  <Th right>Runtime</Th>
                </tr>
              </thead>
              <tbody>
                <Row label="Full GA + BO (3-Objective)" even={false}>
                  <Cell value={f4(masterMetrics?.best_f1)} best />
                  <Cell value={f4(masterMetrics?.hypervolume)} best />
                  <Cell value={fRuntime(masterRuntime)} />
                </Row>
                <Row label="GA-Only — Random Hyperparams" sub="Bayesian optimisation disabled" even>
                  <Cell value={f4(gaOnly?.best_f1)} />
                  <Cell value={f4(gaOnly?.hypervolume)} />
                  <Cell value={fRuntime(gaOnlyRT)} />
                </Row>
                <Row label="2-Objective (No Interpretability)" even={false}>
                  <Cell value={f4(two?.best_f1)} />
                  <Cell value={f4(two?.hypervolume)} />
                  <Cell value={fRuntime(twoRT)} />
                </Row>
              </tbody>
            </table>
          </ComparisonTable>

          {/* ── Run missing ablation experiments ─────────────────────────── */}
          <section>
            <div className="mb-3">
              <h2 className="flex items-center gap-2 text-base font-semibold text-foreground">
                <FlaskConical size={16} className="text-primary" />
                Run Missing Ablations
              </h2>
              <p className="text-xs text-muted-foreground">
                Jobs run in the background. Click{" "}
                <span className="font-medium text-foreground">Refresh</span> (or wait — it
                auto-polls) to see results appear in the tables above.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              {!single && (
                <RunButton
                  label="Run Single-Objective Baseline"
                  queued={!!queued[`single_f1_${dataset}`]}
                  onClick={() => handleRun("single_f1", false)}
                />
              )}
              {!two && (
                <RunButton
                  label="Run 2-Objective (No Interp)"
                  queued={!!queued[`multi_2d_${dataset}`]}
                  onClick={() => handleRun("multi_2d", false)}
                />
              )}
              {!gaOnly && (
                <RunButton
                  label="Run GA-Only (No BO)"
                  queued={!!queued[`ga_only_${dataset}`]}
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

export default ThesisDefense;
