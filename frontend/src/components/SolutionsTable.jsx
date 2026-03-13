/**
 * SolutionsTable
 *
 * Renders the Pareto-optimal pipeline solutions as a styled data table.
 * The "knee" solution (best balanced trade-off across all 3 objectives) is
 * highlighted with a distinct badge. Best F1, Best Speed, and Best
 * Interpretability rows get a small label too.
 *
 * Props:
 *   solutions – array from result.pareto_front
 */

import { fmt } from "../utils/formatters";
import { computeKnee } from "../utils/knee";
import { Star } from "lucide-react";

// ─── helpers ────────────────────────────────────────────────────────────────

/** Score → coloured badge (green / amber / red). */
const ScoreBadge = ({ value, unit = "" }) => {
  const n = Number(value ?? 0);
  const color =
    n >= 0.8
      ? "bg-secondary/20 text-secondary border-secondary/30"
      : n >= 0.6
        ? "bg-primary/20 text-primary border-primary/30"
        : "bg-destructive/20 text-destructive border-destructive/30";

  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-xs font-semibold tabular-nums ${color}`}
    >
      {n.toFixed(4)}
      {unit}
    </span>
  );
};

/** Interpretability pill. */
const InterpBadge = ({ value }) => {
  const n = Number(value ?? 0);
  const color =
    n >= 0.8
      ? "bg-secondary/20 text-secondary border-secondary/30"
      : n >= 0.5
        ? "bg-ring/15 text-ring border-ring/25"
        : "bg-muted text-muted-foreground border-border";

  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-xs font-semibold tabular-nums ${color}`}
    >
      {n.toFixed(3)}
    </span>
  );
};

/** Pipeline gene chip — small rounded tag for categorical values. */
const Chip = ({ label }) => {
  if (!label || label === "None") return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <span className="inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
      {label}
    </span>
  );
};

/** Small inline label for notable rows. */
const RowLabel = ({ text, className }) => (
  <span
    className={`ml-1.5 inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${className}`}
  >
    {text}
  </span>
);

// ─── column definitions ──────────────────────────────────────────────────────

const COLUMNS = [
  { key: "rank", label: "#", align: "center" },
  { key: "model", label: "Model" },
  { key: "vectorizer", label: "Vectorizer" },
  { key: "f1_score", label: "F1 Score", align: "right" },
  { key: "latency_ms", label: "Latency (ms)", align: "right" },
  { key: "interpretability", label: "Interpretability", align: "right" },
];

// ─── component ───────────────────────────────────────────────────────────────

const SolutionsTable = ({ solutions = [], kneePoint = null }) => {
  if (solutions.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-10 text-center text-sm text-muted-foreground">
        No Pareto-optimal solutions to display.
      </div>
    );
  }

  // Sort by F1 descending so rank 1 = best classifier.
  const sorted = [...solutions].sort((a, b) => (b.f1_score ?? 0) - (a.f1_score ?? 0));

  // Identify notable solutions.
  const knee = kneePoint || computeKnee(sorted);
  const bestF1 = sorted[0]; // already sorted
  const bestSpeed = sorted.reduce((b, s) =>
    (s.latency ?? Infinity) < (b.latency ?? Infinity) ? s : b,
  );
  const bestInterp = sorted.reduce((b, s) =>
    (s.interpretability ?? 0) > (b.interpretability ?? 0) ? s : b,
  );
  // Use a stable identity check (f1 + latency + interp) so we can mark rows.
  const id = (s) => `${s.f1_score}_${s.latency}_${s.interpretability}_${s.model}`;
  const kneeId = knee ? id(knee) : null;
  const bestF1Id = id(bestF1);
  const bestSpeedId = id(bestSpeed);
  const bestInterpId = id(bestInterp);

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          {/* ── Header ── */}
          <thead>
            <tr className="border-b border-border bg-muted/60">
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground ${
                    col.align === "right"
                      ? "text-right"
                      : col.align === "center"
                        ? "text-center"
                        : "text-left"
                  }`}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>

          {/* ── Rows ── */}
          <tbody>
            {sorted.map((sol, idx) => {
              const solId = id(sol);
              const isKnee = solId === kneeId;
              const isBestF1 = !isKnee && solId === bestF1Id;
              const isBestSpeed = !isKnee && !isBestF1 && solId === bestSpeedId;
              const isBestInterp = !isKnee && !isBestF1 && !isBestSpeed && solId === bestInterpId;

              return (
                <tr
                  key={idx}
                  className={`border-b border-border transition-colors last:border-0 hover:bg-muted/40 ${
                    isKnee ? "bg-chart-3/8" : ""
                  }`}
                >
                  {/* Rank */}
                  <td className="px-4 py-3 text-center">
                    {isKnee ? (
                      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-white font-bold">
                        <Star className="text-primary size-4 fill-primary" />
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">{idx + 1}</span>
                    )}
                  </td>

                  {/* Model */}
                  <td className="px-4 py-3">
                    <span
                      className={`text-sm font-medium ${
                        isKnee ? "text-foreground" : "text-foreground/80"
                      }`}
                    >
                      {fmt.model(sol.model)}
                    </span>
                    {isKnee && (
                      <RowLabel text="Knee Point" className="bg-chart-3/15 text-chart-3" />
                    )}
                    {isBestF1 && <RowLabel text="Best F1" className="bg-primary/15 text-primary" />}
                    {isBestSpeed && (
                      <RowLabel text="Fastest" className="bg-chart-5/15 text-chart-5" />
                    )}
                    {isBestInterp && (
                      <RowLabel
                        text="Most Interpretable"
                        className="bg-secondary/15 text-secondary"
                      />
                    )}
                  </td>

                  {/* Vectorizer */}
                  <td className="px-4 py-3">
                    <Chip label={fmt.vectorizer(sol.vectorizer)} />
                  </td>

                  {/* F1 Score */}
                  <td className="px-4 py-3 text-right">
                    <ScoreBadge value={sol.f1_score} />
                  </td>

                  {/* Latency (ms) */}
                  <td className="px-4 py-3 text-right">
                    <span className="font-mono text-xs tabular-nums text-foreground/70">
                      {fmt.latency_ms(sol.latency)}
                    </span>
                  </td>

                  {/* Interpretability */}
                  <td className="px-4 py-3 text-right">
                    <InterpBadge value={sol.interpretability} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="border-t border-border bg-muted/30 px-4 py-2 text-xs text-muted-foreground">
        {sorted.length} Pareto-optimal solution{sorted.length !== 1 ? "s" : ""} — ★ marks the knee
        point (best balanced trade-off)
      </div>
    </div>
  );
};

export default SolutionsTable;
