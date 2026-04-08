// Table of Pareto-optimal solutions with relative metric coloring and knee-point highlighting.

import { useMemo, useCallback, useState } from "react";
import { fmt } from "@/utils/formatters.js";
import { Star, Code } from "lucide-react";
import { DataTable, SortableHeader } from "@/components/ui/data-table";
import CodeModal from "./CodeModal.jsx";

// Color-coded metric badge — green/orange/gray based on relative rank in the Pareto set (invert for latency).
const MetricBadge = ({ displayValue, rawValue, min, max, invert = false, decimals = 4 }) => {
  const n = Number(rawValue ?? 0);
  const range = max - min;
  const norm = range > 0 ? (invert ? 1 - (n - min) / range : (n - min) / range) : 0.5;
  const color =
    norm > 0.66
      ? "bg-secondary/20 text-secondary border-secondary/30"
      : norm > 0.33
        ? "bg-primary/20 text-primary border-primary/30"
        : "bg-muted text-muted-foreground border-border";

  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-xs font-semibold tabular-nums ${color}`}
    >
      {Number(displayValue ?? rawValue ?? 0).toFixed(decimals)}
    </span>
  );
};

// Small rounded tag for categorical pipeline values.
const Chip = ({ label }) => {
  if (!label || label === "None") return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <span className="inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
      {label}
    </span>
  );
};

// Inline label for notable rows (knee, best F1, etc.).
const RowLabel = ({ text, className }) => (
  <span
    className={`ml-1.5 inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${className}`}
  >
    {text}
  </span>
);

const SolutionsTable = ({ solutions = [], kneePoint = null }) => {
  const [codeModalSolution, setCodeModalSolution] = useState(null);

  // Stable row identity using key metrics so we can highlight notable solutions.
  const id = (s) => `${s.f1_score}_${s.latency}_${s.interpretability}_${s.model}`;

  // Pre-compute latency in ms and attach stable IDs for row matching.
  const tableData = useMemo(() => {
    if (solutions.length === 0) return [];
    const sorted = [...solutions].sort((a, b) => (b.f1_score ?? 0) - (a.f1_score ?? 0));
    return sorted.map((sol, idx) => ({
      ...sol,
      _rank: idx + 1,
      _latencyMs: (sol.latency ?? 0) * 1000,
      _solId: id(sol),
    }));
  }, [solutions]);

  // Min/max per metric so we can color badges relative to the current set.
  const { f1Range, latRange, interpRange } = useMemo(() => {
    if (tableData.length === 0)
      return {
        f1Range: { min: 0, max: 0 },
        latRange: { min: 0, max: 0 },
        interpRange: { min: 0, max: 0 },
      };
    const f1Values = tableData.map((s) => s.f1_score ?? 0);
    const latValues = tableData.map((s) => s._latencyMs);
    const interpValues = tableData.map((s) => s.interpretability ?? 0);
    const range = (arr) => ({ min: Math.min(...arr), max: Math.max(...arr) });
    return {
      f1Range: range(f1Values),
      latRange: range(latValues),
      interpRange: range(interpValues),
    };
  }, [tableData]);

  // Find the knee, best F1, fastest, and most interpretable solutions.
  const { kneeId, bestF1Id, bestSpeedId, bestInterpId } = useMemo(() => {
    if (tableData.length === 0)
      return { kneeId: null, bestF1Id: null, bestSpeedId: null, bestInterpId: null };
    const kneeId = kneePoint ? id(kneePoint) : null;
    const bestF1Id = tableData[0]._solId;
    const bestSpeed = tableData.reduce((b, s) => (s._latencyMs < b._latencyMs ? s : b));
    const bestInterp = tableData.reduce((b, s) =>
      (s.interpretability ?? 0) > (b.interpretability ?? 0) ? s : b,
    );
    return {
      kneeId,
      bestF1Id,
      bestSpeedId: bestSpeed._solId,
      bestInterpId: bestInterp._solId,
    };
  }, [tableData, kneePoint]);

  // Column definitions for the data table.
  const columns = useMemo(
    () => [
      {
        accessorKey: "_rank",
        header: () => (
          <span className="block text-center text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            #
          </span>
        ),
        cell: ({ row }) => {
          const isKnee = row.original._solId === kneeId;
          return (
            <div className="text-center">
              {isKnee ? (
                <span className="inline-flex h-5 w-5 items-center justify-center rounded-full font-bold">
                  <Star className="text-primary size-4 fill-primary" />
                </span>
              ) : (
                <span className="text-xs text-muted-foreground">{row.original._rank}</span>
              )}
            </div>
          );
        },
        enableSorting: false,
      },
      {
        accessorKey: "model",
        header: () => (
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Model
          </span>
        ),
        cell: ({ row }) => {
          const solId = row.original._solId;
          const isKnee = solId === kneeId;
          const isBestF1 = !isKnee && solId === bestF1Id;
          const isBestSpeed = !isKnee && !isBestF1 && solId === bestSpeedId;
          const isBestInterp = !isKnee && !isBestF1 && !isBestSpeed && solId === bestInterpId;
          return (
            <>
              <span
                className={`text-sm font-medium ${isKnee ? "text-foreground" : "text-foreground/80"}`}
              >
                {fmt.model(row.original.model)}
              </span>
              {isKnee && <RowLabel text="Knee Point" className="bg-chart-3/15 text-chart-3" />}
              {isBestF1 && <RowLabel text="Best F1" className="bg-primary/15 text-primary" />}
              {isBestSpeed && <RowLabel text="Fastest" className="bg-chart-5/15 text-chart-5" />}
              {isBestInterp && (
                <RowLabel text="Most Interpretable" className="bg-secondary/15 text-secondary" />
              )}
            </>
          );
        },
        enableSorting: false,
      },
      {
        accessorKey: "vectorizer",
        header: () => (
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Vectorizer
          </span>
        ),
        cell: ({ row }) => <Chip label={fmt.vectorizer(row.original.vectorizer)} />,
        enableSorting: false,
      },
      {
        accessorKey: "f1_score",
        header: ({ column }) => (
          <div className="text-right">
            <SortableHeader
              column={column}
              className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
            >
              F1 Score
            </SortableHeader>
          </div>
        ),
        cell: ({ row }) => (
          <div className="text-right">
            <MetricBadge
              rawValue={row.original.f1_score}
              min={f1Range.min}
              max={f1Range.max}
              decimals={4}
            />
          </div>
        ),
      },
      {
        accessorKey: "_latencyMs",
        header: ({ column }) => (
          <div className="text-right">
            <SortableHeader
              column={column}
              className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
            >
              Latency (ms)
            </SortableHeader>
          </div>
        ),
        cell: ({ row }) => (
          <div className="text-right">
            <MetricBadge
              rawValue={row.original._latencyMs}
              min={latRange.min}
              max={latRange.max}
              invert
              decimals={4}
            />
          </div>
        ),
      },
      {
        accessorKey: "interpretability",
        header: ({ column }) => (
          <div className="text-right">
            <SortableHeader
              column={column}
              className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
            >
              Interpretability
            </SortableHeader>
          </div>
        ),
        cell: ({ row }) => (
          <div className="text-right">
            <MetricBadge
              rawValue={row.original.interpretability}
              min={interpRange.min}
              max={interpRange.max}
              decimals={4}
            />
          </div>
        ),
      },
      {
        id: "actions",
        header: () => (
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Code
          </span>
        ),
        cell: ({ row }) => (
          <button
            onClick={() => setCodeModalSolution(row.original)}
            className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="Show pipeline code"
          >
            <Code className="size-4" />
          </button>
        ),
        enableSorting: false,
      },
    ],
    [kneeId, bestF1Id, bestSpeedId, bestInterpId, f1Range, latRange, interpRange],
  );

  // Row class name callback for knee-point highlighting.
  const getRowClassName = useCallback(
    (original) => (original._solId === kneeId ? "bg-chart-3/8" : ""),
    [kneeId],
  );

  if (tableData.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-10 text-center text-sm text-muted-foreground">
        No Pareto-optimal solutions to display.
      </div>
    );
  }

  const footerContent = (
    <>
      {tableData.length} Pareto-optimal solution{tableData.length !== 1 ? "s" : ""} — ★ marks the
      knee point (best balanced trade-off)
    </>
  );

  return (
    <>
      <DataTable
        columns={columns}
        data={tableData}
        getRowClassName={getRowClassName}
        footerContent={footerContent}
        initialSorting={[{ id: "f1_score", desc: true }]}
      />
      <CodeModal
        open={!!codeModalSolution}
        onClose={() => setCodeModalSolution(null)}
        solution={codeModalSolution}
      />
    </>
  );
};

export default SolutionsTable;
