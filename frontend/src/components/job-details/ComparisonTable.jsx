import { Info, Loader2 } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function Pending() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
      <Loader2 size={9} className="animate-spin" />
      Run pending
    </span>
  );
}

function GlobalMetricInfo() {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={(e) => e.preventDefault()}
          className="inline-flex cursor-default items-center text-muted-foreground/50 hover:text-muted-foreground transition-colors"
        >
          <Info size={12} />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-48 text-xs">
        Evaluated in global 3D space for baseline comparison.
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * ComparisonTable
 *
 * Data-driven comparison table built on shadcn Table primitives.
 *
 * Props:
 *   title    – section heading
 *   subtitle – optional description below the heading
 *   headers  – column header strings (first left-aligned, rest right-aligned)
 *   rows     – array of { label, sub?, cells: [{ value, best? }], even? }
 */
const ComparisonTable = ({ title, subtitle, headers = [], rows = [] }) => (
  <section>
    <div className="mb-3">
      <h2 className="section-title">{title}</h2>
      {subtitle && <p className="section-subtitle">{subtitle}</p>}
    </div>
    <div className="overflow-x-auto rounded-xl border border-border bg-card">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/60 hover:bg-muted/60">
            {headers.map((h, i) => (
              <TableHead
                key={h}
                className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground ${
                  i > 0 ? "text-right" : "text-left"
                }`}
              >
                {h}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, ri) => (
            <TableRow key={ri} className={`last:border-0 ${row.even ? "bg-muted/20" : ""}`}>
              <TableCell className="px-4 py-3">
                <p className="text-sm font-medium text-foreground">{row.label}</p>
                {row.sub && <p className="text-xs text-muted-foreground">{row.sub}</p>}
              </TableCell>
              {row.cells.map((cell, ci) => (
                <TableCell key={ci} className="px-4 py-3 text-right">
                  {cell.value == null ? (
                    <Pending />
                  ) : (
                    <span
                      className={`inline-flex items-center justify-end gap-1.5 font-mono tabular-nums text-sm ${
                        cell.passive
                          ? "text-muted-foreground/50"
                          : cell.best
                            ? "font-semibold text-primary"
                            : "text-foreground/80"
                      }`}
                    >
                      {cell.value}
                      {cell.passive && (
                        <span className="rounded bg-muted px-1 py-px font-sans text-[10px] font-medium leading-tight text-muted-foreground/60">
                          Passive
                        </span>
                      )}
                      {cell.globalMetric && <GlobalMetricInfo />}
                    </span>
                  )}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  </section>
);

export default ComparisonTable;
