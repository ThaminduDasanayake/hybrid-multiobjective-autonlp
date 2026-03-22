import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, Trash2 } from "lucide-react";
import { DataTable, SortableHeader } from "@/components/ui/data-table";
import { Button } from "@/components/ui/button.jsx";
import { DATASETS } from "@/constants.js";
import { fmt } from "@/utils/formatters.js";

const DATASET_LABEL = Object.fromEntries(DATASETS.map((d) => [d.value, d.label]));

const JobSelectorTable = ({ jobs = {}, onDelete = () => {}, isDeleting }) => {
  const navigate = useNavigate();

  const tableData = useMemo(() => {
    return Object.entries(jobs).map(([id, job]) => ({
      _id: id,
      dataset_name: job.dataset_name ?? "—",
      start_time: job.start_time ?? 0,
      best_f1: job.best_f1 ?? 0,
      best_latency_ms: job.best_latency_ms ?? 0,
      best_interpretability: job.best_interpretability ?? 0,
    }));
  }, [jobs]);

  const columns = useMemo(
    () => [
      {
        accessorKey: "dataset_name",
        header: () => (
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Dataset
          </span>
        ),
        cell: ({ row }) => (
          <span className="text-sm font-medium text-foreground">
            {DATASET_LABEL[row.original.dataset_name] ?? row.original.dataset_name}
          </span>
        ),
        enableSorting: false,
      },
      {
        accessorKey: "start_time",
        header: ({ column }) => (
          <SortableHeader
            column={column}
            className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
          >
            Date
          </SortableHeader>
        ),
        cell: ({ row }) => (
          <span className="text-sm text-muted-foreground">{fmt.date(row.original.start_time)}</span>
        ),
      },
      {
        accessorKey: "best_f1",
        header: ({ column }) => (
          <div className="text-right">
            <SortableHeader
              column={column}
              className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
            >
              Best F1
            </SortableHeader>
          </div>
        ),
        cell: ({ row }) => (
          <div className="text-right font-mono text-sm tabular-nums text-foreground/80">
            {row.original.best_f1 > 0 ? row.original.best_f1.toFixed(4) : "—"}
          </div>
        ),
      },
      {
        accessorKey: "best_latency_ms",
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
          <div className="text-right font-mono text-sm tabular-nums text-foreground/80">
            {row.original.best_latency_ms > 0 ? row.original.best_latency_ms.toFixed(2) : "—"}
          </div>
        ),
      },
      {
        accessorKey: "best_interpretability",
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
          <div className="text-right font-mono text-sm tabular-nums text-foreground/80">
            {row.original.best_interpretability > 0
              ? row.original.best_interpretability.toFixed(4)
              : "—"}
          </div>
        ),
      },
      {
        id: "actions",
        header: () => (
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Actions
          </span>
        ),
        cell: ({ row }) => {
          const id = row.original._id;
          return (
            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate(`/history/${id}`)}
                className="h-7 px-2.5 text-xs"
              >
                <Eye size={13} />
                View
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  if (window.confirm(`Delete job ${id}? This cannot be undone.`)) {
                    onDelete(id);
                  }
                }}
                disabled={isDeleting}
                className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                aria-label={`Delete job ${id}`}
              >
                <Trash2 size={13} />
              </Button>
            </div>
          );
        },
        enableSorting: false,
      },
    ],
    [navigate, onDelete, isDeleting],
  );

  if (tableData.length === 0) return null;

  return (
    <DataTable
      columns={columns}
      data={tableData}
      initialSorting={[{ id: "start_time", desc: true }]}
      footerContent={
        <>
          {tableData.length} completed run{tableData.length !== 1 ? "s" : ""}
        </>
      }
    />
  );
};

export default JobSelectorTable;
