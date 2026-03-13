import { Badge } from "../ui/badge";

/**
 * StatusBadge — maps a job status string to a shadcn <Badge> variant.
 *
 * Props:
 *   status – "created" | "running" | "completed" | "failed" | "terminated"
 */

const STYLE_MAP = {
  created: { variant: "outline" },
  running: { variant: "outline", className: "border-primary/30 bg-primary/15 text-primary" },
  completed: { variant: "secondary" },
  failed: { variant: "destructive" },
  terminated: { variant: "ghost" },
};

const LABEL_MAP = {
  created: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  terminated: "Terminated",
};

export default function StatusBadge({ status }) {
  const { variant, className } = STYLE_MAP[status] ?? STYLE_MAP.created;

  return (
    <Badge variant={variant} className={className}>
      {status === "running" && (
        <span className="mr-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      )}
      {LABEL_MAP[status] ?? status}
    </Badge>
  );
}
