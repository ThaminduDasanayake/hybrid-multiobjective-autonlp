import { Badge } from "../ui/badge.jsx";
import { LABEL_MAP, STYLE_MAP } from "@/constants.js";

/**
 * StatusBadge — maps a job status string to a shadcn <Badge> variant.
 *
 * Props:
 *   status – "created" | "running" | "completed" | "failed" | "terminated"
 */

const StatusBadge = ({ status }) => {
  const { variant, className } = STYLE_MAP[status] ?? STYLE_MAP.created;

  return (
    <Badge variant={variant} className={className}>
      {status === "running" && (
        <span className="mr-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
      )}
      {LABEL_MAP[status] ?? status}
    </Badge>
  );
};

export default StatusBadge;
