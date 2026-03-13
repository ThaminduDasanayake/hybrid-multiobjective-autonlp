import { Card, CardContent } from "../ui/card";

/**
 * StatCard — live-run metric tile used in LiveTracker.
 *
 * Props:
 *   label  – uppercase metric name
 *   value  – formatted string to display
 *   unit   – optional unit suffix (smaller weight)
 *   dimmed – true when data has not yet arrived (fades the value)
 */
export default function StatCard({ label, value, unit, dimmed = false }) {
  return (
    <Card>
      <CardContent>
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
        <p
          className={`mt-1.5 text-2xl font-bold tabular-nums ${
            dimmed ? "text-muted-foreground/40" : "text-foreground"
          }`}
        >
          {value}
          {unit && <span className="ml-1 text-sm font-normal text-muted-foreground">{unit}</span>}
        </p>
      </CardContent>
    </Card>
  );
}
