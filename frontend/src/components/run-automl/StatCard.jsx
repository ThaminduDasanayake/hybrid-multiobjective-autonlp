import { Card, CardContent } from "../ui/card.jsx";

// Metric tile for the live run tracker.
const StatCard = ({ label, value, unit, dimmed = false }) => {
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
};

export default StatCard;
