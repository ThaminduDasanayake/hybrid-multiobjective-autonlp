import { Card, CardContent } from "./ui/card";
import { PARAM_KEYS, PARAM_LABELS } from "@/constants.js";

const JobConfigCard = ({ config }) => {
  if (!config) return null;

  const rows = PARAM_KEYS.filter((k) => config[k] != null);

  return (
    <Card>
      <CardContent>
        <h2 className="mb-3 text-sm font-semibold text-foreground">Run Configuration</h2>
        <dl className="grid md:flex justify-between gap-x-5 gap-y-3 sm:grid-cols-4">
          {rows.map((key) => (
            <div key={key}>
              <dt className="text-xs text-muted-foreground">{PARAM_LABELS[key]}</dt>
              <dd className="mt-0.5 text-sm font-medium text-foreground">{String(config[key])}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
};

export default JobConfigCard;
