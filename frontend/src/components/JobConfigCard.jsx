import { Card, CardContent } from "./ui/card";
import { PARAMS_CONFIG } from "@/constants.js";

const JobConfigCard = ({ config }) => {
  if (!config) return null;

  return (
    <Card>
      <CardContent>
        <h2 className="mb-3 text-sm font-semibold text-foreground">Run Configuration</h2>
        <dl className="grid grid-cols-2 gap-x-5 gap-y-3 sm:grid-cols-5">
          {PARAMS_CONFIG.map(({ id, label }) =>
            config[id] != null ? (
              <div key={id}>
                <dt className="text-xs text-muted-foreground">{label}</dt>
                <dd className="mt-0.5 text-sm font-medium text-foreground">{config[id]}</dd>
              </div>
            ) : null,
          )}
        </dl>
      </CardContent>
    </Card>
  );
};

export default JobConfigCard;
