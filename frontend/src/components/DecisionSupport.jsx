import { useState } from "react";
import { ChevronDown, ChevronRight, Star, TrendingUp, Zap, Lightbulb } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { computeKnee } from "../utils/knee";

const pick = (arr, key, dir) =>
  arr.reduce((best, sol) =>
    dir === 1 ? (sol[key] > best[key] ? sol : best) : sol[key] < best[key] ? sol : best,
  );

const cleanParams = (params) => {
  if (!params) return {};
  const clean = { ...params };
  Object.keys(clean).forEach((k) => {
    if (k.endsWith("_type")) delete clean[k];
  });
  return clean;
};

const HyperparametersToggle = ({ params }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-4 flex flex-col pt-3 border-t border-border">
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className="w-full justify-between px-2 text-xs font-medium text-foreground"
      >
        View Hyperparameters
        {expanded ? (
          <ChevronDown size={14} className="text-muted-foreground" />
        ) : (
          <ChevronRight size={14} className="text-muted-foreground" />
        )}
      </Button>
      {expanded && (
        <div className="mt-2 rounded-md bg-muted/40 p-3 border border-border/40">
          <pre className="text-[10px] leading-relaxed text-muted-foreground overflow-x-auto">
            {JSON.stringify(cleanParams(params), null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

const MetricRow = ({ label, value }) => (
  <div className="flex justify-between items-center py-1 border-b border-border/30 last:border-0 last:pb-0">
    <span className="text-xs text-muted-foreground">{label}</span>
    <span className="font-mono text-xs font-semibold text-foreground">{value}</span>
  </div>
);

const RecommendCard = ({ title, icon: Icon, metricLabel, metricValue, pipeline }) => (
  <Card className="flex flex-col bg-card shadow-sm border-border">
    <CardContent className="p-5 flex flex-col h-full">
      {/* Header */}
      <div className="mb-4 flex flex-row items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">{title}</h3>
        <Icon size={16} className="text-muted-foreground" />
      </div>

      {/* Primary Highlight */}
      <div className="mb-5">
        <p className="text-3xl font-bold tracking-tight text-foreground">{metricValue}</p>
        <p className="mt-1 text-xs text-muted-foreground">{metricLabel}</p>
      </div>

      {/* Pipeline Config Mini-Table */}
      <div className="mb-4 space-y-1.5 bg-muted/20 p-2.5 rounded-lg border border-border/40">
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Model</span>
          <span className="font-medium text-foreground capitalize">{pipeline.model ?? "none"}</span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Vectorizer</span>
          <span className="font-medium text-foreground capitalize">
            {pipeline.vectorizer ?? "none"}
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Scaler</span>
          <span className="font-medium text-foreground capitalize">
            {pipeline.scaler ?? "none"}
          </span>
        </div>
      </div>

      {/* All Objectives */}
      <div className="space-y-1 mt-auto bg-muted/10 p-2.5 rounded-lg border border-border/30">
        <MetricRow label="F1 Score" value={Number(pipeline.f1_score ?? 0).toFixed(4)} />
        <MetricRow
          label="Latency"
          value={`${(Number(pipeline.latency ?? 0) * 1000).toFixed(2)} ms`}
        />
        <MetricRow
          label="Interpretability"
          value={Number(pipeline.interpretability ?? 0).toFixed(4)}
        />
      </div>

      <HyperparametersToggle params={pipeline.params} />
    </CardContent>
  </Card>
);

const DecisionSupport = ({ paretoFront, kneePoint }) => {
  if (!paretoFront || paretoFront.length === 0) return null;

  const bestAccuracy = pick(paretoFront, "f1_score", 1);
  const bestSpeed = pick(paretoFront, "latency", -1);
  const bestInterpretable = pick(paretoFront, "interpretability", 1);
  const knee = kneePoint || computeKnee(paretoFront);

  return (
    <section>
      <div className="mb-4">
        <h2 className="text-lg font-semibold tracking-tight text-foreground">Decision Support</h2>
        <p className="text-sm text-muted-foreground">
          Four recommended pipelines from the Pareto front
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <RecommendCard
          title="Knee Point"
          icon={Star}
          metricLabel="Balanced trade-off"
          metricValue={Number(knee?.f1_score ?? 0).toFixed(4)}
          pipeline={knee}
        />
        <RecommendCard
          title="Best Accuracy"
          icon={TrendingUp}
          metricLabel="F1 Score"
          metricValue={Number(bestAccuracy?.f1_score ?? 0).toFixed(4)}
          pipeline={bestAccuracy}
        />
        <RecommendCard
          title="Best Speed"
          icon={Zap}
          metricLabel="Latency"
          metricValue={`${(Number(bestSpeed?.latency ?? 0) * 1000).toFixed(4)} ms`}
          pipeline={bestSpeed}
        />
        <RecommendCard
          title="Best Interpretable"
          icon={Lightbulb}
          metricLabel="Interpretability Score"
          metricValue={Number(bestInterpretable?.interpretability ?? 0).toFixed(4)}
          pipeline={bestInterpretable}
        />
      </div>
    </section>
  );
};

export default DecisionSupport;
