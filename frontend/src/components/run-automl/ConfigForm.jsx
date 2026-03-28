import { useState, useMemo } from "react";
import { AlertCircle, Loader2, Zap, Info, Calculator } from "lucide-react";
import { useStartJob } from "@/hooks/useApi.js";
import { Alert, AlertDescription } from "@/components/ui/alert.jsx";
import { DATASETS, DEFAULTS, DEMO_CONFIG } from "@/constants.js";
import { Button } from "@/components/ui/button.jsx";
import { Card, CardContent } from "@/components/ui/card.jsx";
import { Label } from "@/components/ui/label.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select.jsx";
import SliderField from "./SliderField.jsx";

const ConfigForm = ({ onJobStarted }) => {
  const [config, setConfig] = useState(DEFAULTS);

  const { totalPipelines, runIntensity, badgeVariant } = useMemo(() => {
    const popSize = config.population_size || 20;
    const nGen = config.n_generations || 10;
    const bo = config.bo_calls || 15;
    
    const total = bo + (popSize * (nGen + 1));
    
    let intensity = "Fast Run";
    let variant = "bg-emerald-500/15 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400 border-emerald-500/20";
    
    if (total >= 300 && total <= 1000) {
      intensity = "Standard Run";
      variant = "bg-amber-500/15 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400 border-amber-500/20";
    } else if (total > 1000) {
      intensity = "Heavy Compute";
      variant = "bg-destructive/15 text-destructive dark:bg-destructive/10 border-destructive/20";
    }
    
    return { totalPipelines: total, runIntensity: intensity, badgeVariant: variant };
  }, [config.population_size, config.n_generations, config.bo_calls]);

  const startJobMutation = useStartJob();

  const set = (key) => (val) => setConfig((prev) => ({ ...prev, [key]: val }));

  const submit = (config) => {
    if (startJobMutation.isPending) return;
    startJobMutation.mutate(config, {
      onSuccess: ({ job_id }) => onJobStarted(job_id),
    });
  };

  return (
    <div className="mx-auto max-w-2xl">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(config);
        }}
        className="space-y-6"
      >
        <div className="space-y-2">
          <Label>Dataset</Label>
          <Select value={config.dataset_name} onValueChange={set("dataset_name")}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent position="popper">
              {DATASETS.map((d) => (
                <SelectItem key={d.value} value={d.value}>
                  {d.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-3 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3">
          <p className="text-sm font-semibold text-foreground">
            3-Objective Mode (F1, Latency, Interpretability)
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            The primary thesis experiment optimises all three objectives simultaneously, producing a
            Pareto front of non-dominated pipelines. This is the default.
          </p>
        </div>

        <Card>
          <CardContent className="p-5 space-y-5">
            <SliderField
              label="Training Samples"
              hint="How many documents are used to train and evaluate each pipeline."
              value={config.max_samples}
              min={100}
              max={10000}
              step={100}
              onChange={set("max_samples")}
            />
            <SliderField
              label="Population Size"
              hint="Number of pipeline candidates per GA generation."
              value={config.population_size}
              min={5}
              max={100}
              step={5}
              onChange={set("population_size")}
            />
            <SliderField
              label="Generations"
              hint="How many evolutionary cycles to run."
              value={config.n_generations}
              min={1}
              max={50}
              step={1}
              onChange={set("n_generations")}
            />
            <SliderField
              label="BO Calls"
              hint="Bayesian optimization evaluations per top candidate."
              value={config.bo_calls}
              min={10}
              max={50}
              step={5}
              onChange={set("bo_calls")}
            />
          </CardContent>
        </Card>

        {startJobMutation.error && (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertDescription>{startJobMutation.error.message}</AlertDescription>
          </Alert>
        )}

        <Alert className="mb-6 border-secondary/20 bg-secondary/10 text-secondary">
          <Info className="h-4 w-4 text-secondary" />
          <AlertDescription className="ml-2 text-sm leading-relaxed">
            <strong>Note:</strong> T-AutoNLP performs rigorous evolutionary search. Depending on
            your configuration, a run may take 10-20 minutes. Progress will be streamed in
            real-time. For a faster run, use the Quick Demo option.
          </AlertDescription>
        </Alert>

        <div className="mb-4 flex items-center justify-between rounded-lg border border-border bg-card px-4 py-3 shadow-sm">
          <div className="flex items-center gap-2.5">
            <Calculator className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-foreground">
              Maximum Pipelines Evaluated
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-sm font-bold tabular-nums text-foreground">
              {totalPipelines.toLocaleString()}
            </span>
            <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors ${badgeVariant}`}>
              {runIntensity}
            </span>
          </div>
        </div>

        <div className="flex gap-3 pt-1">
          <Button type="submit" disabled={startJobMutation.isPending} size="lg" className="grow">
            {startJobMutation.isPending ? <Loader2 className="animate-spin" /> : <Zap />}
            {startJobMutation.isPending ? "Starting…" : "Run AutoML"}
          </Button>

          <Button
            type="button"
            variant="secondary"
            size="lg"
            disabled={startJobMutation.isPending}
            onClick={() => submit(DEMO_CONFIG)}
          >
            Quick Demo
          </Button>
        </div>
      </form>
    </div>
  );
};

export default ConfigForm;
