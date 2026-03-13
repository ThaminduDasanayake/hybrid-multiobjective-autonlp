import { useState } from "react";
import { AlertCircle, Loader2, Zap } from "lucide-react";
import { startJob } from "@/api.js";
import { Alert, AlertDescription } from "@/components/ui/alert.jsx";
import { DATASETS, DEFAULTS, DEMO_CONFIG } from "@/constants.js";
import { Button } from "@/components/ui/button.jsx";
import { Label } from "@/components/ui/label.jsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import SliderField from "./SliderField";

import React from "react";

const ConfigForm = ({ onJobStarted }) => {
  const [config, setConfig] = useState(DEFAULTS);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // For Select components (receives the value string directly).
  const set = (key) => (val) => setConfig((prev) => ({ ...prev, [key]: val }));

  // For SliderField (receives the new number value directly).
  const setNum = (key) => (val) => setConfig((prev) => ({ ...prev, [key]: val }));

  const submit = async (cfg) => {
    setError(null);
    setSubmitting(true);
    try {
      const { job_id } = await startJob(cfg);
      onJobStarted(job_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
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

        <div className="space-y-5 rounded-xl border border-border bg-card p-5">
          <SliderField
            label="Training Samples"
            hint="How many documents are used to train and evaluate each pipeline."
            value={config.max_samples}
            min={100}
            max={5000}
            step={100}
            onChange={setNum("max_samples")}
          />
          <SliderField
            label="Population Size"
            hint="Number of pipeline candidates per GA generation."
            value={config.population_size}
            min={5}
            max={100}
            step={5}
            onChange={setNum("population_size")}
          />
          <SliderField
            label="Generations"
            hint="How many evolutionary cycles to run."
            value={config.n_generations}
            min={1}
            max={50}
            step={1}
            onChange={setNum("n_generations")}
          />
          <SliderField
            label="BO Calls"
            hint="Bayesian optimisation evaluations per top candidate. Set 0 to disable BO."
            value={config.bo_calls}
            min={0}
            max={50}
            step={5}
            onChange={setNum("bo_calls")}
          />
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="flex gap-3 pt-1">
          <Button type="submit" disabled={submitting} size="lg" className="grow">
            {submitting ? <Loader2 className="animate-spin" /> : <Zap />}
            {submitting ? "Starting…" : "Run AutoML"}
          </Button>

          <Button
            type="button"
            variant="seondary"
            size="lg"
            disabled={submitting}
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
