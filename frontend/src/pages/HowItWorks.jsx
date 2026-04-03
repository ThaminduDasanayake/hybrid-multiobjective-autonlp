import { useNavigate } from "react-router-dom";
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  Cpu,
  Download,
  FlaskConical,
  Layers,
  RefreshCw,
  Rocket,
  Settings,
  SlidersHorizontal,
  Target,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button.jsx";

const StepHeader = ({ number, icon: Icon, title, color }) => (
  <div className="flex items-center gap-3 mb-3">
    <div className={`flex items-center justify-center w-8 h-8 rounded-full ${color} shrink-0`}>
      <Icon className="h-4 w-4" />
    </div>
    <div className="flex items-center gap-2.5">
      <span className="text-xs font-bold text-muted-foreground uppercase tracking-widest">
        Step {number}
      </span>
      <h3 className="text-base font-bold text-foreground">{title}</h3>
    </div>
  </div>
);

const GeneRow = ({ gene, options }) => (
  <div className="flex items-center gap-2">
    <span className="w-28 shrink-0 text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
      {gene}
    </span>
    <div className="flex flex-wrap gap-1">
      {options.map((opt) => (
        <span
          key={opt}
          className="rounded border border-violet-300/40 bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-mono text-violet-700 dark:text-violet-300"
        >
          {opt}
        </span>
      ))}
    </div>
  </div>
);

const ObjectiveChip = ({ icon: Icon, label, color }) => (
  <div className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${color}`}>
    <Icon className="h-3 w-3" />
    {label}
  </div>
);

export default function HowItWorks() {
  const navigate = useNavigate();

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      {/* Header */}
      <button
        onClick={() => navigate(-1)}
        className="mb-6 flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft size={15} />
        Back
      </button>

      <div className="flex items-center gap-2.5 mb-2">
        <FlaskConical size={24} className="text-primary" />
        <h1 className="text-2xl font-bold text-foreground">How T-AutoNLP Works</h1>
      </div>
      <p className="text-sm text-muted-foreground mb-10">
        An intelligent pipeline designed to find the perfect balance between predictive power and
        hardware efficiency — explained in three steps.
      </p>

      <div className="flex flex-col gap-6">
        {/* ── Step 1: Configure ── */}
        <div className="rounded-xl border border-border bg-card p-5">
          <StepHeader
            number="1"
            icon={Settings}
            title="Configure Search"
            color="bg-blue-500/10 text-blue-500"
          />
          <p className="text-sm text-muted-foreground leading-relaxed mb-4">
            You control three key knobs that determine how wide and deep the search runs. A larger
            population and more generations explore more of the space; more BO calls give each
            candidate a more accurate score.
          </p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <div className="flex items-start gap-2.5 rounded-lg border border-border bg-muted/40 p-3">
              <Users className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-foreground">Population Size</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  How many candidate pipelines evolve in parallel each generation.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5 rounded-lg border border-border bg-muted/40 p-3">
              <RefreshCw className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-foreground">Generations</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  How many rounds of evolution the Genetic Algorithm runs.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5 rounded-lg border border-primary/30 bg-primary/5 p-3">
              <SlidersHorizontal className="h-4 w-4 text-primary mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-foreground">BO Calls</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  Hyperparameter trials the inner Bayesian Optimiser gets per pipeline.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ── Step 2: Nested Algorithm Diagram ── */}
        <div className="rounded-xl border border-border bg-card p-5">
          <StepHeader
            number="2"
            icon={Cpu}
            title="Multi-Objective Optimization"
            color="bg-violet-500/10 text-violet-500"
          />
          <p className="text-sm text-muted-foreground leading-relaxed mb-5">
            Two algorithms work in a nested loop — an outer "Architect" and an inner "Tuner" — to
            simultaneously optimise for all three objectives.
          </p>

          {/* Outer GA card */}
          <div className="rounded-xl border-2 border-dashed border-violet-400/60 bg-violet-500/5 p-4">
            {/* GA header */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-violet-500/15">
                  <Layers className="h-4 w-4 text-violet-500" />
                </div>
                <div>
                  <p className="text-xs font-bold text-violet-600 dark:text-violet-400 uppercase tracking-wide">
                    Outer Loop
                  </p>
                  <p className="text-sm font-semibold text-foreground leading-none">
                    The Architect&nbsp;
                    <span className="text-xs font-normal text-muted-foreground">
                      (Genetic Algorithm — NSGA-II)
                    </span>
                  </p>
                </div>
              </div>
              <RefreshCw className="h-4 w-4 text-violet-400 animate-spin [animation-duration:4s]" />
            </div>

            <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
              Each individual in the population is a <strong className="text-foreground">chromosome of 6 genes</strong> — one
              choice per pipeline stage.
            </p>

            {/* Gene pool table */}
            <div className="rounded-lg border border-violet-300/30 bg-background/60 p-3 flex flex-col gap-2 mb-4">
              <GeneRow gene="Vectoriser" options={["tfidf", "count"]} />
              <GeneRow gene="Scaler" options={["none", "maxabs", "robust"]} />
              <GeneRow gene="Dim. Reduction" options={["none", "select_k_best"]} />
              <GeneRow gene="Classifier" options={["logistic", "naive_bayes", "svm"]} />
              <GeneRow gene="n-gram Range" options={["1-1", "1-2"]} />
              <GeneRow gene="Max Features" options={["5 000", "10 000", "none"]} />
            </div>

            {/* Arrow down into BO */}
            <div className="flex flex-col items-center my-2 gap-0.5">
              <p className="text-[10px] text-muted-foreground italic">
                "Here's a fixed structure — find its best score"
              </p>
              <ArrowDown className="h-4 w-4 text-muted-foreground" />
            </div>

            {/* Inner BO card */}
            <div className="rounded-lg border border-amber-400/50 bg-amber-500/5 p-3.5 mx-2">
              <div className="flex items-center gap-2 mb-2">
                <div className="flex items-center justify-center w-6 h-6 rounded-md bg-amber-500/15">
                  <Target className="h-3.5 w-3.5 text-amber-500" />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wide">
                    Inner Loop
                  </p>
                  <p className="text-xs font-semibold text-foreground leading-none">
                    The Tuner&nbsp;
                    <span className="font-normal text-muted-foreground">(Bayesian Optimisation)</span>
                  </p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed mb-2.5">
                Receives the fixed structure and uses a Gaussian Process surrogate to efficiently
                search the continuous hyperparameter space — finding the optimal settings in far
                fewer trials than grid search.
              </p>
              {/* BO param groups */}
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-amber-400/30 bg-amber-500/5 p-2">
                  <p className="text-[10px] font-semibold text-amber-700 dark:text-amber-300 mb-1.5">
                    Vectoriser params
                  </p>
                  <div className="flex flex-col gap-1">
                    <span className="font-mono text-[10px] text-muted-foreground">
                      min_df &nbsp;<span className="text-foreground">∈ [1, 10]</span>
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      max_df &nbsp;<span className="text-foreground">∈ [0.5, 1.0]</span>
                    </span>
                  </div>
                </div>
                <div className="rounded-md border border-amber-400/30 bg-amber-500/5 p-2">
                  <p className="text-[10px] font-semibold text-amber-700 dark:text-amber-300 mb-1.5">
                    Classifier params
                  </p>
                  <div className="flex flex-col gap-1">
                    <span className="font-mono text-[10px] text-muted-foreground">
                      C &nbsp;<span className="text-foreground">∈ [0.01, 10] log</span>
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      alpha / penalty&nbsp;<span className="text-foreground">(model-specific)</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Arrow up back to GA */}
            <div className="flex flex-col items-center mt-2 gap-0.5">
              <ArrowUp className="h-4 w-4 text-muted-foreground" />
              <p className="text-[10px] text-muted-foreground italic">
                "Here are the three objective scores for that pipeline"
              </p>
            </div>

            {/* Objectives footer */}
            <div className="mt-3 pt-3 border-t border-violet-300/30 flex flex-wrap gap-2">
              <ObjectiveChip
                icon={Target}
                label="F1 Score"
                color="bg-green-500/10 text-green-700 dark:text-green-400"
              />
              <ObjectiveChip
                icon={FlaskConical}
                label="Latency"
                color="bg-sky-500/10 text-sky-700 dark:text-sky-400"
              />
              <ObjectiveChip
                icon={Layers}
                label="Interpretability"
                color="bg-orange-500/10 text-orange-700 dark:text-orange-400"
              />
              <span className="text-[10px] text-muted-foreground self-center ml-auto">
                Pareto front updated every generation
              </span>
            </div>
          </div>
        </div>

        {/* ── Step 3: Export ── */}
        <div className="rounded-xl border border-border bg-card p-5">
          <StepHeader
            number="3"
            icon={Download}
            title="Zero-Lock-in Export"
            color="bg-emerald-500/10 text-emerald-500"
          />
          <p className="text-sm text-muted-foreground leading-relaxed mb-4">
            Once the search converges, explore the Pareto front and pick the trade-off that suits
            your deployment.
          </p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <div className="flex items-start gap-2.5 rounded-lg border border-border bg-muted/30 p-3">
              <Target className="h-4 w-4 text-primary mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-foreground">Explore Pareto Front</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  Interactive 2D/3D scatter &amp; parallel coordinates charts.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5 rounded-lg border border-border bg-muted/30 p-3">
              <Settings className="h-4 w-4 text-primary mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-foreground">Compare Solutions</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  Side-by-side pipeline configs with full metric breakdown.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
              <Download className="h-4 w-4 text-emerald-500 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-foreground">Export Pure Python</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  One-click Scikit-Learn code. No vendor lock-in.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="mt-8 flex justify-center">
        <Button onClick={() => navigate("/run")} size="lg">
          <Rocket size={16} />
          Start Optimization
        </Button>
      </div>
    </div>
  );
}
