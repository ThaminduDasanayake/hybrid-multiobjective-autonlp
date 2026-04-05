import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Plot from "react-plotly.js";
import {
  ArrowLeft,
  BrainCircuit,
  Zap,
  BookOpen,
  Scale,
  Users,
  Settings,
  Target,
  Sparkles,
  Rocket,
  TrendingUp,
  Cpu,
  ChevronDown,
  ChevronUp,
  GitMerge,
  Dna,
  FlaskConical,
  BarChart2,
  Download,
} from "lucide-react";
import { Button } from "@/components/ui/button.jsx";
import { Badge } from "@/components/ui/badge.jsx";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card.jsx";
import {
  COLORS,
  AXIS_STYLE,
  CHART_LAYOUT,
  CHART_CONFIG_MINIMAL,
  LEGEND_STYLE,
} from "@/utils/chartTheme.js";

// ─── Helper: collapsible technical deep-dive wrapper ────────────────────────
function TechnicalBlock({ title, icon: Icon, children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-6 rounded-lg border border-border bg-muted/30">
      <button
        onClick={() => setOpen((p) => !p)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-foreground hover:bg-muted/60 transition-colors rounded-lg"
      >
        <span className="flex items-center gap-2">
          <Icon size={15} className="text-muted-foreground" />
          {title}
        </span>
        {open ? (
          <ChevronUp size={15} className="text-muted-foreground" />
        ) : (
          <ChevronDown size={15} className="text-muted-foreground" />
        )}
      </button>
      {open && (
        <div className="px-4 pb-4 pt-1 border-t border-border text-sm text-muted-foreground leading-relaxed space-y-4">
          {children}
        </div>
      )}
    </div>
  );
}

// ─── Plotly: Pareto Front Diagram ────────────────────────────────────────────
// Illustrative data – all solutions scatter + Pareto front line + knee point
function ParetoExplainerChart() {
  // Dominated (grey) solutions
  const dominated = {
    x: [120, 200, 350, 90, 450, 280, 170, 400, 60, 310],
    y: [0.71, 0.72, 0.79, 0.65, 0.82, 0.74, 0.68, 0.77, 0.62, 0.76],
  };

  // Pareto-optimal solutions (sorted by latency asc)
  const pareto = {
    x: [55, 80, 115, 160, 230, 390],
    y: [0.61, 0.68, 0.75, 0.81, 0.87, 0.91],
  };

  // Knee point index (in pareto array)
  const kneeIdx = 3;

  const data = [
    {
      type: "scatter",
      mode: "markers",
      name: "All Solutions",
      x: dominated.x,
      y: dominated.y,
      marker: { size: 8, color: COLORS.dominatedSolution },
      hovertemplate: "Latency: %{x} ms<br>F1: %{y:.2f}<extra>Dominated</extra>",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Pareto Front",
      x: pareto.x,
      y: pareto.y,
      marker: {
        size: 10,
        color: COLORS.primary,
        symbol: "diamond",
        line: { color: COLORS.primaryDark, width: 1 },
      },
      line: { color: COLORS.primary, width: 1.5, dash: "dot" },
      hovertemplate: "Latency: %{x} ms<br>F1: %{y:.2f}<extra>Pareto Optimal</extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Knee Point ★",
      x: [pareto.x[kneeIdx]],
      y: [pareto.y[kneeIdx]],
      marker: {
        size: 18,
        color: COLORS.amber,
        symbol: "star",
        line: { color: "#fff", width: 1.5 },
      },
      hovertemplate:
        "<b>★ Recommended Sweet Spot</b><br>Latency: %{x} ms<br>F1: %{y:.2f}<extra></extra>",
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 60, r: 20, t: 30, b: 60 },
    showlegend: true,
    legend: { x: 0.01, y: 0.99, ...LEGEND_STYLE },
    xaxis: {
      ...AXIS_STYLE,
      title: {
        text: "Prediction Latency (ms) ↓  Lower is faster",
        font: { size: 11, color: COLORS.slate500 },
      },
    },
    yaxis: {
      ...AXIS_STYLE,
      title: {
        text: "F1 Score ↑  Higher is more accurate",
        font: { size: 11, color: COLORS.slate500 },
      },
      range: [0.55, 0.97],
    },
    annotations: [
      {
        x: pareto.x[kneeIdx],
        y: pareto.y[kneeIdx],
        xref: "x",
        yref: "y",
        text: "Best Balance",
        showarrow: true,
        arrowhead: 2,
        arrowcolor: COLORS.amber,
        ax: 50,
        ay: -40,
        font: { color: COLORS.amber, size: 11, family: "Inter" },
      },
    ],
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ ...CHART_CONFIG_MINIMAL, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%", height: "320px" }}
    />
  );
}

// ─── Plotly: GA Evolution Scatter (animated by generation) ───────────────────
function GAEvolutionChart() {
  // Simulated population across 5 generations
  // Each generation the population improves (moves toward better F1 + lower latency)
  const genCount = 5;

  // Seeds for reproducible mock data
  const baseLatency = [480, 420, 390, 350, 300, 260, 200, 150, 120, 90];
  const baseF1 = [0.6, 0.63, 0.65, 0.68, 0.71, 0.74, 0.77, 0.8, 0.84, 0.88];

  const frames = Array.from({ length: genCount }, (_, gi) => {
    const noise = (seed) => Math.sin(seed * 97.3 + gi * 31.7) * 0.5 + 0.5 - 0.5;
    const xs = baseLatency.map((v, i) => Math.max(40, v - gi * 25 + noise(i * 3) * 60));
    const ys = baseF1.map((v, i) => Math.min(0.98, v + gi * 0.02 + noise(i * 5) * 0.04));
    return {
      name: `Gen ${gi + 1}`,
      data: [
        {
          type: "scatter",
          mode: "markers",
          name: `Generation ${gi + 1}`,
          x: xs,
          y: ys,
          marker: {
            size: 10,
            color: xs.map((_, i) => {
              // Colour gradient: orange for better solutions (higher F1, lower latency)
              const score = ys[i] - xs[i] / 600;
              return score > 0.6 ? COLORS.primary : COLORS.dominatedSolutionMed;
            }),
            line: { color: COLORS.primaryDark, width: 0.5 },
          },
          hovertemplate: `<b>Generation ${gi + 1}</b><br>Latency: %{x:.0f} ms<br>F1: %{y:.3f}<extra></extra>`,
        },
      ],
    };
  });

  const sliderSteps = frames.map((f, i) => ({
    args: [[f.name], { frame: { duration: 400, redraw: true }, mode: "immediate" }],
    label: `Gen ${i + 1}`,
    method: "animate",
  }));

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 60, r: 20, t: 20, b: 80 },
    showlegend: false,
    xaxis: {
      ...AXIS_STYLE,
      title: { text: "Prediction Latency (ms) ↓", font: { size: 11, color: COLORS.slate500 } },
      range: [0, 550],
    },
    yaxis: {
      ...AXIS_STYLE,
      title: { text: "F1 Score ↑", font: { size: 11, color: COLORS.slate500 } },
      range: [0.55, 1.0],
    },
    updatemenus: [
      {
        type: "buttons",
        showactive: false,
        y: -0.32,
        x: 0.5,
        xanchor: "center",
        yanchor: "top",
        buttons: [
          {
            label: "▶  Play Evolution",
            method: "animate",
            args: [
              null,
              {
                frame: { duration: 700, redraw: true },
                fromcurrent: true,
                transition: { duration: 300 },
              },
            ],
          },
          {
            label: "⏸  Pause",
            method: "animate",
            args: [[null], { frame: { duration: 0, redraw: false }, mode: "immediate" }],
          },
        ],
        font: { size: 11, color: COLORS.slate500 },
        bgcolor: COLORS.legendBg,
        bordercolor: COLORS.slate800,
        borderwidth: 1,
      },
    ],
    sliders: [
      {
        active: 0,
        steps: sliderSteps,
        y: -0.18,
        x: 0,
        len: 1,
        currentvalue: {
          prefix: "Showing: ",
          font: { size: 11, color: COLORS.slate500 },
          xanchor: "center",
        },
        font: { size: 10, color: COLORS.slate500 },
        bgcolor: COLORS.slate800,
        bordercolor: COLORS.slate800,
        tickcolor: COLORS.slate600,
      },
    ],
  };

  return (
    <Plot
      data={frames[0].data}
      layout={layout}
      frames={frames}
      config={{ ...CHART_CONFIG_MINIMAL, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%", height: "380px" }}
    />
  );
}

const HowItWorks = () => {
  const navigate = useNavigate();

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 py-10 space-y-10">
      {/* Back */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors group"
      >
        <ArrowLeft size={15} className="group-hover:-translate-x-0.5 transition-transform" />
        Back
      </button>

      {/* Hero */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 text-primary mb-2">
          <BrainCircuit size={32} />
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight">
          How T-AutoNLP Works
        </h1>
        <p className="text-muted-foreground max-w-xl mx-auto leading-relaxed">
          A plain-language guide to what happens from the moment you press
          <strong className="text-foreground"> Start</strong> to when you download your finished
          model. Technical deep-dives are available for the curious.
        </p>
        <div className="flex flex-wrap justify-center gap-2 pt-1">
          <Badge variant="outline">No ML experience needed</Badge>
          <Badge variant="outline">Interactive charts</Badge>
          <Badge variant="outline">Technical details inside ↓</Badge>
        </div>
      </div>

      {/* ── SECTION 1 ─────────────────────────────────────────── */}
      <Card>
        <CardHeader className="border-b">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 text-primary">
              <Scale size={18} />
            </div>
            <div>
              <CardTitle>1 · The Balancing Act</CardTitle>
              <CardDescription>Why there's no single "best" model</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-4 space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Imagine you're buying a car. You want it to be{" "}
            <strong className="text-foreground">fast</strong>,{" "}
            <strong className="text-foreground">reliable</strong>, and{" "}
            <strong className="text-foreground">affordable</strong>. In practice you can usually
            only optimise two of those at the cost of the third. Text-classification models have
            exactly the same dilemma, here mapped to three measurable objectives:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                icon: Target,
                label: "Accuracy",
                sublabel: "F1 Score",
                desc: "How often the model correctly classifies a piece of text. Higher is better.",
                color: "text-primary bg-primary/10",
              },
              {
                icon: Zap,
                label: "Speed",
                sublabel: "Latency",
                desc: "How quickly the model returns a prediction (milliseconds). Lower is better.",
                color: "text-secondary bg-secondary/10",
              },
              {
                icon: BookOpen,
                label: "Simplicity",
                sublabel: "Interpretability",
                desc: "How easy it is for a human to trace why the model made a decision.",
                color: "text-foreground bg-accent",
              },
            ].map(({ icon: Icon, label, sublabel, desc, color }) => (
              <div
                key={label}
                className="flex flex-col items-center gap-2 p-4 rounded-lg border border-border bg-muted/20 text-center"
              >
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${color}`}>
                  <Icon size={20} />
                </div>
                <div>
                  <p className="font-semibold text-foreground text-sm">{label}</p>
                  <p className="text-[11px] text-muted-foreground">{sublabel}</p>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>

          <p className="text-sm text-muted-foreground leading-relaxed">
            T-AutoNLP <em>doesn't pick one winner</em>. Instead it searches thousands of different
            model configurations simultaneously and learns which combinations sit on the outer edge
            of what's possible — the <strong className="text-foreground">Pareto Front</strong>.
          </p>
        </CardContent>
      </Card>

      {/* ── SECTION 2 ─────────────────────────────────────────── */}
      <Card>
        <CardHeader className="border-b">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 text-primary">
              <Cpu size={18} />
            </div>
            <div>
              <CardTitle>2 · The Hybrid Engine</CardTitle>
              <CardDescription>Two AI algorithms working as a tag-team</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-4 space-y-5">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Searching all possible model combinations by brute force would take{" "}
            <strong className="text-foreground">years</strong>. T-AutoNLP uses two interlocking
            algorithms to explore smartly and quickly.
          </p>

          {/* Two-column algo cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* GA */}
            <div className="rounded-lg border border-border p-4 space-y-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-8 h-8 rounded-md bg-primary/10 text-primary">
                  <Dna size={16} />
                </div>
                <div>
                  <p className="font-semibold text-sm text-foreground">The Architect</p>
                  <p className="text-[11px] text-muted-foreground">Genetic Algorithm (NSGA-II)</p>
                </div>
                <Badge variant="outline" className="ml-auto text-[10px]">
                  Outer loop
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Inspired by natural evolution. Starts with a diverse "population" of pipeline
                blueprints, scores each one, then recombines the best performers — like breeding the
                fastest horses — to produce an ever-improving next generation.
              </p>
              <div className="rounded-md bg-muted/40 p-3">
                <p className="text-[11px] font-semibold text-foreground mb-2">
                  Each blueprint encodes 6 choices:
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  {[
                    ["Vectoriser", "TF-IDF / Count"],
                    ["Scaler", "None / Max-Abs / Robust"],
                    ["Dimensionality", "None / Select-K-Best"],
                    ["Classifier", "Logistic / NB / SVM"],
                    ["N-gram range", "1-1 / 1-2"],
                    ["Max features", "5 K / 10 K / None"],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <p className="text-[10px] font-medium text-foreground">{k}</p>
                      <p className="text-[10px] text-muted-foreground font-mono">{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* BO */}
            <div className="rounded-lg border border-border p-4 space-y-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-8 h-8 rounded-md bg-secondary/10 text-secondary">
                  <FlaskConical size={16} />
                </div>
                <div>
                  <p className="font-semibold text-sm text-foreground">The Tuner</p>
                  <p className="text-[11px] text-muted-foreground">Bayesian Optimisation</p>
                </div>
                <Badge variant="outline" className="ml-auto text-[10px]">
                  Inner loop
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Once the Architect fixes a blueprint, the Tuner step in to adjust the fine
                continuous knobs — things like regularisation strength or vocabulary frequency
                thresholds. It builds an internal <em>model-of-the-model</em> (a Gaussian Process)
                that predicts which settings will score best, so it wastes far fewer trials than a
                grid search.
              </p>
              <div className="rounded-md bg-muted/40 p-3 space-y-1.5">
                <p className="text-[11px] font-semibold text-foreground">
                  Example hyperparameters tuned:
                </p>
                {[
                  ["min_df", "∈ [1, 10]", "Ignore very rare words"],
                  ["max_df", "∈ [0.5, 1.0]", "Ignore very common words"],
                  ["C / alpha", "log-scale", "Model regularisation strength"],
                ].map(([name, range, note]) => (
                  <div key={name} className="flex items-baseline justify-between gap-2">
                    <code className="text-[10px] text-primary font-mono">
                      {name} {range}
                    </code>
                    <span className="text-[10px] text-muted-foreground">{note}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Flow diagram */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-2 py-2 text-[11px] text-muted-foreground font-medium">
            {[
              "GA picks a blueprint",
              "→",
              "BO tunes its knobs",
              "→",
              "3 scores returned to GA",
              "→",
              "GA evolves population",
              "→",
              "Repeat ×Generations",
            ].map((s, i) =>
              s === "→" ? (
                <span key={i} className="text-border font-light hidden sm:inline">
                  →
                </span>
              ) : (
                <span
                  key={i}
                  className="px-2 py-1 rounded-md bg-muted/60 border border-border text-foreground"
                >
                  {s}
                </span>
              ),
            )}
          </div>

          {/* Technical block: GA evolution animator */}
          <TechnicalBlock
            title="Technical deep-dive · GA Population Evolution (interactive)"
            icon={BarChart2}
          >
            <p>
              The chart below animates how the population shifts across generations. Press
              <strong className="text-foreground"> ▶ Play Evolution</strong> or drag the slider to
              step through manually. You should see points migrate toward the{" "}
              <strong className="text-foreground">top-left</strong> corner (high accuracy, low
              latency) as the genetic algorithm selects and recombines stronger individuals.
            </p>
            <div className="rounded-lg border border-border bg-card p-2 overflow-hidden">
              <GAEvolutionChart />
            </div>
            <p className="text-xs">
              <strong className="text-foreground">NSGA-II</strong> (Non-dominated Sorting Genetic
              Algorithm II) is the specific variant used. It ranks individuals by Pareto dominance:
              a solution <em>dominates</em> another if it is no worse in all objectives and strictly
              better in at least one. Solutions on the same dominance rank are spread apart using{" "}
              <em>crowding distance</em> — a diversity preservation trick that prevents the entire
              population collapsing onto a single point of the Pareto Front.
            </p>
          </TechnicalBlock>

          <TechnicalBlock
            title="Technical deep-dive · Bayesian Optimisation intuition"
            icon={GitMerge}
          >
            <p>
              A plain grid search over continuous hyperparameters (e.g. trying C = 0.01, 0.02, …,
              10.00) would require hundreds of slow train-evaluate cycles <em>per blueprint</em>.
              Bayesian Optimisation drastically reduces this by fitting a{" "}
              <strong className="text-foreground">Gaussian Process (GP)</strong> — a cheap-to-query
              probabilistic surrogate — to all previous observations.
            </p>
            <ol className="list-decimal list-inside space-y-1 text-xs pl-2">
              <li>Evaluate a handful of random configurations to seed the GP.</li>
              <li>
                Use an <em>acquisition function</em> (Expected Improvement) to pick the{" "}
                <em>most promising untried point</em> balancing exploration and exploitation.
              </li>
              <li>Train the real model at that point, observe the score, update the GP.</li>
              <li>Repeat for the configured number of BO calls, then return the best result.</li>
            </ol>
            <p className="text-xs">
              In practice, this means the inner tuner finds near-optimal hyperparameters in only{" "}
              <strong className="text-foreground">10–30 evaluations</strong> rather than the
              thousands a grid search would require.
            </p>
          </TechnicalBlock>
        </CardContent>
      </Card>

      {/* ── SECTION 3 ─────────────────────────────────────────── */}
      <Card>
        <CardHeader className="border-b">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 text-primary">
              <TrendingUp size={18} />
            </div>
            <div>
              <CardTitle>3 · Reading the Pareto Front</CardTitle>
              <CardDescription>Picking your personal sweet spot from the results</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-4 space-y-5">
          <p className="text-sm text-muted-foreground leading-relaxed">
            When the optimization finishes, T-AutoNLP does <em>not</em> hand you one single answer.
            It presents the best possible trade-offs as an interactive scatter chart. The example
            below uses illustrative data — in a real run the points represent actual trained models
            from your dataset.
          </p>

          <div className="rounded-lg border border-border bg-card overflow-hidden">
            <ParetoExplainerChart />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg border border-border bg-muted/20 space-y-1.5">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ background: COLORS.primary }} />
                <p className="text-sm font-semibold text-foreground">Pareto-Optimal Solutions</p>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Orange diamonds sit on the outer "frontier". You{" "}
                <strong className="text-foreground">cannot</strong> improve accuracy without
                accepting higher latency, and vice versa. Any grey dot is strictly outclassed by at
                least one orange diamond.
              </p>
            </div>
            <div className="p-4 rounded-lg border border-border bg-primary/5 space-y-1.5">
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-primary" />
                <p className="text-sm font-semibold text-foreground">The Knee Point ★</p>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Mathematically the point of <em>maximum curvature</em> on the Pareto Front. Moving
                away from it along the curve gives diminishing returns — a small gain in one
                objective costs a large sacrifice in another. Our system highlights it automatically
                as the default recommended model.
              </p>
            </div>
          </div>

          <TechnicalBlock
            title="Technical deep-dive · How the Knee Point is calculated"
            icon={Target}
          >
            <p>
              The system applies the{" "}
              <strong className="text-foreground">Minimum Manhattan Distance (MMD)</strong> method
              with objective normalisation:
            </p>
            <ol className="list-decimal list-inside space-y-1 text-xs pl-2">
              <li>Normalize all three objective values to [0, 1] across the Pareto Front.</li>
              <li>
                Define the <em>ideal point</em> — (1, 0, 1) meaning maximum F1, minimum latency,
                maximum interpretability.
              </li>
              <li>Compute the Euclidean distance from each Pareto solution to the ideal point.</li>
              <li>
                The solution <strong className="text-foreground">closest to the ideal</strong> is
                the knee point.
              </li>
            </ol>
            <p className="text-xs">
              This is a deliberate compromise: techniques such as NSGA-II crowding distance or
              reference-vector approaches also exist, but MMD is interpretable and requires no
              user-supplied weight preferences.
            </p>
          </TechnicalBlock>
        </CardContent>
      </Card>

      {/* ── SECTION 4: Export ─────────────────────────────────── */}
      <Card>
        <CardHeader className="border-b">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 text-primary">
              <Download size={18} />
            </div>
            <div>
              <CardTitle>4 · Exporting Your Model</CardTitle>
              <CardDescription>Own what you build — no lock-in</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-4 space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Once you've picked the trade-off that suits your deployment, T-AutoNLP generates a
            self-contained <strong className="text-foreground">Scikit-Learn Python pipeline</strong>{" "}
            you can run anywhere — your laptop, a cloud server, or an edge device. No proprietary
            runtime required.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                icon: BarChart2,
                title: "Explore the Front",
                desc: "Interactive 2D & 3D scatter charts. Filter by model type, compare side-by-side.",
              },
              {
                icon: Settings,
                title: "Compare Pipelines",
                desc: "Full metric breakdown per solution: F1, precision, recall, latency, and interpretability score.",
              },
              {
                icon: Download,
                title: "Export Pure Python",
                desc: "One click generates a runnable Scikit-Learn Pipeline object. Copy, paste, deploy.",
              },
            ].map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                className="p-4 rounded-lg border border-border bg-muted/20 space-y-2"
              >
                <div className="flex items-center gap-2">
                  <Icon size={15} className="text-primary" />
                  <p className="text-sm font-semibold text-foreground">{title}</p>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* CTA */}
      <div className="flex flex-col items-center gap-3 py-6 text-center">
        <Users size={28} className="text-primary" />
        <h2 className="text-xl font-bold text-foreground">Ready to run your first experiment?</h2>
        <p className="text-sm text-muted-foreground max-w-sm">
          Upload your labelled text dataset and let the engine find your best model automatically.
        </p>
        <Button onClick={() => navigate("/run")} size="lg" className="mt-2">
          <Rocket size={16} />
          Start Optimization
        </Button>
      </div>
    </div>
  );
};

export default HowItWorks;
