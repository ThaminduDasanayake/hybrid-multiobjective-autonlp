import Plot from "react-plotly.js";
import { COLORS, CHART_LAYOUT, CHART_CONFIG_MINIMAL } from "@/utils/chartTheme.js";

const GROUPS = [
  {
    key: "vectorizer",
    label: "Vectorizer",
    values: ["tfidf", "count"],
    displayNames: { tfidf: "TF-IDF", count: "Count" },
  },
  {
    key: "scaler",
    label: "Scaler",
    values: ["standard", "maxabs", "robust", null],
    displayNames: { standard: "Standard", maxabs: "MaxAbs", robust: "Robust", null: "None" },
  },
  {
    key: "dim_reduction",
    label: "Dim Reduction",
    values: ["pca", "select_k_best", null],
    displayNames: { pca: "PCA", select_k_best: "SelectKBest", null: "None" },
  },
];

const countBy = (solutions, key, value) =>
  solutions.filter((s) => (s[key] ?? null) === value).length;

const PipelineBreakdownChart = ({ allSolutions = [], paretoFront = [] }) => {
  if (allSolutions.length === 0) {
    return <div className="chart-empty h-40">No solutions available.</div>;
  }

  const allTraces = [];
  const annotations = [];
  const shapes = [];

  const domainStep = 1 / GROUPS.length;
  const gap = 0.06;

  GROUPS.forEach((group, gi) => {
    const xaxis = gi === 0 ? "x" : `x${gi + 1}`;
    const domainStart = gi * domainStep + gap / 2;
    const domainEnd = (gi + 1) * domainStep - gap / 2;

    const xLabels = group.values.map((v) => group.displayNames[v] ?? String(v));
    const allCounts = group.values.map((v) => countBy(allSolutions, group.key, v));
    const paretoCounts = group.values.map((v) => countBy(paretoFront, group.key, v));

    allTraces.push({
      type: "bar",
      name: gi === 0 ? "All Solutions" : undefined,
      legendgroup: "all",
      showlegend: gi === 0,
      x: xLabels,
      y: allCounts,
      xaxis,
      marker: { color: COLORS.slate600 },
      hovertemplate: "%{x}: %{y}<extra>All Solutions</extra>",
    });

    allTraces.push({
      type: "bar",
      name: gi === 0 ? "Pareto Front" : undefined,
      legendgroup: "pareto",
      showlegend: gi === 0,
      x: xLabels,
      y: paretoCounts,
      xaxis,
      marker: { color: COLORS.primary },
      hovertemplate: "%{x}: %{y}<extra>Pareto Front</extra>",
    });

    annotations.push({
      x: (domainStart + domainEnd) / 2,
      y: 1.08,
      xref: "paper",
      yref: "paper",
      text: `<b>${group.label}</b>`,
      showarrow: false,
      font: { size: 12, color: COLORS.slate400 },
    });

    if (gi < GROUPS.length - 1) {
      shapes.push({
        type: "line",
        xref: "paper",
        yref: "paper",
        x0: (gi + 1) * domainStep,
        x1: (gi + 1) * domainStep,
        y0: 0,
        y1: 1,
        line: { color: COLORS.slate800, width: 1.5, dash: "dot" },
      });
    }
  });

  const xaxisConfigs = {};
  GROUPS.forEach((_, gi) => {
    const key = gi === 0 ? "xaxis" : `xaxis${gi + 1}`;
    const domainStart = gi * domainStep + gap / 2;
    const domainEnd = (gi + 1) * domainStep - gap / 2;
    xaxisConfigs[key] = {
      domain: [domainStart, domainEnd],
      tickfont: { color: COLORS.slate400, size: 11 },
      gridcolor: COLORS.slate800,
      zeroline: false,
      anchor: "y",
    };
  });

  const layout = {
    ...CHART_LAYOUT,
    barmode: "group",
    margin: { l: 40, r: 20, t: 40, b: 40 },
    yaxis: {
      tickfont: { color: COLORS.slate400, size: 11 },
      gridcolor: COLORS.slate800,
      zeroline: false,
      title: { text: "Count", font: { size: 11, color: COLORS.slate500 } },
    },
    legend: {
      orientation: "h",
      x: 0,
      y: -0.15,
      font: { size: 11, color: COLORS.slate400 },
    },
    annotations,
    shapes,
    ...xaxisConfigs,
  };

  return (
    <Plot
      data={allTraces}
      layout={layout}
      config={CHART_CONFIG_MINIMAL}
      useResizeHandler
      style={{ width: "100%", height: "300px" }}
    />
  );
};

export default PipelineBreakdownChart;
