import Plot from "react-plotly.js";
import { fmt } from "@/utils/formatters.js";
import {
  COLORS,
  AXIS_STYLE,
  CHART_LAYOUT,
  LEGEND_STYLE,
  CHART_CONFIG,
} from "@/utils/chartTheme.js";

function buildCustomData(solutions) {
  return solutions.map((s) => [
    fmt.model(s.model),
    fmt.vectorizer(s.vectorizer),
    fmt.scaler(s.scaler),
    s.ngram_range ?? "—",
  ]);
}

const HOVER_TEMPLATE =
  "<b>%{customdata[0]}</b><br>" +
  "Vectorizer: %{customdata[1]}<br>" +
  "Scaler: %{customdata[2]}<br>" +
  "N-gram: %{customdata[3]}<br>" +
  "─────────────────<br>" +
  "F1: %{x:.4f}<br>" +
  "Latency: %{y:.4f} ms<br>" +
  "Interpretability: %{z:.3f}" +
  "<extra></extra>";

const PARETO_HOVER_TEMPLATE =
  "<b>⭐ %{customdata[0]}</b><br>" +
  "Vectorizer: %{customdata[1]}<br>" +
  "Scaler: %{customdata[2]}<br>" +
  "N-gram: %{customdata[3]}<br>" +
  "─────────────────<br>" +
  "F1: %{x:.4f}<br>" +
  "Latency: %{y:.4f} ms<br>" +
  "Interpretability: %{z:.3f}" +
  "<extra></extra>";

const SCENE_AXIS = {
  gridcolor: COLORS.slate800,
  backgroundcolor: COLORS.transparent,
  showspikes: false,
  tickfont: { size: 10, color: COLORS.slate500 },
  titlefont: { size: 12, color: COLORS.slate400 },
  zerolinecolor: COLORS.slate800,
};

const ParetoFront3D = ({ allSolutions = [], paretoFront = [] }) => {
  if (allSolutions.length === 0 && paretoFront.length === 0) {
    return <div className="chart-empty h-96">No solution data available for this run.</div>;
  }

  const data = [
    {
      type: "scatter3d",
      mode: "markers",
      name: `All Solutions (${allSolutions.length})`,
      x: allSolutions.map((s) => s.f1_score),
      y: allSolutions.map((s) => (s.latency ?? 0) * 1000),
      z: allSolutions.map((s) => s.interpretability),
      customdata: buildCustomData(allSolutions),
      hovertemplate: HOVER_TEMPLATE,
      marker: { size: 4, color: COLORS.dominatedSolution, line: { width: 0 } },
    },
    {
      type: "scatter3d",
      mode: "markers",
      name: `Pareto Front (${paretoFront.length})`,
      x: paretoFront.map((s) => s.f1_score),
      y: paretoFront.map((s) => (s.latency ?? 0) * 1000),
      z: paretoFront.map((s) => s.interpretability),
      customdata: buildCustomData(paretoFront),
      hovertemplate: PARETO_HOVER_TEMPLATE,
      marker: {
        size: 9,
        color: COLORS.primary,
        symbol: "diamond",
        line: { color: COLORS.primaryDark, width: 1 },
        opacity: 0.95,
      },
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true,
    legend: { x: 0.01, y: 0.99, ...LEGEND_STYLE, font: { size: 12, color: COLORS.slate200 } },
    scene: {
      xaxis: { ...SCENE_AXIS, title: { text: "F1 Score ↑" } },
      yaxis: { ...SCENE_AXIS, title: { text: "Latency (ms) ↓" } },
      zaxis: { ...SCENE_AXIS, title: { text: "Interpretability ↑" } },
      bgcolor: COLORS.transparent,
      camera: { eye: { x: 1.6, y: -1.6, z: 0.8 } },
    },
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={CHART_CONFIG}
      useResizeHandler
      style={{ width: "100%", height: "520px" }}
    />
  );
};

export default ParetoFront3D;
