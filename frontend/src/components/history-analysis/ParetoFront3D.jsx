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
  titlefont: { size: 12, color: COLORS.slate500 },
  zerolinecolor: COLORS.slate800,
  rangemode: "tozero",
};

const ParetoFront3D = ({ allSolutions = [], paretoFront = [] }) => {
  const validSolutions = allSolutions.filter((s) => s.latency != null && (s.latency * 1000) < 1e5);
  const validPareto = paretoFront.filter((s) => s.latency != null && (s.latency * 1000) < 1e5);

  if (validSolutions.length === 0 && validPareto.length === 0) {
    return <div className="chart-empty h-96">No valid solution data available for this run.</div>;
  }

  const data = [
    {
      type: "scatter3d",
      mode: "markers",
      name: `All Solutions (${validSolutions.length})`,
      x: validSolutions.map((s) => s.f1_score),
      y: validSolutions.map((s) => s.latency * 1000),
      z: validSolutions.map((s) => s.interpretability),
      customdata: buildCustomData(validSolutions),
      hovertemplate: HOVER_TEMPLATE,
      marker: { size: 4, color: COLORS.dominatedSolution, line: { width: 0 } },
    },
    {
      type: "scatter3d",
      mode: "markers",
      name: `Pareto Front (${validPareto.length})`,
      x: validPareto.map((s) => s.f1_score),
      y: validPareto.map((s) => s.latency * 1000),
      z: validPareto.map((s) => s.interpretability),
      customdata: buildCustomData(validPareto),
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
      xaxis: { ...SCENE_AXIS, title: { text: "F1 Score ↑" }, range: [0, 1.05], tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0] },
      yaxis: { ...SCENE_AXIS, title: { text: "Latency (ms) ↓" } },
      zaxis: { ...SCENE_AXIS, title: { text: "Interpretability ↑" }, range: [0, 1.05], tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0] },
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
