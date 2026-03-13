import Plot from "react-plotly.js";
import { fmt } from "../utils/formatters";

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

const ParetoFront3D = ({ allSolutions = [], paretoFront = [] }) => {
  if (allSolutions.length === 0 && paretoFront.length === 0) {
    return (
      <div className="flex h-96 items-center justify-center rounded-xl border border-border bg-card text-sm text-muted-foreground">
        No solution data available for this run.
      </div>
    );
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
      marker: {
        size: 4,
        color: "rgba(148, 163, 184, 0.35)", // Muted slate for dominated solutions
        line: { width: 0 },
      },
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
        color: "#f97316", // Bright orange for Pareto front
        symbol: "diamond",
        line: { color: "#ea580c", width: 1 },
        opacity: 0.95,
      },
    },
  ];

  // DARK MODE STYLING
  const axisStyle = {
    gridcolor: "#1e293b", // Dark grid lines
    backgroundcolor: "rgba(0,0,0,0)", // Transparent background
    showspikes: false,
    tickfont: { size: 10, color: "#64748b" },
    titlefont: { size: 12, color: "#94a3b8" }, // Lighter text for dark mode
    zerolinecolor: "#1e293b",
  };

  const layout = {
    autosize: true,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: true,
    legend: {
      x: 0.01,
      y: 0.99,
      bgcolor: "rgba(2, 6, 23, 0.8)", // Dark translucent legend
      bordercolor: "#1e293b",
      borderwidth: 1,
      font: { size: 12, color: "#cbd5e1" },
    },
    scene: {
      xaxis: { ...axisStyle, title: { text: "F1 Score ↑" } },
      yaxis: { ...axisStyle, title: { text: "Latency (ms) ↓" } },
      zaxis: { ...axisStyle, title: { text: "Interpretability ↑" } },
      bgcolor: "rgba(0,0,0,0)", // Removes the grey 3D box
      camera: {
        eye: { x: 1.6, y: -1.6, z: 0.8 },
      },
    },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
    displaylogo: false,
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      useResizeHandler
      style={{ width: "100%", height: "520px" }}
    />
  );
};

export default ParetoFront3D;
