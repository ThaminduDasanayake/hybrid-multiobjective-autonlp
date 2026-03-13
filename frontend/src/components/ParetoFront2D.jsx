import Plot from "react-plotly.js";
import { fmt } from "../utils/formatters";

function buildCustomData(solutions) {
  return solutions.map((s) => [
    fmt.model(s.model),
    fmt.vectorizer(s.vectorizer),
    fmt.scaler(s.scaler),
  ]);
}

// DARK MODE STYLING
const AXIS_STYLE = {
  gridcolor: "#1e293b", // Dark grid lines
  zerolinecolor: "#1e293b",
  tickfont: { size: 10, color: "#64748b" },
};

const ParetoFront2D = ({
  allSolutions = [],
  paretoFront = [],
  xKey,
  yKey,
  xLabel,
  yLabel,
  xScale = 1,
}) => {
  if (allSolutions.length === 0 && paretoFront.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-card text-sm text-muted-foreground">
        No solution data available.
      </div>
    );
  }

  const xVal = (s) => (s[xKey] ?? 0) * xScale;
  const yVal = (s) => s[yKey] ?? 0;

  const sortedPareto = [...paretoFront].sort((a, b) => xVal(a) - xVal(b));

  const hoverTpl =
    "<b>%{customdata[0]}</b><br>" +
    "Vectorizer: %{customdata[1]}<br>" +
    "Scaler: %{customdata[2]}<br>" +
    "────────────────<br>" +
    `${xLabel}: %{x:.4f}<br>` +
    `${yLabel}: %{y:.4f}` +
    "<extra></extra>";

  const data = [
    {
      type: "scatter",
      mode: "markers",
      name: `All Solutions (${allSolutions.length})`,
      x: allSolutions.map(xVal),
      y: allSolutions.map(yVal),
      customdata: buildCustomData(allSolutions),
      hovertemplate: hoverTpl,
      marker: {
        size: 6,
        color: "rgba(148, 163, 184, 0.35)", // Muted slate
        line: { width: 0 },
      },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: `Pareto Front (${paretoFront.length})`,
      x: sortedPareto.map(xVal),
      y: sortedPareto.map(yVal),
      customdata: buildCustomData(sortedPareto),
      hovertemplate: hoverTpl,
      marker: {
        size: 10,
        color: "#f97316", // Bright orange
        symbol: "diamond",
        line: { color: "#ea580c", width: 1 },
        opacity: 0.95,
      },
      line: { color: "#f97316", width: 1.5, dash: "dot" },
    },
  ];

  const layout = {
    autosize: true,
    margin: { l: 60, r: 20, t: 20, b: 55 },
    paper_bgcolor: "rgba(0,0,0,0)", // Transparent background
    plot_bgcolor: "rgba(0,0,0,0)", // Transparent background
    showlegend: true,
    legend: {
      x: 0.01,
      y: 0.01,
      bgcolor: "rgba(2, 6, 23, 0.8)", // Dark translucent legend
      bordercolor: "#1e293b",
      borderwidth: 1,
      font: { size: 11, color: "#cbd5e1" },
    },
    xaxis: {
      ...AXIS_STYLE,
      title: { text: xLabel, font: { size: 12, color: "#94a3b8" } },
    },
    yaxis: {
      ...AXIS_STYLE,
      title: { text: yLabel, font: { size: 12, color: "#94a3b8" } },
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
      style={{ width: "100%", height: "360px" }}
    />
  );
};

export default ParetoFront2D;
