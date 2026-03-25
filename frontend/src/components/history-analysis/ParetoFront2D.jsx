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
  ]);
}

const ParetoFront2D = ({
  allSolutions = [],
  paretoFront = [],
  xKey,
  yKey,
  xLabel,
  yLabel,
  xScale = 1,
}) => {
  const validSolutions = allSolutions.filter((s) => s.latency == null || (s.latency * 1000) < 1e5);
  const validPareto = paretoFront.filter((s) => s.latency == null || (s.latency * 1000) < 1e5);

  if (validSolutions.length === 0 && validPareto.length === 0) {
    return <div className="chart-empty h-64">No solution data available.</div>;
  }

  const xVal = (s) => (s[xKey] ?? 0) * xScale;
  const yVal = (s) => s[yKey] ?? 0;

  const sortedPareto = [...validPareto].sort((a, b) => xVal(a) - xVal(b));

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
      name: `All Solutions (${validSolutions.length})`,
      x: validSolutions.map(xVal),
      y: validSolutions.map(yVal),
      customdata: buildCustomData(validSolutions),
      hovertemplate: hoverTpl,
      marker: { size: 6, color: COLORS.dominatedSolution, line: { width: 0 } },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: `Pareto Front (${validPareto.length})`,
      x: sortedPareto.map(xVal),
      y: sortedPareto.map(yVal),
      customdata: buildCustomData(sortedPareto),
      hovertemplate: hoverTpl,
      marker: {
        size: 10,
        color: COLORS.primary,
        symbol: "diamond",
        line: { color: COLORS.primaryDark, width: 1 },
        opacity: 0.95,
      },
      line: { color: COLORS.primary, width: 1.5, dash: "dot" },
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 60, r: 20, t: 20, b: 55 },
    showlegend: true,
    legend: { x: 0.01, y: 0.01, ...LEGEND_STYLE },
    xaxis: {
      ...AXIS_STYLE,
      title: { text: xLabel, font: { size: 12, color: COLORS.slate500 } },
    },
    yaxis: {
      ...AXIS_STYLE,
      title: { text: yLabel, font: { size: 12, color: COLORS.slate500 } },
    },
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={CHART_CONFIG}
      useResizeHandler
      style={{ width: "100%", height: "360px" }}
    />
  );
};

export default ParetoFront2D;
