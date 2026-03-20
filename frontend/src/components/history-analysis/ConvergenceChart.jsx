import Plot from "react-plotly.js";
import { COLORS, AXIS_STYLE, CHART_LAYOUT, CHART_CONFIG } from "@/utils/chartTheme.js";

const ConvergenceChart = ({ searchHistory = [] }) => {
  if (searchHistory.length === 0) {
    return (
      <div className="chart-empty h-64">
        No search history data available.
      </div>
    );
  }

  // Max F1 per generation.
  const genMap = {};
  for (const entry of searchHistory) {
    const g = entry.generation;
    if (genMap[g] === undefined || entry.f1_score > genMap[g]) {
      genMap[g] = entry.f1_score;
    }
  }
  const generations = Object.keys(genMap)
    .map(Number)
    .sort((a, b) => a - b);
  const maxF1 = generations.map((g) => genMap[g]);

  const minScore = Math.min(...maxF1);
  const maxScore = Math.max(...maxF1);
  const yPadding = (maxScore - minScore) * 0.1 || 0.01;

  const data = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Max F1 per Generation",
      x: generations,
      y: maxF1,
      line: { color: COLORS.primary, width: 2, shape: "spline" },
      fill: "tozeroy",
      fillcolor: COLORS.orangeFill,
      marker: { size: 7, color: COLORS.primary, symbol: "circle" },
      hovertemplate: "Generation %{x}<br>Max F1: %{y:.4f}<extra></extra>",
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 60, r: 20, t: 20, b: 55 },
    showlegend: false,
    xaxis: {
      ...AXIS_STYLE,
      title: { text: "Generation", font: { size: 12, color: COLORS.slate400 } },
      zeroline: false,
      dtick: 1,
    },
    yaxis: {
      ...AXIS_STYLE,
      title: { text: "Max F1 Score ↑", font: { size: 12, color: COLORS.slate400 } },
      zeroline: false,
      range: [minScore - yPadding, maxScore + yPadding],
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

export default ConvergenceChart;
