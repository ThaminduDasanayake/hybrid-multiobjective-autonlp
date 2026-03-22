import Plot from "react-plotly.js";
import { COLORS, AXIS_STYLE, CHART_LAYOUT, CHART_CONFIG } from "@/utils/chartTheme.js";

const HypervolumeConvergenceChart = ({ hvHistory = [] }) => {
  if (hvHistory.length === 0) {
    return <div className="chart-empty h-64">No hypervolume history available.</div>;
  }

  const generations = hvHistory.map((h) => h.generation);
  const hypervolumes = hvHistory.map((h) => h.hypervolume);

  const minHV = Math.min(...hypervolumes);
  const maxHV = Math.max(...hypervolumes);
  const yPadding = (maxHV - minHV) * 0.1 || 0.01;

  const data = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Hypervolume per Generation",
      x: generations,
      y: hypervolumes,
      line: { color: COLORS.primary, width: 2, shape: "spline" },
      fill: "tozeroy",
      fillcolor: COLORS.orangeFill,
      marker: { size: 7, color: COLORS.primary, symbol: "circle" },
      hovertemplate: "Generation %{x}<br>HV: %{y:.4f}<extra></extra>",
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
      title: { text: "Hypervolume ↑", font: { size: 12, color: COLORS.slate400 } },
      zeroline: false,
      range: [minHV - yPadding, maxHV + yPadding],
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

export default HypervolumeConvergenceChart;
