import Plot from "react-plotly.js";
import {
  COLORS,
  AXIS_STYLE,
  CHART_LAYOUT,
  LEGEND_STYLE,
  CHART_CONFIG,
} from "@/utils/chartTheme.js";

const AblationBarChart = ({ masterMetrics, single, two, gaOnly }) => {
  const configs = [
    { label: "Full GA+BO", metrics: masterMetrics },
    { label: "GA-Only", metrics: gaOnly },
    { label: "2-Objective", metrics: two },
    { label: "Single-Obj", metrics: single },
  ].filter((c) => c.metrics != null);

  if (configs.length === 0) {
    return (
      <div className="chart-empty h-64">
        No ablation data available yet. Run ablation experiments to see the comparison.
      </div>
    );
  }

  const labels = configs.map((c) => c.label);

  const data = [
    {
      type: "bar",
      name: "Best F1",
      x: labels,
      y: configs.map((c) => c.metrics.best_f1 ?? 0),
      marker: { color: COLORS.primary, line: { color: COLORS.primaryDark, width: 1 } },
      hovertemplate: "%{x}<br>Best F1: %{y:.4f}<extra></extra>",
    },
    {
      type: "bar",
      name: "Hypervolume",
      x: labels,
      y: configs.map((c) => c.metrics.hypervolume ?? 0),
      marker: { color: COLORS.blue, line: { color: COLORS.blueDark, width: 1 } },
      hovertemplate: "%{x}<br>Hypervolume: %{y:.4f}<extra></extra>",
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 55, r: 20, t: 20, b: 55 },
    barmode: "group",
    showlegend: true,
    legend: { x: 0.01, y: 0.99, ...LEGEND_STYLE },
    xaxis: {
      ...AXIS_STYLE,
      zeroline: false,
    },
    yaxis: {
      ...AXIS_STYLE,
      title: { text: "Score", font: { size: 12, color: COLORS.slate400 } },
      zeroline: false,
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

export default AblationBarChart;
