import Plot from "react-plotly.js";
import { COLORS, AXIS_STYLE, CHART_LAYOUT, CHART_CONFIG } from "@/utils/chartTheme.js";

const METRICS = [
  {
    key: "best_f1",
    label: "Best F1 Score ↑",
    color: COLORS.primary,
    colorDark: COLORS.primaryDark,
  },
  {
    key: "best_latency_ms",
    label: "Best Latency (ms) ↓",
    color: COLORS.primary,
    colorDark: COLORS.primaryDark,
  },
  {
    key: "best_interpretability",
    label: "Best Interpretability ↑",
    color: COLORS.primary,
    colorDark: COLORS.primaryDark,
  },
  {
    key: "hypervolume",
    label: "Hypervolume ↑",
    color: COLORS.primary,
    colorDark: COLORS.primaryDark,
  },
];

const AblationBarChart = ({ masterMetrics, single, two, gaOnly, randomSearch }) => {
  const configs = [
    { label: "Full GA+BO", metrics: masterMetrics },
    { label: "GA-Only", metrics: gaOnly },
    { label: "2-Objective", metrics: two },
    { label: "Single-Obj", metrics: single },
    { label: "Random", metrics: randomSearch },
  ].filter((c) => c.metrics != null);

  if (configs.length === 0) {
    return (
      <div className="chart-empty h-64">
        No ablation data available yet. Run ablation experiments to see the comparison.
      </div>
    );
  }

  const labels = configs.map((c) => c.label);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {METRICS.map((m) => {
        const values = configs.map((c) => c.metrics[m.key] ?? 0);

        const min = Math.min(...values);
        const max = Math.max(...values);
        const span = max - min || max * 0.1 || 0.1;
        const pad = span * 0.15;

        const data = [
          {
            type: "scatter",
            mode: "lines+markers",
            x: labels,
            y: values,
            marker: { color: m.color, size: 10, line: { color: m.colorDark, width: 1 } },
            line: { color: m.color, width: 3 },
            hovertemplate: `%{x}<br>${m.label}: %{y:.4f}<extra></extra>`,
          },
        ];

        const layout = {
          ...CHART_LAYOUT,
          margin: { l: 50, r: 10, t: 30, b: 50 },
          showlegend: false,
          title: {
            text: m.label,
            font: { size: 13, color: COLORS.slate500 },
            x: 0.5,
            y: 0.97,
          },
          xaxis: {
            ...AXIS_STYLE,
            zeroline: false,
          },
          yaxis: {
            ...AXIS_STYLE,
            zeroline: false,
            autorange: false,
            range: [min - pad, max + pad],
          },
        };

        return (
          <Plot
            key={m.key}
            data={data}
            layout={layout}
            config={CHART_CONFIG}
            useResizeHandler
            style={{ width: "100%", height: "220px" }}
          />
        );
      })}
    </div>
  );
};

export default AblationBarChart;
