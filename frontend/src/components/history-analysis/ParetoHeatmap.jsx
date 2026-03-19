import Plot from "react-plotly.js";
import { COLORS, CHART_LAYOUT, CHART_CONFIG_MINIMAL } from "@/utils/chartTheme.js";
import { MODEL_LABEL } from "@/constants";

const normalize = (values) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return values.map(() => 0.5);
  return values.map((v) => (v - min) / (max - min));
};

const ParetoHeatmap = ({ paretoFront = [] }) => {
  if (paretoFront.length === 0) {
    return <div className="chart-empty h-40">No Pareto-optimal solutions available.</div>;
  }

  const sorted = [...paretoFront].sort((a, b) => b.f1_score - a.f1_score);

  const f1Raw = sorted.map((s) => s.f1_score);
  const latRaw = sorted.map((s) => (s.latency ?? 0) * 1000);
  const intRaw = sorted.map((s) => s.interpretability);

  const f1Norm = normalize(f1Raw);
  const latNorm = normalize(latRaw).map((v) => 1 - v);
  const intNorm = normalize(intRaw);

  const rowLabels = sorted.map((s, i) => `#${i + 1} ${MODEL_LABEL[s.model] ?? s.model ?? "?"}`);
  const colLabels = ["F1 Score", "Latency (ms)", "Interpretability"];

  const z = sorted.map((_, i) => [f1Norm[i], latNorm[i], intNorm[i]]);

  const annotations = [];
  sorted.forEach((_, ri) => {
    const raw = [f1Raw[ri], latRaw[ri], intRaw[ri]];
    raw.forEach((val, ci) => {
      annotations.push({
        x: ci,
        y: ri,
        text: val.toFixed(3),
        showarrow: false,
        font: { size: 11, color: "#f8fafc" },
        xref: "x",
        yref: "y",
      });
    });
  });

  const height = Math.max(160, sorted.length * 44 + 60);

  const data = [
    {
      type: "heatmap",
      z,
      x: colLabels,
      y: rowLabels,
      colorscale: [
        [0, COLORS.slate950],
        [0.4, COLORS.slate800],
        [0.7, COLORS.amber],
        [1, COLORS.primary],
      ],
      showscale: false,
      hovertemplate: "%{y}<br>%{x}<extra></extra>",
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 90, r: 20, t: 40, b: 20 },
    font: { color: COLORS.slate400, size: 11 },
    xaxis: {
      tickfont: { color: COLORS.slate400, size: 12 },
      gridcolor: COLORS.slate800,
      side: "top",
      fixedrange: true,
    },
    yaxis: {
      tickfont: { color: COLORS.slate400, size: 11 },
      gridcolor: COLORS.slate800,
      autorange: "reversed",
      fixedrange: true,
    },
    annotations,
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={CHART_CONFIG_MINIMAL}
      useResizeHandler
      style={{ width: "100%", height: `${height}px` }}
    />
  );
};

export default ParetoHeatmap;
