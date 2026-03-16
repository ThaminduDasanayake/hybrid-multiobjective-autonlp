import Plot from "react-plotly.js";
import { PRIMARY_COLOR } from "@/constants.js";

const ConvergenceChart = ({ searchHistory = [] }) => {
  if (searchHistory.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-border bg-card text-sm text-muted-foreground">
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

  // --- NEW: Calculate exact bounds to prevent the axis from dropping to 0 ---
  const minScore = Math.min(...maxF1);
  const maxScore = Math.max(...maxF1);
  const yPadding = (maxScore - minScore) * 0.1 || 0.01; // Adds a 10% visual padding

  const data = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Max F1 per Generation",
      x: generations,
      y: maxF1,
      line: {
        color: PRIMARY_COLOR,
        width: 2,
        shape: "spline",
      },
      fill: "tozeroy",
      fillcolor: "rgba(249, 115, 22, 0.1)", // Faint transparent orange fill
      marker: { size: 7, color: PRIMARY_COLOR, symbol: "circle" },
      hovertemplate: "Generation %{x}<br>Max F1: %{y:.4f}<extra></extra>",
    },
  ];

  const layout = {
    autosize: true,
    margin: { l: 60, r: 20, t: 20, b: 55 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
    xaxis: {
      title: { text: "Generation", font: { size: 12, color: "#94a3b8" } },
      gridcolor: "#1e293b",
      zeroline: false,
      tickfont: { size: 10, color: "#64748b" },
      dtick: 1,
    },
    yaxis: {
      title: { text: "Max F1 Score ↑", font: { size: 12, color: "#94a3b8" } },
      gridcolor: "#1e293b",
      zeroline: false,
      tickfont: { size: 10, color: "#64748b" },
      range: [minScore - yPadding, maxScore + yPadding],
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

export default ConvergenceChart;
