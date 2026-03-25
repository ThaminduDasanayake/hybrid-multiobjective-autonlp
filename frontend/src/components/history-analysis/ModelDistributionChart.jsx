import Plot from "react-plotly.js";
import { fmt } from "@/utils/formatters.js";
import {
  COLORS,
  AXIS_STYLE,
  CHART_LAYOUT,
  LEGEND_STYLE,
  CHART_CONFIG,
} from "@/utils/chartTheme.js";

const ModelDistributionChart = ({ allSolutions = [], paretoFront = [] }) => {
  if (allSolutions.length === 0 && paretoFront.length === 0) {
    return <div className="chart-empty h-64">No solution data available.</div>;
  }

  const countBy = (solutions) => {
    const counts = {};
    for (const s of solutions) {
      const label = fmt.model(s.model);
      counts[label] = (counts[label] ?? 0) + 1;
    }
    return counts;
  };

  const allCounts = countBy(allSolutions);
  const paretoCounts = countBy(paretoFront);

  const models = Object.keys(allCounts).sort((a, b) => allCounts[b] - allCounts[a]);

  const data = [
    {
      type: "bar",
      name: `All Solutions (${allSolutions.length})`,
      x: models,
      y: models.map((m) => allCounts[m] ?? 0),
      marker: { color: COLORS.dominatedSolutionMed, line: { color: COLORS.slate500, width: 1 } },
      hovertemplate: "<b>All Solutions</b><br>%{x}: %{y}<extra></extra>",
    },
    {
      type: "bar",
      name: `Pareto Front (${paretoFront.length})`,
      x: models,
      y: models.map((m) => paretoCounts[m] ?? 0),
      marker: { color: COLORS.primary, line: { color: COLORS.primaryDark, width: 1 } },
      hovertemplate: "<b>Pareto Front</b><br>%{x}: %{y}<extra></extra>",
    },
  ];

  const layout = {
    ...CHART_LAYOUT,
    margin: { l: 50, r: 20, t: 20, b: 80 },
    barmode: "group",
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0,
      y: -0.15,
      font: { size: 11, color: COLORS.slate500 },
    },
    xaxis: {
      ...AXIS_STYLE,
      zeroline: false,
    },
    yaxis: {
      ...AXIS_STYLE,
      title: { text: "Count", font: { size: 12, color: COLORS.slate500 } },
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

export default ModelDistributionChart;
