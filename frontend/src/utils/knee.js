/**
 * computeKnee(paretoFront) → solution object
 *
 * Identifies the "knee point" — the Pareto-optimal solution that offers the
 * best balanced trade-off across all three objectives (F1 ↑, Latency ↓,
 * Interpretability ↑).
 *
 * Algorithm:
 *   1. Normalise each objective to [0, 1] across the front.
 *   2. Flip latency so that higher = better (1 - normalised_latency).
 *   3. Find the solution with the smallest Euclidean distance to the ideal
 *      point (1, 1, 1) in the normalised objective space.
 *
 * If the front has only one solution it is trivially the knee.
 * Returns null for an empty front.
 */
export function computeKnee(paretoFront) {
  if (!paretoFront || paretoFront.length === 0) return null;
  if (paretoFront.length === 1) return paretoFront[0];

  const f1s = paretoFront.map((s) => s.f1_score ?? 0);
  const lats = paretoFront.map((s) => s.latency ?? 0);
  const interps = paretoFront.map((s) => s.interpretability ?? 0);

  const minMax = (arr) => [Math.min(...arr), Math.max(...arr)];

  const [minF1, maxF1] = minMax(f1s);
  const [minLat, maxLat] = minMax(lats);
  const [minInterp, maxInterp] = minMax(interps);

  // Normalise to [0,1]; guard against division by zero when all values are equal.
  const norm = (v, lo, hi) => (hi === lo ? 0.5 : (v - lo) / (hi - lo));

  let kneeIdx = 0;
  let minDist = Infinity;

  paretoFront.forEach((sol, i) => {
    const nf1 = norm(sol.f1_score ?? 0, minF1, maxF1); // higher = better → keep as-is
    const nlat = 1 - norm(sol.latency ?? 0, minLat, maxLat); // lower latency = better → flip
    const ninterp = norm(sol.interpretability ?? 0, minInterp, maxInterp); // higher = better → keep

    // Distance to ideal point (1, 1, 1)
    const dist = Math.sqrt((1 - nf1) ** 2 + (1 - nlat) ** 2 + (1 - ninterp) ** 2);

    if (dist < minDist) {
      minDist = dist;
      kneeIdx = i;
    }
  });

  return paretoFront[kneeIdx];
}
