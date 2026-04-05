"""
THE ACCOUNTANT — utils/evaluation.py
======================================
After the AutoML search finishes, this module computes the final summary analytics
that the frontend displays: hypervolume, knee-point, objective statistics, and the
comparison table between different solutions.

It sits one layer above the raw dominance primitives in automl/pareto.py.
ParetoAnalyzer delegates dominance checks to those primitives and adds:
  - Hypervolume indicator (a scalar measure of Pareto front quality)
  - Knee-point detection (the "best balanced" trade-off on the front)
  - Summary statistics (mean, std, min, max for each objective)
  - Head-to-head solution comparison for the UI's comparison panel
"""

import numpy as np
from typing import List, Dict, Any

from pymoo.indicators.hv import HV

from automl.pareto import is_dominated, get_pareto_front

# Immutable module-level constants for the three-objective setup.
# Using tuples (not lists) and defining them once at module level avoids:
#   1. Mutable default argument footgun — Python evaluates list defaults once at
#      class definition time; any accidental mutation would corrupt future calls.
#   2. Repeated allocation — these are constructed on every method call otherwise.
_DEFAULT_OBJECTIVES: tuple = ("f1_score", "latency", "interpretability")
_DEFAULT_MAXIMIZE: tuple = (True, False, True)
# Paired (objective, maximize) tuple used by compare_solutions.
_COMPARE_OBJECTIVES: tuple = (("f1_score", True), ("latency", False), ("interpretability", True))


class ParetoAnalyzer:
    """High-level analytics on top of the Pareto dominance primitives.

    Used by worker.py to produce the final metrics payload written to MongoDB,
    and by server.py to compute the hypervolume convergence history for the chart.
    """

    @staticmethod
    def is_dominated(
        solution_a: Dict[str, float],
        solution_b: Dict[str, float],
        objectives: tuple = None,
        maximize: tuple = None,
    ) -> bool:
        return is_dominated(
            solution_a,
            solution_b,
            objectives if objectives is not None else _DEFAULT_OBJECTIVES,
            maximize if maximize is not None else _DEFAULT_MAXIMIZE,
        )

    @staticmethod
    def get_pareto_front(
        solutions: List[Dict[str, Any]],
        objectives: tuple = None,
        maximize: tuple = None,
    ) -> List[Dict[str, Any]]:
        return get_pareto_front(
            solutions,
            objectives if objectives is not None else _DEFAULT_OBJECTIVES,
            maximize if maximize is not None else _DEFAULT_MAXIMIZE,
        )

    @staticmethod
    def compute_knee_point(
        pareto_front: List[Dict[str, Any]],
        objectives: tuple = None,
        maximize: tuple = None,
    ) -> Dict[str, Any]:
        """Return the Pareto-front solution that best balances all objectives simultaneously.

        The "utopia point" is the hypothetical perfect solution: the best F1, the lowest
        latency, and the highest interpretability all at once. In practice, no real solution
        achieves this — the Pareto front shows us the actual trade-offs.

        The knee point is the Pareto-front solution whose normalised objective vector
        is closest (Euclidean distance) to the utopia point. It is the "least bad" solution
        in all directions at once — the natural recommendation when you don't have a strong
        preference for one objective over another.
        """
        if not pareto_front:
            return None

        if len(pareto_front) == 1:
            return pareto_front[0]

        if objectives is None:
            objectives = _DEFAULT_OBJECTIVES
        if maximize is None:
            maximize = _DEFAULT_MAXIMIZE

        obj_values = np.array(
            [[sol[obj] for sol in pareto_front] for obj in objectives]
        ).T  # shape: (n_solutions, n_objectives)

        # Normalise each objective to [0, 1] in the direction that maximisation = 1.
        normalized = np.zeros_like(obj_values)
        for i, (is_max, obj_name) in enumerate(zip(maximize, objectives)):
            col = obj_values[:, i]
            min_val = np.min(col)
            max_val = np.max(col)

            if max_val - min_val > 1e-10:
                if is_max:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    normalized[:, i] = 1.0 - (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 1.0

        utopia = np.ones(len(objectives))
        distances = np.linalg.norm(normalized - utopia, axis=1)
        knee_idx = np.argmin(distances)

        return pareto_front[knee_idx]

    @staticmethod
    def compute_metrics(
        solutions: List[Dict[str, Any]], pareto_front: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute summary metrics including Pareto front size, hypervolume, and knee point."""
        if not solutions:
            return {}

        f1_scores = [sol["f1_score"] for sol in solutions]
        latencies = [sol["latency"] for sol in solutions]
        interpretabilities = [sol["interpretability"] for sol in solutions]

        if pareto_front is None:
            pareto_front = ParetoAnalyzer.get_pareto_front(solutions)

        knee_point = ParetoAnalyzer.compute_knee_point(pareto_front)

        metrics = {
            "total_solutions": len(solutions),
            "pareto_front_size": len(pareto_front),
            "dominated_solutions": len(solutions) - len(pareto_front),
            "f1_score": {
                "min": np.min(f1_scores),
                "max": np.max(f1_scores),
                "mean": np.mean(f1_scores),
                "std": np.std(f1_scores),
            },
            "latency": {
                "min": np.min(latencies),
                "max": np.max(latencies),
                "mean": np.mean(latencies),
                "std": np.std(latencies),
            },
            "interpretability": {
                "min": np.min(interpretabilities),
                "max": np.max(interpretabilities),
                "mean": np.mean(interpretabilities),
                "std": np.std(interpretabilities),
            },
            "knee_point": knee_point,
            "hypervolume": ParetoAnalyzer.calculate_hypervolume(
                pareto_front,
                bounds={
                    "f1_score": (float(np.min(f1_scores)), float(np.max(f1_scores))),
                    "latency": (float(np.min(latencies)), float(np.max(latencies))),
                    "interpretability": (
                        float(np.min(interpretabilities)),
                        float(np.max(interpretabilities)),
                    ),
                },
            ),
        }

        return metrics

    @staticmethod
    def compare_solutions(
        sol_a: Dict[str, Any], sol_b: Dict[str, Any]
    ) -> Dict[str, str]:
        """Return per-objective comparison dict ('A', 'B', or 'tie') for two solutions."""
        comparison = {}

        for obj, maximize in _COMPARE_OBJECTIVES:
            val_a = sol_a[obj]
            val_b = sol_b[obj]

            if abs(val_a - val_b) < 1e-6:
                comparison[obj] = "tie"
            elif maximize:
                comparison[obj] = "A" if val_a > val_b else "B"
            else:
                comparison[obj] = "A" if val_a < val_b else "B"

        return comparison

    @staticmethod
    def calculate_hypervolume(
        pareto_front_solutions: List[Dict[str, Any]],
        bounds: Dict[str, tuple[float, float]] = None,
        ref_point: np.ndarray = None,
    ) -> float:
        """Return the hypervolume indicator for the Pareto front.

        Hypervolume measures the volume of the objective space that is "dominated"
        by the Pareto front relative to a fixed worst-case reference point. A larger
        hypervolume means the front covers more of the space — i.e., the system found
        better trade-off solutions.

        Two normalisation steps keep the value comparable across runs:
        1. Each objective is scaled to [0, 1] using global min/max bounds.
        2. Maximised objectives (F1, interpretability) are flipped (1 - value) so the
           whole problem is expressed as minimization — which is what pymoo's HV expects.

        The reference point [1.1, 1.1, 1.1] sits slightly beyond the worst possible
        normalised value (1.0) in each dimension, ensuring all Pareto-front solutions
        contribute a positive volume.
        """
        if not pareto_front_solutions:
            return 0.0

        # Single-pass construction: one Python-level loop, one numpy allocation,
        # three O(1) column views — vs. three separate loops and three temp lists.
        _raw = np.array(
            [[s["f1_score"], s["latency"], s["interpretability"]] for s in pareto_front_solutions]
        )
        f1_scores, latencies, interp_scores = _raw[:, 0], _raw[:, 1], _raw[:, 2]

        def _normalise(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
            if hi - lo > 1e-10:
                return (arr - lo) / (hi - lo)
            return np.zeros_like(arr)

        if bounds is None:
            bounds = {
                "f1_score": (float(f1_scores.min()), float(f1_scores.max())),
                "latency": (float(latencies.min()), float(latencies.max())),
                "interpretability": (
                    float(interp_scores.min()),
                    float(interp_scores.max()),
                ),
            }

        f1_norm = _normalise(f1_scores, *bounds["f1_score"])
        lat_norm = _normalise(latencies, *bounds["latency"])
        interp_norm = _normalise(interp_scores, *bounds["interpretability"])

        # Convert to minimization form: negate maximized objectives.
        F = np.column_stack(
            [
                1.0 - f1_norm,
                lat_norm,
                1.0 - interp_norm,
            ]
        )

        if ref_point is None:
            ref_point = np.array([1.1, 1.1, 1.1])

        hv_indicator = HV(ref_point=ref_point)
        return float(hv_indicator.do(F))

    @staticmethod
    def select_from_pareto(
        results: Dict[str, Any],
        strategy: str = "max_f1",
    ) -> Dict[str, Any]:
        """Select one solution from the Pareto front using a named deployment strategy.

        The Pareto front gives the user a set of trade-off solutions, not a single answer.
        This method lets a caller express a preference and get a concrete recommendation:
          - max_f1:      best classification accuracy (e.g., academic benchmarking)
          - min_latency: fastest inference (e.g., real-time production systems)
          - max_interp:  most interpretable (e.g., regulated domains like finance/healthcare)
          - knee:        balanced trade-off — the natural default when no priority is set
        """
        front = [p for p in results.get("pipelines", []) if p.get("is_pareto_optimal")]
        if not front:
            raise ValueError("Pareto front is empty — cannot select a solution.")

        if strategy == "max_f1":
            return max(front, key=lambda s: s["f1_score"])
        elif strategy == "min_latency":
            return min(front, key=lambda s: s["latency"])
        elif strategy == "max_interp":
            return max(front, key=lambda s: s["interpretability"])
        elif strategy == "knee":
            return ParetoAnalyzer.compute_knee_point(front) or front[0]
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: max_f1, min_latency, max_interp, knee.")
