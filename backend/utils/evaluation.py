"""
Computes summary analytics for multi-objective optimization, including hypervolume and knee-point detection.
"""

from typing import Any, Dict, List

import numpy as np
# pymoo: Blank & Deb (2020). "pymoo: Multi-Objective Optimization in Python." IEEE Access, 8, 89497-89509.
# https://doi.org/10.1109/ACCESS.2020.2990567
from pymoo.indicators.hv import HV

from automl.pareto import get_pareto_front, is_dominated

# Immutable objective configurations prevent mutable default argument bugs
_DEFAULT_OBJECTIVES: tuple = ("f1_score", "latency", "interpretability")
_DEFAULT_MAXIMIZE: tuple = (True, False, True)
_COMPARE_OBJECTIVES: tuple = (
    ("f1_score", True),
    ("latency", False),
    ("interpretability", True),
)


class ParetoAnalyzer:
    """Provides high-level analytics on top of core Pareto dominance primitives."""

    @staticmethod
    def is_dominated(
        solution_a: Dict[str, float],
        solution_b: Dict[str, float],
        objectives: tuple = None,
        maximize: tuple = None,
    ) -> bool:
        """Pass-through to primitive dominance check."""
        return is_dominated(
            solution_a,
            solution_b,
            list(objectives if objectives is not None else _DEFAULT_OBJECTIVES),
            list(maximize if maximize is not None else _DEFAULT_MAXIMIZE),
        )

    @staticmethod
    def get_pareto_front(
        solutions: List[Dict[str, Any]],
        objectives: tuple = None,
        maximize: tuple = None,
    ) -> List[Dict[str, Any]]:
        """Pass-through to primitive Pareto front extraction."""
        return get_pareto_front(
            solutions,
            list(objectives if objectives is not None else _DEFAULT_OBJECTIVES),
            list(maximize if maximize is not None else _DEFAULT_MAXIMIZE),
        )

    @staticmethod
    def compute_knee_point(
        pareto_front: List[Dict[str, Any]],
        objectives: tuple = None,
        maximize: tuple = None,
    ) -> Dict[str, Any] | None:
        """Identifies the Pareto-optimal solution with the minimum Euclidean distance to the utopia point [1,1,1]."""
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
        ).T

        # Normalize objectives to [0, 1] mapped toward the maximization direction
        normalized = np.zeros_like(obj_values)
        for i, (is_max, obj_name) in enumerate(zip(maximize, objectives)):
            col = obj_values[:, i]
            min_val = float(np.min(col))
            max_val = float(np.max(col))

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
        """Performs a direct objective-by-objective comparison between two solutions."""
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
        """Calculates the hypervolume indicator for the Pareto front after normalizing objectives to [0, 1]."""
        if not pareto_front_solutions:
            return 0.0

        # Single-pass construction: one Python-level loop, one numpy allocation,
        # three O(1) column views — vs. three separate loops and three temp lists.
        _raw = np.array(
            [
                [s["f1_score"], s["latency"], s["interpretability"]]
                for s in pareto_front_solutions
            ]
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

        # Convert to minimization format for pymoo
        F = np.column_stack(
            [
                1.0 - f1_norm,
                lat_norm,
                1.0 - interp_norm,
            ]
        )

        # Use a reference point slightly worse than the maximum normalized bounds
        if ref_point is None:
            ref_point = np.array([1.1, 1.1, 1.1])

        hv_indicator = HV(ref_point=ref_point)
        return float(hv_indicator.do(F))

    @staticmethod
    def select_from_pareto(
        results: Dict[str, Any],
        strategy: str = "max_f1",
    ) -> Dict[str, Any]:
        """Selects a specific Pareto-optimal solution"""
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
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: max_f1, min_latency, max_interp, knee."
            )
