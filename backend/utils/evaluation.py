import numpy as np
from typing import List, Dict, Any

from pymoo.indicators.hv import HV

from automl.pareto import is_dominated, get_pareto_front


class ParetoAnalyzer:
    """Analyzes multi-objective optimization results: dominance, knee-point, hypervolume, and metrics."""

    @staticmethod
    def is_dominated(
        solution_a: Dict[str, float],
        solution_b: Dict[str, float],
        objectives: List[str] = ["f1_score", "latency", "interpretability"],
        maximize: List[bool] = [True, False, True],
    ) -> bool:
        return is_dominated(solution_a, solution_b, objectives, maximize)

    @staticmethod
    def get_pareto_front(
        solutions: List[Dict[str, Any]],
        objectives: List[str] = ["f1_score", "latency", "interpretability"],
        maximize: List[bool] = [True, False, True],
    ) -> List[Dict[str, Any]]:
        return get_pareto_front(solutions, objectives, maximize)

    @staticmethod
    def compute_knee_point(
        pareto_front: List[Dict[str, Any]],
        objectives: List[str] = ["f1_score", "latency", "interpretability"],
        maximize: List[bool] = [True, False, True],
    ) -> Dict[str, Any]:
        """Return the Pareto-front solution with minimum Euclidean distance to the utopia point."""
        if not pareto_front:
            return None

        if len(pareto_front) == 1:
            return pareto_front[0]

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

        objectives = [
            ("f1_score", True),
            ("latency", False),
            ("interpretability", True),
        ]

        for obj, maximize in objectives:
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
        """Return the pymoo hypervolume indicator for a Pareto front in normalised minimization space."""
        if not pareto_front_solutions:
            return 0.0

        f1_scores = np.array([s["f1_score"] for s in pareto_front_solutions])
        latencies = np.array([s["latency"] for s in pareto_front_solutions])
        interp_scores = np.array(
            [s["interpretability"] for s in pareto_front_solutions]
        )

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
        """Select one solution from the Pareto front by strategy: max_f1, min_latency, max_interp, or knee."""
        front = results.get("pareto_front", [])
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
