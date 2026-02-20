import numpy as np
from typing import List, Dict, Tuple, Any


class ParetoAnalyzer:
    """
    Analyzer for multi-objective optimization results.

    Provides methods for:
    - Pareto dominance checking
    - Non-dominated solution identification
    - Knee-point computation
    - Metric aggregation
    """

    @staticmethod
    def is_dominated(solution_a: Dict[str, float], solution_b: Dict[str, float],
                     objectives: List[str] = ["f1_score", "latency", "interpretability"],
                     maximize: List[bool] = [True, False, True]) -> bool:
        """
        Check if solution_a is dominated by solution_b.

        Solution A is dominated by B if B is at least as good in all objectives
        and strictly better in at least one.

        Args:
            solution_a: First solution
            solution_b: Second solution
            objectives: List of objective names
            maximize: Whether each objective should be maximized

        Returns:
            True if solution_a is dominated by solution_b
        """
        better_or_equal = True
        strictly_better = False

        for obj, is_max in zip(objectives, maximize):
            val_a = solution_a[obj]
            val_b = solution_b[obj]

            if is_max:
                # For maximization objectives
                if val_b < val_a:
                    better_or_equal = False
                    break
                if val_b > val_a:
                    strictly_better = True
            else:
                # For minimization objectives
                if val_b > val_a:
                    better_or_equal = False
                    break
                if val_b < val_a:
                    strictly_better = True

        return better_or_equal and strictly_better

    @staticmethod
    def get_pareto_front(solutions: List[Dict[str, Any]],
                         objectives: List[str] = ["f1_score", "latency", "interpretability"],
                         maximize: List[bool] = [True, False, True]) -> List[Dict[str, Any]]:
        """
        Extract the Pareto front from a set of solutions.

        Args:
            solutions: List of solution dictionaries
            objectives: List of objective names
            maximize: Whether each objective should be maximized

        Returns:
            List of non-dominated solutions
        """
        pareto_front = []

        for i, sol_a in enumerate(solutions):
            is_dominated_by_any = False

            for j, sol_b in enumerate(solutions):
                if i != j:
                    if ParetoAnalyzer.is_dominated(sol_a, sol_b, objectives, maximize):
                        is_dominated_by_any = True
                        break

            if not is_dominated_by_any:
                pareto_front.append(sol_a)

        return pareto_front

    @staticmethod
    def compute_knee_point(pareto_front: List[Dict[str, Any]],
                           objectives: List[str] = ["f1_score", "latency", "interpretability"],
                           maximize: List[bool] = [True, False, True]) -> Dict[str, Any]:
        """
        Compute the knee point of the Pareto front.

        The knee point is the solution with maximum distance to the utopia point
        in normalized objective space.

        Args:
            pareto_front: List of Pareto-optimal solutions
            objectives: List of objective names
            maximize: Whether each objective should be maximized

        Returns:
            Knee-point solution
        """
        if not pareto_front:
            return None

        if len(pareto_front) == 1:
            return pareto_front[0]

        # Extract objective values
        obj_values = []
        for obj in objectives:
            obj_values.append([sol[obj] for sol in pareto_front])
        obj_values = np.array(obj_values).T  # Shape: (n_solutions, n_objectives)

        # Normalize objectives to [0, 1]
        normalized = np.zeros_like(obj_values)
        for i, (is_max, obj_name) in enumerate(zip(maximize, objectives)):
            col = obj_values[:, i]
            min_val = np.min(col)
            max_val = np.max(col)

            if max_val - min_val > 1e-10:
                if is_max:
                    # For maximization: higher is better → normalize to [0, 1]
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    # For minimization: lower is better → invert
                    normalized[:, i] = 1.0 - (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 1.0

        # Utopia point (all objectives = 1 in normalized space)
        utopia = np.ones(len(objectives))

        # Compute distances to utopia point
        distances = np.linalg.norm(normalized - utopia, axis=1)

        # Knee point has minimum distance to utopia
        knee_idx = np.argmin(distances)

        return pareto_front[knee_idx]

    @staticmethod
    def compute_metrics(solutions: List[Dict[str, Any]],
                        pareto_front: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for a set of solutions.

        Args:
            solutions: List of solution dictionaries
            pareto_front: Optional pre-computed Pareto front (avoids O(n²) recomputation)

        Returns:
            Dictionary of metrics
        """
        if not solutions:
            return {}

        # Extract objective values
        f1_scores = [sol["f1_score"] for sol in solutions]
        latencies = [sol["latency"] for sol in solutions]
        interpretabilities = [sol["interpretability"] for sol in solutions]

        # Use pre-computed Pareto front if provided, otherwise compute it
        if pareto_front is None:
            pareto_front = ParetoAnalyzer.get_pareto_front(solutions)

        # Compute knee point
        knee_point = ParetoAnalyzer.compute_knee_point(pareto_front)

        metrics = {
            "total_solutions": len(solutions),
            "pareto_front_size": len(pareto_front),
            "dominated_solutions": len(solutions) - len(pareto_front),
            "f1_score": {
                "min": np.min(f1_scores),
                "max": np.max(f1_scores),
                "mean": np.mean(f1_scores),
                "std": np.std(f1_scores)
            },
            "latency": {
                "min": np.min(latencies),
                "max": np.max(latencies),
                "mean": np.mean(latencies),
                "std": np.std(latencies)
            },
            "interpretability": {
                "min": np.min(interpretabilities),
                "max": np.max(interpretabilities),
                "mean": np.mean(interpretabilities),
                "std": np.std(interpretabilities)
            },
            "knee_point": knee_point
        }

        return metrics

    @staticmethod
    def compare_solutions(sol_a: Dict[str, Any], sol_b: Dict[str, Any]) -> Dict[str, str]:
        """
        Compare two solutions across all objectives.

        Args:
            sol_a: First solution
            sol_b: Second solution

        Returns:
            Dictionary indicating which solution is better for each objective
        """
        comparison = {}

        objectives = [
            ("f1_score", True),
            ("latency", False),
            ("interpretability", True)
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


class MetricsTracker:
    """
    Track metrics during AutoML search.

    Useful for monitoring progress and generating plots.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.history = {
            "generation": [],
            "best_f1": [],
            "best_latency": [],
            "best_interpretability": [],
            "pareto_size": []
        }

    def update(self, generation: int, solutions: List[Dict[str, Any]]):
        """
        Update metrics for a generation.

        Args:
            generation: Generation number
            solutions: Solutions from this generation
        """
        self.history["generation"].append(generation)

        # Best objectives
        f1_scores = [sol["f1_score"] for sol in solutions]
        latencies = [sol["latency"] for sol in solutions]
        interpretabilities = [sol["interpretability"] for sol in solutions]

        self.history["best_f1"].append(max(f1_scores))
        self.history["best_latency"].append(min(latencies))
        self.history["best_interpretability"].append(max(interpretabilities))

        # Pareto front size
        pareto_front = ParetoAnalyzer.get_pareto_front(solutions)
        self.history["pareto_size"].append(len(pareto_front))

    def get_history(self) -> Dict[str, List]:
        """
        Get the full history.

        Returns:
            History dictionary
        """
        return self.history
