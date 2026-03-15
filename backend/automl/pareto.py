"""
Pareto dominance utilities used by the core AutoML engine.

Extracted here so that automl/ does not depend on experiments/ — the correct
dependency direction is: experiments → automl, not the other way around.

experiments/evaluation.ParetoAnalyzer delegates to these functions for
dominance checking and front extraction; it adds hypervolume, knee-point,
and metrics aggregation on top.
"""

from typing import Any, Dict, List


def is_dominated(
    solution_a: Dict[str, float],
    solution_b: Dict[str, float],
    objectives: List[str] = ["f1_score", "latency", "interpretability"],
    maximize: List[bool] = [True, False, True],
) -> bool:
    """Return True if solution_a is Pareto-dominated by solution_b.

    B dominates A when B is at least as good in every objective and strictly
    better in at least one.
    """
    better_or_equal = True
    strictly_better = False

    for obj, is_max in zip(objectives, maximize):
        val_a = solution_a[obj]
        val_b = solution_b[obj]

        if is_max:
            if val_b < val_a:
                better_or_equal = False
                break
            if val_b > val_a:
                strictly_better = True
        else:
            if val_b > val_a:
                better_or_equal = False
                break
            if val_b < val_a:
                strictly_better = True

    return better_or_equal and strictly_better


def get_pareto_front(
    solutions: List[Dict[str, Any]],
    objectives: List[str] = ["f1_score", "latency", "interpretability"],
    maximize: List[bool] = [True, False, True],
) -> List[Dict[str, Any]]:
    """Return the subset of solutions that are not Pareto-dominated."""
    pareto_front = []
    for i, sol_a in enumerate(solutions):
        dominated = any(
            is_dominated(sol_a, sol_b, objectives, maximize)
            for j, sol_b in enumerate(solutions)
            if i != j
        )
        if not dominated:
            pareto_front.append(sol_a)
    return pareto_front
