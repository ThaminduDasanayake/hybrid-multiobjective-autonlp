"""
Implements Pareto dominance logic to extract the non-dominated front of multi-objective solutions.
"""

from typing import Any, Dict, List

_DEFAULT_OBJECTIVES = ("f1_score", "latency", "interpretability")
_DEFAULT_MAXIMIZE = (True, False, True)


def is_dominated(
    solution_a: Dict[str, float],
    solution_b: Dict[str, float],
    objectives: List[str] = None,
    maximize: List[bool] = None,
) -> bool:
    """
    Evaluates whether solution_a is strictly Pareto-dominated by solution_b.
    A solution is dominated if it is no better on all objectives and strictly worse on at least one.
    """

    if objectives is None:
        objectives = _DEFAULT_OBJECTIVES

    if maximize is None:
        maximize = _DEFAULT_MAXIMIZE

    if len(objectives) != len(maximize):
        raise ValueError("`objectives` and `maximize` must have the same length")

    better_or_equal = True
    strictly_better = False

    for obj, is_max in zip(objectives, maximize, strict=True):
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
    objectives: List[str] = None,
    maximize: List[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts the Pareto-optimal front from a given population using an O(n²) pairwise dominance check.
    """
    if objectives is None:
        objectives = _DEFAULT_OBJECTIVES

    if maximize is None:
        maximize = _DEFAULT_MAXIMIZE

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
