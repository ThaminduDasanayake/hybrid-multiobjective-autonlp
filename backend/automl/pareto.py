"""
THE SORTING HAT — pareto.py
=============================
This module implements the core Pareto dominance logic that separates the "good"
pipelines from the "bad" ones. In multi-objective optimisation, there is rarely
a single "best" solution — instead, there is a set of trade-off solutions where
improving one objective forces a sacrifice on another.

A solution is Pareto-optimal (on the Pareto front) if no other solution beats it
on ALL objectives at the same time. For example, a high-F1 but slow pipeline and
a fast but slightly lower-F1 pipeline can both be Pareto-optimal — neither dominates
the other, because each wins on a different objective.

Architecture note: This module contains only the raw dominance primitives.
ParetoAnalyzer in utils/evaluation.py imports these and adds higher-level analytics
(hypervolume, knee-point detection, solution comparison) on top. Keeping primitives
here ensures the experiments/ scripts can import them without pulling in the full
evaluation stack.
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
    """Return True if solution_a is Pareto-dominated by solution_b.

    Plain English: B dominates A if B is *no worse* than A on every objective,
    AND *strictly better* than A on at least one objective. If B is better on
    F1 but worse on latency, neither dominates the other — both survive.

    The `maximize` list tells us the direction for each objective:
      True  = higher is better (F1, interpretability)
      False = lower is better (latency)
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
    """Return the subset of solutions that are not Pareto-dominated by any other solution.

    Uses a brute-force O(n²) pairwise comparison. This is intentional and safe here:
    a typical run evaluates fewer than 200 unique pipelines, so n² is at most ~40,000
    comparisons — negligible. A more complex algorithm (e.g., fast non-dominated sort)
    would add code complexity with no measurable benefit at this scale.
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
