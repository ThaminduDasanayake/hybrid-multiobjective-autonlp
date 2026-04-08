def compute_hypervolume_history(result: dict) -> list:
    """Compute per-generation hypervolume history
    """
    pipelines = result.get("pipelines", [])
    if not pipelines:
        return []

    from collections import defaultdict

    import numpy as np

    from automl.pareto import get_pareto_front
    from utils.evaluation import ParetoAnalyzer

    # Group entries by generation (pipelines only contains status==success entries)
    by_gen: dict[int, list[dict]] = defaultdict(list)
    for entry in pipelines:
        gen = entry.get("generation")
        if gen is not None:
            by_gen[gen].append(entry)

    if not by_gen:
        return []

    # Compute global bounds across ALL pipelines for consistent normalization
    all_f1 = [e["f1_score"] for e in pipelines]
    all_lat = [e["latency"] for e in pipelines]
    all_interp = [e["interpretability"] for e in pipelines]
    bounds = {
        "f1_score": (float(np.min(all_f1)), float(np.max(all_f1))),
        "latency": (float(np.min(all_lat)), float(np.max(all_lat))),
        "interpretability": (float(np.min(all_interp)), float(np.max(all_interp))),
    }

    # Build cumulative solution set and compute HV at each generation
    generations = sorted(by_gen.keys())
    cumulative = []
    hypervolume_history = []

    for gen in generations:
        cumulative.extend(by_gen[gen])
        front = get_pareto_front(cumulative)
        hypervolume = ParetoAnalyzer.calculate_hypervolume(front, bounds=bounds)
        hypervolume_history.append(
            {"generation": gen, "hypervolume": round(hypervolume, 6)}
        )

    return hypervolume_history
