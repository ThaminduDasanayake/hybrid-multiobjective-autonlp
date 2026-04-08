"""
Implements the NSGA-II evolutionary algorithm for discrete pipeline architecture search.
"""

import random
import time
from typing import Callable, Iterable, List, Optional

import numpy as np
from deap import base, creator, tools

from utils.formatting import format_time
from utils.logger import get_logger

from .evaluator import ERROR_FITNESS, PENALTY_FITNESS
from .persistence import ResultStore

logger = get_logger("search_engine")

# Optimization weights for (F1, Latency, Interpretability)
# Small negative epsilon (-1e-10) is used to suppress objectives during ablation without triggering DEAP ZeroDivisionErrors
OPTIMIZATION_MODES = {
    "multi_3d": (1.0, -1.0, 1.0),  # Default: all three objectives active
    "single_f1": (
        1.0,
        -1e-10,
        -1e-10,
    ),  # Ablation: F1 only (latency and interpretability suppressed)
    "multi_2d": (1.0, -1.0, -1e-10),  # Ablation: F1 + Latency only
    "random_search": (
        1.0,
        -1.0,
        1.0,
    ),  # Baseline: same weights as multi_3d, but GA operators are bypassed entirely
}


class EvolutionarySearch:
    """
    Executes NSGA-II multi-objective search over a 6-gene discrete pipeline space.
    """

    def __init__(
        self,
        population_size: int,
        n_generations: int,
        random_state: int,
        result_store: ResultStore,
        evaluate_fn: Callable[[list], tuple],
        optimization_mode: str = "multi_3d",
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.5,
        stagnation_threshold: int = 3,
    ):

        self.population_size = population_size
        self.n_generations = n_generations
        self.random_state = random_state
        self.result_store = result_store
        self.evaluate_fn = evaluate_fn
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        if optimization_mode not in OPTIMIZATION_MODES:
            raise ValueError(
                f"Unknown optimization_mode '{optimization_mode}'. "
                f"Choose from: {list(OPTIMIZATION_MODES.keys())}"
            )
        self.optimization_mode = optimization_mode
        self._weights = OPTIMIZATION_MODES[optimization_mode]

        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.last_pareto_hash = None

        # Define discrete search space for pipeline components
        self.gene_pool = {
            # scaler: how to normalize feature values before the model sees them
            "scaler": [None, "maxabs", "robust"],
            # dim_reduction: optional feature selection / compression step
            "dim_reduction": [None, "select_k_best"],
            # vectorizer: how raw text is converted into numbers (bag-of-words)
            "vectorizer": ["tfidf", "count"],
            # Restricts models to interpretable linear types. Ensemble models excluded
            "model": ["logistic", "naive_bayes", "svm"],
            # ngram_range: whether to use single words (1-1) or word pairs (1-2)
            "ngram_range": ["1-1", "1-2"],
            # max_features: vocabulary size cap (None = unlimited)
            "max_features": [5000, 10000, "None"],
        }

        # Set seeds
        random.seed(random_state)
        np.random.seed(random_state)

        self._setup_deap()

        # Track penalties per generation
        self.penalty_history = []

    def _random_gene(self, gene_name: str):
        """Sample a random value from the gene pool with proper type safety."""
        val = np.random.choice(self.gene_pool[gene_name])
        if gene_name == "max_features":
            return int(val) if str(val) != "None" else "None"
        if val is None or str(val) == "None":
            return None
        return str(val)

    def _setup_deap(self):
        # Generate unique DEAP classes based on fitness weights to prevent collisions during ablation studies
        weight_tag = abs(hash(self._weights))
        self._fitness_cls_name = f"FitnessMulti_{weight_tag}"
        self._individual_cls_name = f"Individual_{weight_tag}"

        if not hasattr(creator, self._fitness_cls_name):
            creator.create(self._fitness_cls_name, base.Fitness, weights=self._weights)
        if not hasattr(creator, self._individual_cls_name):
            creator.create(self._individual_cls_name, list, fitness=getattr(creator, self._fitness_cls_name))

        logger.info(f"DEAP configured for mode '{self.optimization_mode}' with weights {self._weights} (classes: {self._fitness_cls_name}, {self._individual_cls_name})")

        individual_cls = getattr(creator, self._individual_cls_name)

        self.toolbox = base.Toolbox()

        # Register attribute generation functions for pipeline genes
        self.toolbox.register("attr_scaler", lambda: self._random_gene("scaler"))
        self.toolbox.register("attr_dim_reduction", lambda: self._random_gene("dim_reduction"))
        self.toolbox.register("attr_vectorizer", lambda: self._random_gene("vectorizer"))
        self.toolbox.register("attr_model", lambda: self._random_gene("model"))
        self.toolbox.register("attr_ngram_range", lambda: self._random_gene("ngram_range"))
        self.toolbox.register("attr_max_features", lambda: self._random_gene("max_features"))

        # Define individual structure
        self.toolbox.register(
            "individual",
            tools.initCycle,
            individual_cls,
            (
                self.toolbox.attr_scaler,
                self.toolbox.attr_dim_reduction,
                self.toolbox.attr_vectorizer,
                self.toolbox.attr_model,
                self.toolbox.attr_ngram_range,
                self.toolbox.attr_max_features,
            ),
            n=1,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register uniform crossover and mutation suitable for categorical genes
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.5)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self.evaluate_fn)

    _GENE_NAMES = [
        "scaler",
        "dim_reduction",
        "vectorizer",
        "model",
        "ngram_range",
        "max_features",
    ]

    def _mutate_individual(self, individual, indpb: float):
        """Mutate each gene with independent probability."""
        for i, gene_name in enumerate(self._GENE_NAMES):
            if np.random.random() < indpb:
                individual[i] = self._random_gene(gene_name)
        return (individual,)

    def _check_early_stopping(
        self, pareto_front: Iterable, new_individuals_ratio: float
    ) -> bool:
        """Check for stagnation by monitoring changes to the Pareto front."""
        # Only check stagnation if the generation introduced genuinely novel individuals
        if new_individuals_ratio < 0.2:
            return False

        # Hash current Pareto front to track stagnation across generations
        pareto_configs = {
            self.result_store.get_individual_key(ind) for ind in pareto_front
        }
        current_hash = hash(frozenset(pareto_configs))

        if current_hash == self.last_pareto_hash:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_pareto_hash = current_hash

        if self.stagnation_counter >= self.stagnation_threshold:
            logger.info(
                f"Early stopping: Pareto front unchanged for {self.stagnation_threshold} generations"
            )
            return True
        return False

    def run(self, callback: Optional[Callable] = None):
        """Execute the configured search strategy (Random or NSGA-II)."""
        population = self.toolbox.population(n=self.population_size)

        # Execute random search baseline bypassing all GA operators
        if self.optimization_mode == "random_search":
            total_budget = self.population_size * self.n_generations
            population = self.toolbox.population(n=total_budget)
            logger.info(
                f"Random search: evaluating {total_budget} random individuals "
                f"(pop={self.population_size} x gen={self.n_generations})"
            )

            batch_start_time = time.time()
            batch_penalties = 0
            for i, ind in enumerate(population):
                pseudo_gen = i // self.population_size
                fit = self.toolbox.evaluate(ind, generation=pseudo_gen)
                ind.fitness.values = fit

                if fit == PENALTY_FITNESS:
                    batch_penalties += 1

                # End of a pseudo-generation batch
                if (i + 1) % self.population_size == 0:
                    batch_time = time.time() - batch_start_time
                    self.result_store.add_time_stats(generation_time=batch_time)
                    self.penalty_history.append(batch_penalties)

                    if callback:
                        pseudo_gen = (i + 1) // self.population_size
                        callback(
                            {
                                "current_generation": pseudo_gen,
                                "total_generations": self.n_generations,
                                "message": f"Random search: evaluated {i + 1}/{total_budget}",
                                "progress": int(((i + 1) / total_budget) * 100),
                            }
                        )

                    # Reset for next batch
                    batch_start_time = time.time()
                    batch_penalties = 0

            logger.info(f"Random search completed ({total_budget} evaluations)")
            return

        # Execute NSGA-II evolutionary loop
        hof = tools.ParetoFront()

        for gen in range(self.n_generations):
            gen_start_time = time.time()
            logger.info(f"Generation {gen + 1}/{self.n_generations}")

            if callback:
                callback(
                    {
                        "current_generation": gen + 1,
                        "total_generations": self.n_generations,
                        "message": f"Running generation {gen + 1}/{self.n_generations}",
                        "progress": int((gen / self.n_generations) * 100),
                    }
                )

            # Identify individuals requiring evaluation
            invalid_ind = [ind for ind in population if not ind.fitness.valid]

            # Calculate ratio of genuinely novel architectures to prevent false early stopping
            new_individuals_count = sum(
                1
                for ind in invalid_ind
                if not self.result_store.peek(self.result_store.get_individual_key(ind))
            )
            new_individuals_ratio = (
                new_individuals_count / len(population) if population else 0
            )
            logger.info(f"New individuals ratio: {new_individuals_ratio:.2f}")

            # Evaluate invalid individuals
            def evaluate_with_gen(individual):
                return self.toolbox.evaluate(individual, generation=gen)

            fitnesses = map(evaluate_with_gen, invalid_ind)

            # Track failure reasons for logging
            failure_counts = {"penalty": 0, "error": 0, "valid": 0}

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if fit == PENALTY_FITNESS:
                    failure_counts["penalty"] += 1
                elif fit == ERROR_FITNESS:
                    failure_counts["error"] += 1
                else:
                    failure_counts["valid"] += 1

            # Track penalties for this generation
            self.penalty_history.append(failure_counts["penalty"])

            if failure_counts["penalty"] > 0:
                logger.info(
                    f"Constraint violations: {failure_counts['penalty']} individuals penalized for invalid structures"
                )

            hof.update(population)

            if self._check_early_stopping(hof, new_individuals_ratio):
                logger.info(f"Stopped at generation {gen + 1}")
                break

            # Apply NSGA-II selection, crossover, and mutation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            population[:] = offspring

            gen_time = time.time() - gen_start_time
            self.result_store.add_time_stats(generation_time=gen_time)
            logger.info(
                f" Generation time: {format_time(gen_time)} | Valid: {failure_counts['valid']} | Penalties: {failure_counts['penalty']} | Errors: {failure_counts['error']}"
            )

    def get_penalty_history(self) -> List[int]:
        """Return list of penalty counts per generation."""
        return self.penalty_history
