import random
import time
import numpy as np
from typing import Callable, List, Dict, Any, Optional
from deap import base, creator, tools
from utils.logger import get_logger
from .persistence import ResultStore
from .evaluator import PENALTY_FITNESS, ERROR_FITNESS

from utils.formatting import format_time

logger = get_logger("search_engine")


class EvolutionarySearch:
    """
    Handles the Evolutionary Algorithm (NSGA-II) logic.
    """

    def __init__(
        self,
        population_size: int,
        n_generations: int,
        random_state: int,
        result_store: ResultStore,
        evaluate_fn: Callable[[list], tuple],
    ):

        self.population_size = population_size
        self.n_generations = n_generations
        self.random_state = random_state
        self.result_store = result_store
        self.evaluate_fn = evaluate_fn

        self.stagnation_threshold = 3
        self.stagnation_counter = 0
        self.last_pareto_hash = None

        # Expanded gene pool (Structural Complexity)
        self.gene_pool = {
            "scaler": [None, "standard", "maxabs", "robust"],
            "dim_reduction": [None, "pca", "select_k_best"],
            "vectorizer": ["tfidf", "count"],
            "model": ["logistic", "naive_bayes", "svm", "random_forest", "sgd"],
            "ngram_range": ["1-1", "1-2", "1-3"],
            "max_features": [5000, 10000, 20000, "None"],
        }

        # Conditionally add LightGBM (Fix 6)
        try:
            import lightgbm

            self.gene_pool["model"].append("lightgbm")
        except ImportError:
            logger.warning("LightGBM not found. Excluding from search space.")

        # Set seeds
        random.seed(random_state)
        np.random.seed(random_state)

        self._setup_deap()

        # Track penalties per generation
        self.penalty_history = []

    def _random_gene(self, gene_name: str):
        """
        Sample a random value from the gene pool with proper type safety.
        Centralised to avoid duplicated logic across mutation and initialisation.
        """
        val = np.random.choice(self.gene_pool[gene_name])
        if gene_name == "max_features":
            return int(val) if str(val) != "None" else "None"
        if val is None or str(val) == "None":
            return None
        return str(val)

    def _setup_deap(self):
        # NOTE: DEAP's creator.create modifies global module state. The hasattr
        # guards prevent re-creation but also prevent changing weights across
        # multiple EvolutionarySearch instances in the same process.
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register gene attribute factories using _random_gene
        self.toolbox.register("attr_scaler", lambda: self._random_gene("scaler"))
        self.toolbox.register(
            "attr_dim_reduction", lambda: self._random_gene("dim_reduction")
        )
        self.toolbox.register(
            "attr_vectorizer", lambda: self._random_gene("vectorizer")
        )
        self.toolbox.register("attr_model", lambda: self._random_gene("model"))
        self.toolbox.register(
            "attr_ngram_range", lambda: self._random_gene("ngram_range")
        )
        self.toolbox.register(
            "attr_max_features", lambda: self._random_gene("max_features")
        )

        # Individual structure: [scaler, dim_reduction, vectorizer, model, ngram_range, max_features]
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
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

        self.toolbox.register("mate", tools.cxTwoPoint)
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
        self, pareto_front: List, new_individuals_ratio: float
    ) -> bool:
        # Only check stagnation if we are actually exploring new space
        # If > 20% of population was new, we are exploring.
        if new_individuals_ratio < 0.2:
            # If we are just re-evaluating old stuff, don't count towards stagnation
            # or maybe we should? If we can't find anything new, maybe we are done?
            # User requirement: "Only count generations where at least 20% of the population was actually new"
            return False

        # Create hash of Pareto front solutions
        pareto_configs = set()
        for ind in pareto_front:
            key = self.result_store.get_individual_key(ind)
            pareto_configs.add(key)
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
        population = self.toolbox.population(n=self.population_size)
        # HOF is used for early stopping only — final Pareto front is
        # recomputed from all evaluated solutions in hybrid_automl.py.
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

            # Identify invalid individuals (those that need evaluation)
            invalid_ind = [ind for ind in population if not ind.fitness.valid]

            # Count how many are truly new (not in the persistent cache)
            new_individuals_count = sum(
                1
                for ind in invalid_ind
                if not self.result_store.get_cached_evaluation(
                    self.result_store.get_individual_key(ind)
                )
            )
            new_individuals_ratio = (
                new_individuals_count / len(population) if population else 0
            )
            logger.info(f"New individuals ratio: {new_individuals_ratio:.2f}")

            # Evaluate invalid individuals
            evaluate_with_gen = lambda ind: self.toolbox.evaluate(ind, generation=gen)
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
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
            self.result_store.save_checkpoint()

            if self._check_early_stopping(hof, new_individuals_ratio):
                logger.info(f"Stopped at generation {gen + 1}")
                break

            # Selection & Offspring
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < 0.5:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # ---------------------------------------------------------
            # Structural Uniqueness Check & Heavy Mutation
            # ---------------------------------------------------------
            # For each offspring, check if it exists in the evaluation cache.
            # If so, force a heavy mutation to encourage exploration.

            max_retries = 3
            for i, ind in enumerate(offspring):
                if ind.fitness.valid:
                    continue

                retries = 0
                while retries < max_retries:
                    key = self.result_store.get_individual_key(ind)
                    if self.result_store.get_cached_evaluation(key):
                        logger.info(
                            f"Collision detected for {ind}. Applying Aggressive Mutation (3 genes)."
                        )
                        gene_indices = np.random.choice(range(6), size=3, replace=False)
                        for gi in gene_indices:
                            ind[gi] = self._random_gene(self._GENE_NAMES[gi])
                        del ind.fitness.values
                        retries += 1
                    else:
                        break

                # Fallback: generate a completely new individual if still colliding
                if retries >= max_retries:
                    key = self.result_store.get_individual_key(ind)
                    if self.result_store.get_cached_evaluation(key):
                        logger.warning(
                            f"Failed to resolve collision after {max_retries} retries. Generating new individual."
                        )
                        new_ind = self.toolbox.individual()
                        offspring[i] = new_ind
                        del offspring[i].fitness.values
            # ---------------------------------------------------------

            population[:] = offspring

            gen_time = time.time() - gen_start_time
            self.result_store.add_time_stats(generation_time=gen_time)
            logger.info(
                f" Generation time: {format_time(gen_time)} | Valid: {failure_counts['valid']} | Penalties: {failure_counts['penalty']} | Errors: {failure_counts['error']}"
            )

    def get_penalty_history(self) -> List[int]:
        """Return list of penalty counts per generation."""
        return self.penalty_history
