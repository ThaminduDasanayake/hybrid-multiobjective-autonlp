import time
import hashlib
import json
import random
from typing import List, Dict, Tuple, Any
import numpy as np
from deap import base, creator, tools, algorithms
from .bayesian_optimization import BayesianOptimizer


class HybridAutoML:
    """
    Hybrid AutoML system combining Genetic Algorithm for architecture search
    and Bayesian Optimization for hyperparameter tuning.

    This system jointly optimizes three objectives:
    1. Predictive performance (F1 score)
    2. Computational efficiency (inference time)
    3. Intrinsic interpretability
    """

    def __init__(self, X_train: list, y_train: np.ndarray,
                 population_size: int = 20, n_generations: int = 10,
                 bo_calls: int = 15, random_state: int = 42, early_stopping: bool = True, ):
        """
        Initialize the HybridAutoML system.

        Args:
            X_train: Training texts
            y_train: Training labels
            population_size: Size of GA population
            n_generations: Number of GA generations
            bo_calls: Number of BO iterations per pipeline
            random_state: Random seed for reproducibility
            early_stopping: Enable early stopping if Pareto front stagnates
        """
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.n_generations = n_generations
        self.bo_calls = bo_calls
        self.random_state = random_state
        self.early_stopping = early_stopping

        # Set random seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)

        # Core state
        self.eval_cache = {}  # Cache evaluation results
        self.search_history = []  # Track all evaluated individuals
        self.gene_pool = {
            "vectorizer": ["tfidf", "count"],
            "model": ["logistic", "naive_bayes", "svm", "random_forest"]
        }

        # Track objective ranges for normalization
        self.objective_ranges = {
            "f1_score": {"min": float('inf'), "max": float('-inf')},
            "latency": {"min": float('inf'), "max": float('-inf')},
            "interpretability": {"min": float('inf'), "max": float('-inf')}
        }

        # Early stopping tracking
        self.stagnation_threshold = 3  # Stop if no improvement for N generations
        self.stagnation_counter = 0
        self.last_pareto_size = 0
        self.last_pareto_hash = None

        # Initialize Bayesian Optimizer
        self.bo_optimizer = BayesianOptimizer(
            n_calls=bo_calls,
            cv=3,
            random_state=random_state
        )

        # Set up DEAP framework
        self._setup_deap()

    def _setup_deap(self):
        # Set up DEAP genetic algorithm framework
        # Create fitness class (multi-objective minimization)
        # Minimize negative F1, latency, and negative interpretability
        if not hasattr(creator, "FitnessMulti"):
            # creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Gene generators
        self.toolbox.register("attr_vectorizer",
                              self._random_choice,
                              self.gene_pool["vectorizer"])
        self.toolbox.register("attr_model",
                              self._random_choice,
                              self.gene_pool["model"])

        # Individual generator
        self.toolbox.register("individual",
                              tools.initCycle,
                              creator.Individual,
                              (self.toolbox.attr_vectorizer, self.toolbox.attr_model),
                              n=1)

        # Population generator
        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.3)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate_individual)

    def _random_choice(self, options):
        # Controlled random choice using numpy for reproducibility
        return np.random.choice(options)

    def _mutate_individual(self, individual, indpb: float):
        """
        Mutate an individual by randomly changing genes.

        Args:
            individual: Individual to mutate
            indpb: Probability of mutating each gene

        Returns:
            Mutated individual (tuple for DEAP compatibility)
        """
        if np.random.random() < indpb:
            individual[0] = np.random.choice(self.gene_pool["vectorizer"])
        if np.random.random() < indpb:
            individual[1] = np.random.choice(self.gene_pool["model"])
        return individual,

    def _individual_to_key(self, individual: list) -> str:
        """
        Convert individual to a hashable cache key.

        Args:
            individual: GA individual

        Returns:
            Cache key string
        """
        config = {
            "vectorizer": individual[0],
            "model": individual[1]
        }
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def _normalize_objective(self, value: float, obj_name: str,
                             minimize: bool = False) -> float:
        """
        Normalize objective value to [0, 1] range. This ensures NSGA-II treats all objectives fairly.
        Without normalization, F1 (0.2-0.6) would dominate latency (1e-4).

        Args:
            value: Raw objective value
            obj_name: Name of objective ('f1_score', 'latency', 'interpretability')
            minimize: Whether this objective should be minimized

        Returns:
            Normalized value in [0, 1]
        """
        # Update ranges
        if value < self.objective_ranges[obj_name]["min"]:
            self.objective_ranges[obj_name]["min"] = value
        if value > self.objective_ranges[obj_name]["max"]:
            self.objective_ranges[obj_name]["max"] = value

        min_val = self.objective_ranges[obj_name]["min"]
        max_val = self.objective_ranges[obj_name]["max"]

        # Avoid division by zero
        if abs(max_val - min_val) < 1e-10:
            return 0.5

        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)

        # If minimizing, invert so larger is still better for NSGA-II
        if minimize:
            normalized = 1.0 - normalized

        return normalized

    def _evaluate_individual(self, individual: list) -> Tuple[float, float, float]:
        """
        Evaluate an individual using Bayesian Optimization.

        This is the core evaluation function that:
        1. Checks the cache
        2. Runs BO to optimize hyperparameters
        3. Computes all three objectives
        4. Caches the result

        Args:
            individual: GA individual [vectorizer_type, model_type]

        Returns:
            Tuple of (neg_f1, latency, neg_interpretability)
            We use negatives because DEAP minimizes by default
        """
        cache_key = self._individual_to_key(individual)

        # Check cache
        if cache_key in self.eval_cache:
            cached = self.eval_cache[cache_key]
            return cached["f1_score"], cached["latency"], cached["interpretability"]

            # return -cached["neg_f1"], cached["latency"], -cached["neg_interpretability"]

        vectorizer_type = individual[0]
        model_type = individual[1]

        try:
            # Run Bayesian Optimization
            bo_result = self.bo_optimizer.optimize(
                vectorizer_type, model_type,
                self.X_train, self.y_train
            )

            f1_score = bo_result["best_score"]
            latency = bo_result["inference_time"]

            # Compute interpretability score
            interpretability = self.intrinsic_interpretability_score(
                vectorizer_type, model_type, bo_result["best_params"]
            )

            # Normalize objectives
            # F1 and interpretability: maximize (minimize=False)
            # Latency: minimize (minimize=True, so gets inverted)
            norm_f1 = self._normalize_objective(f1_score, "f1_score", minimize=False)
            norm_latency = self._normalize_objective(latency, "latency", minimize=True)
            norm_interp = self._normalize_objective(interpretability, "interpretability", minimize=False)

            # Store in cache with both raw and normalized values
            self.eval_cache[cache_key] = {
                "vectorizer": vectorizer_type,
                "model": model_type,
                "params": bo_result["best_params"],
                "f1_score": f1_score,
                "latency": latency,
                "interpretability": interpretability,
                # # Normalized values for NSGA-II
                # "norm_f1": norm_f1,
                # "norm_latency": norm_latency,
                # "norm_interpretability": norm_interp,
                # # Negatives for DEAP (minimization)
                # "neg_norm_f1": -norm_f1,
                # "neg_norm_latency": -norm_latency,
                # "neg_norm_interpretability": -norm_interp,
                "variance": bo_result["variance"]
            }

            # Add to search history
            self.search_history.append({
                "generation": len(self.search_history) // self.population_size,
                "vectorizer": vectorizer_type,
                "model": model_type,
                "f1_score": f1_score,
                "latency": latency,
                "interpretability": interpretability
            })

            return -norm_f1, -norm_latency, -norm_interp

        except Exception as e:
            print(f"Error evaluating individual {vectorizer_type}-{model_type}: {e}")
            # Return poor fitness values
            return -0.5, -0.5, -0.5

    def intrinsic_interpretability_score(self, vectorizer_type: str,
                                         model_type: str,
                                         params: Dict[str, Any]) -> float:
        """
        Compute intrinsic interpretability score for a pipeline.

        Interpretability is based on:
        1. Model complexity (simpler is better)
        2. Feature transparency
        3. Hyperparameter simplicity

        Score ranges from 0 (not interpretable) to 1 (highly interpretable).

        Args:
            vectorizer_type: Type of vectorizer
            model_type: Type of model
            params: Hyperparameters

        Returns:
            Interpretability score between 0 and 1
        """
        score = 0.0

        # Model complexity component (40% weight)
        model_scores = {
            "logistic": 1.0,  # Linear, coefficients interpretable
            "naive_bayes": 0.9,  # Probabilistic, simple
            "svm": 0.7,  # Linear but less interpretable
            "random_forest": 0.4  # Ensemble, black box
        }
        score += 0.4 * model_scores.get(model_type, 0.5)

        # Feature transparency component (30% weight)
        vectorizer_scores = {
            "count": 1.0,  # Raw counts, most transparent
            "tfidf": 0.8  # Weighted, still interpretable
        }
        score += 0.3 * vectorizer_scores.get(vectorizer_type, 0.5)

        # Hyperparameter simplicity component (30% weight)
        simplicity = 0.0

        # Prefer smaller ngram ranges
        ngram_max = params.get("ngram_range_max", 1)
        simplicity += 0.4 * (1.0 / ngram_max)

        # Prefer fewer features
        max_features = params.get("max_features", 5000)
        simplicity += 0.3 * (1.0 - min(max_features / 10000, 1.0))

        # Model-specific simplicity
        if model_type == "logistic":
            # Prefer stronger regularization (smaller C)
            C = params.get("C", 1.0)
            simplicity += 0.3 * (1.0 / (1.0 + C))
        elif model_type == "random_forest":
            # Prefer shallower trees
            max_depth = params.get("max_depth", 10)
            simplicity += 0.3 * (1.0 / (1.0 + max_depth / 10))
        else:
            simplicity += 0.3 * 0.5  # Default

        score += 0.3 * simplicity

        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    def _check_early_stopping(self, pareto_front: List) -> bool:
        """
        Check if search has stagnated and should stop early.

        Args:
            pareto_front: Current Pareto front

        Returns:
            True if it should stop early
        """
        if not self.early_stopping:
            return False

        current_size = len(pareto_front)

        # Create hash of Pareto front solutions
        pareto_configs = set()
        for ind in pareto_front:
            key = self._individual_to_key(ind)
            pareto_configs.add(key)
        current_hash = hash(frozenset(pareto_configs))

        # Check if Pareto front has changed
        if current_hash == self.last_pareto_hash:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_pareto_hash = current_hash

        self.last_pareto_size = current_size

        # Stop if stagnated for threshold generations
        if self.stagnation_counter >= self.stagnation_threshold:
            print(f"\nEarly stopping: Pareto front unchanged for {self.stagnation_threshold} generations")
            return True

        return False

    def run(self) -> Dict[str, Any]:
        """
        Run the hybrid AutoML optimization.

        Returns:
            Dictionary containing:
                - pareto_front: List of non-dominated solutions
                - all_solutions: All evaluated solutions
                - search_history: Generation-by-generation history
                - stats: Summary statistics
        """
        print(f"Starting Hybrid AutoML with {self.population_size} individuals "
              f"for {self.n_generations} generations...")
        print(f"Random seed: {self.random_state}")
        print(f"Early stopping: {'enabled' if self.early_stopping else 'disabled'}")

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Hall of fame (Pareto front)
        hof = tools.ParetoFront()

        # Custom evolution with early stopping
        for gen in range(self.n_generations):
            print(f"\nGeneration {gen + 1}/{self.n_generations}")

            # Evaluate population
            fitnesses = map(self.toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Update hall of fame
            hof.update(population)

            # Check early stopping
            if self._check_early_stopping(hof):
                print(f"Stopped at generation {gen + 1}")
                break

            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.7:  # Crossover probability
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < 0.3:  # Mutation probability
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            # population[:] = offspring
            population[:] = tools.selBest(population + offspring, self.population_size)

            # Print statistics
            record = stats.compile(population)
            print(f"  Pareto front size: {len(hof)}")
            print(f"  Unique configs evaluated: {len(self.eval_cache)}")

        # # Run evolutionary algorithm
        # algorithms.eaMuPlusLambda(
        #     population, self.toolbox,
        #     mu=self.population_size,
        #     lambda_=self.population_size,
        #     cxpb=0.7,  # Crossover probability
        #     mutpb=0.3,  # Mutation probability
        #     ngen=self.n_generations,
        #     stats=stats,
        #     halloffame=hof,
        #     verbose=True
        # )

        # Extract Pareto front
        pareto_solutions = []
        for ind in hof:
            key = self._individual_to_key(ind)
            if key in self.eval_cache:
                solution = {
                    "vectorizer": self.eval_cache[key]["vectorizer"],
                    "model": self.eval_cache[key]["model"],
                    "params": self.eval_cache[key]["params"],
                    "f1_score": self.eval_cache[key]["f1_score"],
                    "latency": self.eval_cache[key]["latency"],
                    "interpretability": self.eval_cache[key]["interpretability"],
                    "variance": self.eval_cache[key]["variance"]
                }
                pareto_solutions.append(solution)

        # Get all solutions
        all_solutions = []
        for cached in self.eval_cache.values():
            solution = {
                "vectorizer": cached["vectorizer"],
                "model": cached["model"],
                "params": cached["params"],
                "f1_score": cached["f1_score"],
                "latency": cached["latency"],
                "interpretability": cached["interpretability"],
                "variance": cached["variance"]
            }
            all_solutions.append(solution)

        print(f"\nSearch complete")
        print(f"  Total evaluations: {len(self.eval_cache)}")
        print(f"  Pareto front size (population): {len(pareto_solutions)}")
        print(f"  Objective ranges:")
        print(f"    F1: [{self.objective_ranges['f1_score']['min']:.4f}, "
              f"{self.objective_ranges['f1_score']['max']:.4f}]")
        print(f"    Latency: [{self.objective_ranges['latency']['min']:.4f}, "
              f"{self.objective_ranges['latency']['max']:.4f}]")
        print(f"    Interpretability: [{self.objective_ranges['interpretability']['min']:.4f}, "
              f"{self.objective_ranges['interpretability']['max']:.4f}]")

        return {
            "pareto_front": pareto_solutions,
            "all_solutions": all_solutions,
            "search_history": self.search_history,
            "stats": {
                "total_evaluations": len(self.eval_cache),
                "unique_configurations": len(self.eval_cache),
                "pareto_size": len(pareto_solutions),
                "objective_ranges": self.objective_ranges
            }
        }
