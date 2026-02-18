import random
import time
import numpy as np
from typing import Callable, List, Dict, Any, Optional
from deap import base, creator, tools
from utils.logger import get_logger
from .persistence import ResultStore

from utils.formatting import format_time

logger = get_logger("search_engine")

class EvolutionarySearch:
    """
    Handles the Evolutionary Algorithm (NSGA-II) logic.
    """
    def __init__(self, 
                 population_size: int, 
                 n_generations: int, 
                 random_state: int,
                 result_store: ResultStore,
                 evaluate_fn: Callable[[list], tuple]):
        
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
            "scaler": [None, "standard", "minmax", "robust"],
            "dim_reduction": [None, "pca", "select_k_best"],
            "vectorizer": ["tfidf", "count"],
            "model": ["logistic", "naive_bayes", "svm", "random_forest", "sgd"],
            "ngram_range": ["1-1", "1-2", "1-3"],
            "max_features": [5000, 10000, 20000, "None"]
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
        
        # Track penalties per generation (Fix for data pollution)
        self.penalty_history = []

    def _setup_deap(self):
        # Create fitness class (multi-objective minimization)
        # Minimize negative F1 (max), latency (min), negative interpretability (max)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        
        # Register genes
        # Register genes with safe type casting (Fix 4 & 8)
        def safe_choice(key):
            val = np.random.choice(self.gene_pool[key])
            # Handle actual None values
            if val is None:
                return None
            # Handle string "None" for scaler/dim_reduction (should remain None)
            val_str = str(val)
            if val_str == "None":
                return None
            # Return as string for scaler and dim_reduction
            return val_str

        def safe_max_features():
            val = np.random.choice(self.gene_pool["max_features"])
            if str(val) == "None":
                return "None"
            return int(val)

        self.toolbox.register("attr_scaler", lambda: safe_choice("scaler"))
        self.toolbox.register("attr_dim_reduction", lambda: safe_choice("dim_reduction"))
        self.toolbox.register("attr_vectorizer", lambda: str(np.random.choice(self.gene_pool["vectorizer"])))
        self.toolbox.register("attr_model", lambda: str(np.random.choice(self.gene_pool["model"])))
        self.toolbox.register("attr_ngram_range", lambda: str(np.random.choice(self.gene_pool["ngram_range"])))
        self.toolbox.register("attr_max_features", safe_max_features)
        
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
                self.toolbox.attr_max_features
            ),
            n=1,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxTwoPoint) # TwoPoint is better for 4 genes
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.5) # Increased mutation rate for diversity
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self.evaluate_fn)

    def _mutate_individual(self, individual, indpb: float):
        # Mutate each gene with independent probability (Fix 7)
        if np.random.random() < indpb:
            val = np.random.choice(self.gene_pool["scaler"])
            individual[0] = None if val is None else (None if str(val) == "None" else str(val))
        if np.random.random() < indpb:
            val = np.random.choice(self.gene_pool["dim_reduction"])
            individual[1] = None if val is None else (None if str(val) == "None" else str(val))
        if np.random.random() < indpb:
            individual[2] = str(np.random.choice(self.gene_pool["vectorizer"]))
        if np.random.random() < indpb:
            individual[3] = str(np.random.choice(self.gene_pool["model"]))
        if np.random.random() < indpb:
            individual[4] = str(np.random.choice(self.gene_pool["ngram_range"]))
        if np.random.random() < indpb:
            val = np.random.choice(self.gene_pool["max_features"])
            individual[5] = int(val) if str(val) != "None" else "None"
            
        return (individual,)

    def _check_early_stopping(self, pareto_front: List, new_individuals_ratio: float) -> bool:
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
            logger.info(f"Early stopping: Pareto front unchanged for {self.stagnation_threshold} generations")
            return True
        return False

    def run(self, callback: Optional[Callable] = None):
        population = self.toolbox.population(n=self.population_size)
        hof = tools.ParetoFront()
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        
        for gen in range(self.n_generations):
            gen_start_time = time.time()
            logger.info(f"Generation {gen + 1}/{self.n_generations}")
            
            if callback:
                callback({
                    "current_generation": gen + 1,
                    "total_generations": self.n_generations,
                    "message": f"Running generation {gen + 1}/{self.n_generations}",
                    "progress": int((gen / self.n_generations) * 100)
                })

            # Check how many are new
            # We need to know before evaluation, or during?
            # Let's count how many have valid fitness (re-evaluated or cached) 
            # effectively "new" means not in cache before this generation loop started?
            # Actually, `evaluate_fn` (via evaluator) handles caching. 
            # We can check if fitness is valid before `evaluate`.
            # But in DEAP, invalid fitness means needs evaluation.
            
            # Identify invalid individuals (those that need evaluation)
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            
            # Count how many of these are NOT in the persistent cache
            # This requires access to result_store from here, which we have.
            new_individuals_count = 0
            for ind in invalid_ind:
                key = self.result_store.get_individual_key(ind)
                if not self.result_store.get_cached_evaluation(key):
                    new_individuals_count += 1
            
            new_individuals_ratio = new_individuals_count / len(population) if population else 0
            logger.info(f"New individuals ratio: {new_individuals_ratio:.2f}")

            # Evaluate
            evaluate_with_gen = lambda ind: self.toolbox.evaluate(ind, generation=gen)
            
            # Map evaluate only on invalid ones? The original code mapped on ALL.
            # "fitnesses = map(evaluate_with_gen, population)" -> re-evaluates everyone?
            # Typically in DEAP you only evaluate invalid ones. 
            # But the original code was:
            # fitnesses = map(evaluate_with_gen, population) 
            # for ind, fit in zip(population, fitnesses): ind.fitness.values = fit
            
            # We should stick to DEAP pattern: evaluate invalid
            # But wait, `population` is fresh from `toolbox.population()` in Gen 0, so all invalid.
            # In later gens, we select & clone, so fitnesses are inherited.
            # Then we mate/mutate and `del fitness.values`.
            
            # Identify invalid individuals (generic for all generations)
                
            # Evaluate invalid
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = map(evaluate_with_gen, invalid_ind)

            # Track failure reasons for logging
            failure_counts = {"penalty": 0, "error": 0, "valid": 0}

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if fit == (-1.0, 2.0, -1.0): # Penalty Fitness (Fix 1 & 5)
                    failure_counts["penalty"] += 1
                elif fit == (-0.5, -0.5, -0.5): # Error Fitness
                    failure_counts["error"] += 1
                else:
                    failure_counts["valid"] += 1

            # Track penalties for this generation
            self.penalty_history.append(failure_counts['penalty'])

            if failure_counts['penalty'] > 0:
                logger.info(f"Constraint violations: {failure_counts['penalty']} individuals penalized for invalid structures")

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
            # Structural Uniqueness Check & Heavy Mutation (Task 4)
            # ---------------------------------------------------------
            # For each offspring, check if it exists in the evaluation cache.
            # If so, force a "Heavy Mutation" to encourage exploration.
            
            max_retries = 3
            for i, ind in enumerate(offspring):
                if ind.fitness.valid:
                    continue  # Skip if already valid (e.g. from elitism/survival)
                
                # We need to construct the key to check cache
                # But notice: `del mutant.fitness.values` above makes them invalid
                # So mostly we are checking items that were mutated or mated.
                
                retries = 0
                while retries < max_retries:
                    # Check compatibility/cache
                    # Note: We can check if it stays in cache
                    key = self.result_store.get_individual_key(ind)
                    if self.result_store.get_cached_evaluation(key):
                        # Collision! This individual has been evaluated before.
                        # Apply Heavy Mutation: Mutate 3 random genes (Aggressive)
                        logger.info(f"Collision detected for {ind}. Applying Aggressive Mutation (3 genes).")
                        
                        # Pick 3 distinct gene indices to mutate
                        # Prioritize model (3) and vectorizer (2) if possible?
                        # For simplicity and effectiveness, standard random choice of 3 is fine.
                        gene_indices = np.random.choice(range(6), size=3, replace=False)
                        
                        # Mutate those genes guaranteed with type safety (Fix 7)
                        if 0 in gene_indices: 
                            val = np.random.choice(self.gene_pool["scaler"])
                            ind[0] = None if val is None else (None if str(val) == "None" else str(val))
                        if 1 in gene_indices: 
                            val = np.random.choice(self.gene_pool["dim_reduction"])
                            ind[1] = None if val is None else (None if str(val) == "None" else str(val))
                        if 2 in gene_indices: ind[2] = str(np.random.choice(self.gene_pool["vectorizer"]))
                        if 3 in gene_indices: ind[3] = str(np.random.choice(self.gene_pool["model"]))
                        if 4 in gene_indices: ind[4] = str(np.random.choice(self.gene_pool["ngram_range"]))
                        if 5 in gene_indices: 
                            val = np.random.choice(self.gene_pool["max_features"])
                            ind[5] = int(val) if str(val) != "None" else "None"
                        
                        del ind.fitness.values # Ensure it's invalid
                        retries += 1
                    else:
                        # Improved or unique enough
                        break
                
                # Fallback if still colliding (Fix 3)
                if retries >= max_retries:
                    key = self.result_store.get_individual_key(ind)
                    if self.result_store.get_cached_evaluation(key):
                        logger.warning(f"Failed to resolve collision after {max_retries} retries. Generating new individual.")
                        # Generate completely new individual
                        new_ind = self.toolbox.individual()
                        # Replace in offspring list
                        offspring[i] = new_ind
                        # Ensure it's invalid
                        del offspring[i].fitness.values
            # ---------------------------------------------------------

            # Recalculate new ratio for the offspring? 
            # The next loop will handle it.
            
            population[:] = offspring
            
            # Record stats
            gen_time = time.time() - gen_start_time
            self.result_store.add_time_stats(generation_time=gen_time)
            logger.info(f" Generation time: {format_time(gen_time)} | Valid: {failure_counts['valid']} | Penalties: {failure_counts['penalty']} | Errors: {failure_counts['error']}")
            
        return hof

    def get_penalty_history(self) -> List[int]:
        """Return list of penalty counts per generation."""
        return self.penalty_history
