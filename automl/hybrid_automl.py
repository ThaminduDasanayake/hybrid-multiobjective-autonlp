import random
import time

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from automl.bayesian_optimization import BayesianOptimizer


class HybridAutoML:
    """
    Hybrid GA + BO AutoML framework for NLP.

    Objectives:
      1. Predictive accuracy
      2. Computational efficiency
      3. Intrinsic interpretability (explainability-by-design)

    Interpretability is measured using intrinsic, structural properties
    of models. No post-hoc explanation methods are used in optimization.
    """

    def __init__(self, X, y, gene_pool, param_space):
        self.X = X
        self.y = y
        self.gene_pool = gene_pool
        self.param_space = param_space  # The BO parameter space
        self.toolbox = base.Toolbox()

        # Three-objective optimization
        # Only create once to avoid DEAP conflicts
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self._setup_deap()

    # Core evaluation logic
    def _evaluate_individual(self, individual):
        # 1. Decode the GA individual into scikit-learn classes
        try:
            vectorizer_class = self.gene_pool[0][individual[0]]
            classifier_class = self.gene_pool[1][individual[1]]
            mode_name = self.gene_pool[2][individual[2]]  # "in_situ"/"post_hoc"/"mixed"

            print(
                f"\nEvaluating structure: {vectorizer_class.__name__} -> {classifier_class.__name__}"
            )

            pipeline_structure = [
                ("vectorizer", vectorizer_class),
                ("classifier", classifier_class),
            ]

            # 2. Create a Bayesian Optimizer for this specific structure
            bo = BayesianOptimizer(
                X=self.X,
                y=self.y,
                pipeline_steps=pipeline_structure,
                param_space=self.param_space,
            )

            # 3. Run BO to find the best hyperparameters and score
            best_params, best_accuracy, bo_variance = bo.run(n_calls=15)

            if bo_variance > 0:
                stability_score = 1.0 / (1.0 + bo_variance)
            else:
                stability_score = 0.5

            # 4. Decode the ngram_range from the returned best_params dictionary
            if "vectorizer__ngram_range" in best_params:
                ngram_str = best_params["vectorizer__ngram_range"]
                if isinstance(ngram_str, str):
                    best_params["vectorizer__ngram_range"] = tuple(
                        map(int, ngram_str.split(","))
                    )

            # 5. Train ONE final model with the best parameters found
            pipeline = bo.pipeline
            pipeline.set_params(**best_params)

            # 6. Efficiency objective
            """
            Now measuring Development efficiency, Deployment efficiency, Memory footprint
            """
            start = time.time()
            pipeline.fit(self.X, self.y)
            train_time = time.time() - start

            # Inference latency
            start = time.time()
            pipeline.predict(self.X[:50])
            inference_time = time.time() - start

            # Memory proxy: vocabulary size
            vocab_size = len(
                getattr(pipeline.named_steps["vectorizer"], "vocabulary_", {})
            )

            # Normalized efficiency score
            efficiency_score = (
                0.5 * (1.0 / (1.0 + train_time))
                + 0.3 * (1.0 / (1.0 + inference_time))
                + 0.2 * (1.0 / (1.0 + vocab_size / 10000.0))  # Normalize vocab size
            )

            interpretability_score = self.intrinsic_interpretability_score(pipeline)

            # Penalize wide n-grams for linear models (cognitive overload)
            if classifier_class == LogisticRegression:
                if best_params.get("vectorizer__ngram_range") == (1, 2):
                    interpretability_score *= 0.85

            # Penalize deep trees even if accuracy improves
            if classifier_class == DecisionTreeClassifier:
                if pipeline.named_steps["classifier"].get_depth() > 6:
                    interpretability_score *= 0.8

            if mode_name == "in_situ":
                # soft bonus: add +0.1 to interpretability_score if classifier is in interpretable set
                if classifier_class in (LogisticRegression, DecisionTreeClassifier):
                    interpretability_score = min(1.0, interpretability_score + 0.1)
                else:
                    # impose penalty
                    interpretability_score *= 0.5

            print(
                f"  --> Result: Accuracy={best_accuracy:.4f}, Efficiency={efficiency_score:.4f}, Interpretability={interpretability_score:.4f}"
            )

            return (
                best_accuracy,
                efficiency_score,
                0.8 * interpretability_score + 0.2 * stability_score,
            )

        except Exception as e:
            print(f"  Error during evaluation: {e}")
            return 0.0, 0.0, 0.0

    def intrinsic_interpretability_score(self, pipeline):
        """
        Fast, structural proxies:
          - LogisticRegression: score ~ sparsity of coef (L1)
          - DecisionTree: score ~ 1/(1 + max_depth)
          - RuleFit/EBM: pre-known high
          - Black-box (SVC, RF, Transformer): low unless constrained
        Returns 0..1
        """
        clf = pipeline.named_steps.get("classifier")

        # Linear models: coefficient sparsity
        if hasattr(clf, "coef_"):
            # sparsity: fraction of near-zero coefficients
            coef = np.ravel(clf.coef_)
            sparsity = np.mean(np.abs(coef) < 0.01)  # Count near-zero coefficients
            return (
                0.7 + 0.3 * sparsity
            )  # [0.7,1.0] favor sparse linear models ← Higher baseline

        # Decision trees: depth-based simplicity
        if hasattr(clf, "tree_"):
            depth = clf.get_depth()
            # Normalize assuming max depth ~15
            normalized_depth = min(depth / 15.0, 1.0)
            return 1.0 - 0.5 * normalized_depth  # Range: [0.5, 1.0] ← Higher

        # Other models: low interpretability by default
        return 0.1

    def _setup_deap(self):
        for i in range(len(self.gene_pool)):
            self.toolbox.register(
                f"attribute_gene_{i}", random.randint, 0, len(self.gene_pool[i]) - 1
            )

        gene_tuple = tuple(
            self.toolbox.__getattribute__(f"attribute_gene_{i}")
            for i in range(len(self.gene_pool))
        )
        self.toolbox.register(
            "individual", tools.initCycle, creator.Individual, gene_tuple, n=1
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # CRUCIAL: Register the new, powerful evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=[len(pool) - 1 for pool in self.gene_pool],
            indpb=0.2,
        )

        # NSGA-II is the right selector for multi-objective
        self.toolbox.register("select", tools.selNSGA2)

    def run(
        self,
        ngen=5,
        population_size=8,
        crossover_probability=0.5,
        mutation_probability=0.2,
    ):
        """Runs the full AutoML evolution."""
        print("🚀 Starting Hybrid AutoML Search...")

        population = self.toolbox.population(n=population_size)

        pareto_front = tools.ParetoFront()

        algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=crossover_probability,
            mutpb=mutation_probability,
            ngen=ngen,
            halloffame=pareto_front,
            verbose=True,
        )

        algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=crossover_probability,
            mutpb=mutation_probability,
            ngen=ngen,
            halloffame=pareto_front,
            verbose=True,
        )

        print("✅ AutoML search finished!")
        # return pareto_front
        results = []

        for sol in pareto_front:
            results.append(
                {
                    "vectorizer": self.gene_pool[0][sol[0]].__name__,
                    "classifier": self.gene_pool[1][sol[1]].__name__,
                    "mode": self.gene_pool[2][sol[2]],
                    "accuracy": sol.fitness.values[0],
                    "efficiency": sol.fitness.values[1],
                    "interpretability": sol.fitness.values[2],
                    "solution": sol,
                }
            )

        # Analyze diversity
        model_types = [r["classifier"] for r in results]
        print(f"\nPareto Front Diversity:")
        print(f"  Total solutions: {len(results)}")
        print(f"  SVC: {model_types.count('SVC')}")
        print(f"  LogisticRegression: {model_types.count('LogisticRegression')}")
        print(
            f"  DecisionTreeClassifier: {model_types.count('DecisionTreeClassifier')}"
        )

        return results
