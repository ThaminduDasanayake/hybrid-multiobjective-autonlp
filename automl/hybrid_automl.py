import random
import numpy as np

from genetic_algorithm import GENE_POOL  # the gene definitions
from bayesian_optimization import BayesianOptimizer, PARAM_SPACE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from deap import base, creator, tools, algorithms

from lime.lime_text import LimeTextExplainer

import time


class HybridAutoML:
    def __init__(self, X, y, gene_pool, param_space):
        self.X = X
        self.y = y
        self.gene_pool = gene_pool
        self.param_space = param_space  # The BO parameter space
        self.toolbox = base.Toolbox()

        # 2 way
        # creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))

        # 3 way
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))

        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self._setup_deap()

    def _evaluate_individual(self, individual):
        # 1. Decode the GA individual into scikit-learn classes
        try:
            vectorizer_class = self.gene_pool[0][individual[0]]
            classifier_class = self.gene_pool[1][individual[1]]
            mode = individual[2]  # index of interpretability_mode gene
            mode_name = self.gene_pool[2][mode]  # "in_situ"/"post_hoc"/"mixed"

            pipeline_structure = [
                ("vectorizer", vectorizer_class),
                ("classifier", classifier_class),
            ]

            print(
                f"\nEvaluating structure: {vectorizer_class.__name__} -> {classifier_class.__name__}"
            )

            # 2. Create a Bayesian Optimizer for this specific structure
            bo_optimizer = BayesianOptimizer(
                X=self.X,
                y=self.y,
                pipeline_steps=pipeline_structure,
                param_space=self.param_space,
            )

            # 3. Run BO to find the best hyperparameters and score
            # NOTE: For speed, we use a small n_calls here. Increase for better results.
            best_params, best_accuracy, bo_variance = bo_optimizer.run(n_calls=15)

            """
            Hyperparameter stability is incorporated as a secondary signal, 
            favoring pipelines that are not only accurate but also robust and tunable.
            """
            stability_score = 1.0 / (1.0 + bo_variance)

            # 4. Decode the ngram_range from the returned best_params dictionary
            if "vectorizer__ngram_range" in best_params:
                ngram_str = best_params["vectorizer__ngram_range"]
                best_params["vectorizer__ngram_range"] = tuple(
                    map(int, ngram_str.split(","))
                )

            # 5. Train ONE final model with the best parameters found
            final_pipeline = bo_optimizer.pipeline
            final_pipeline.set_params(**best_params)
            final_pipeline.fit(self.X, self.y)

            # 6. Efficiency score
            """
            Now measuring Development efficiency, Deployment efficiency, Memory footprint
            """
            start = time.time()
            final_pipeline.fit(self.X, self.y)
            train_time = time.time() - start

            # Inference latency
            sample_texts = self.X[:50]
            start = time.time()
            _ = final_pipeline.predict(sample_texts)
            inference_time = time.time() - start

            # Memory proxy: vocabulary size
            vectorizer = final_pipeline.named_steps["vectorizer"]
            vocab_size = len(getattr(vectorizer, "vocabulary_", {}))

            # Normalized efficiency score
            efficiency_score = (
                0.5 * (1.0 / (1.0 + train_time))
                + 0.3 * (1.0 / (1.0 + inference_time))
                + 0.2 * (1.0 / (1.0 + vocab_size))
            )

            # efficiency_score = 1.0 / (1.0 + train_time)  # lower time → higher score

            # # 7. Calculate the interpretability score for this best-tuned model (Calculate fidelity)
            # interpretability_score = self.calculate_lime_fidelity(
            #     final_pipeline, self.X, num_samples=10
            # )  # Smaller num_samples for speed

            # interpretability_score = self.calculate_interpretability(
            #     final_pipeline, self.X, num_samples=10
            # )

            interpretability_score = self.calculate_interpretability(
                final_pipeline, self.X, num_samples=10, mode=mode_name
            )

            # Penalize wide n-grams for linear models (cognitive overload)
            if classifier_class == LogisticRegression:
                if "vectorizer__ngram_range" in best_params:
                    if best_params["vectorizer__ngram_range"] == (1, 2):
                        interpretability_score *= 0.85

            # Penalize deep trees even if accuracy improves
            if classifier_class == DecisionTreeClassifier:
                tree = final_pipeline.named_steps["classifier"]
                max_depth = getattr(tree, "max_depth", None)

                # fallback if depth is unconstrained
                effective_depth = (
                    max_depth if max_depth is not None else tree.get_depth()
                )

                if effective_depth > 6:
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

    # def calculate_interpretability(self, pipeline, X_train, num_samples=25):
    #     """
    #     Combines explanation fidelity and model simplicity into a single interpretability score.
    #     """
    #     fidelity = self.calculate_lime_fidelity(pipeline, X_train, num_samples)
    #
    #     # Model simplicity proxy: fewer parameters or coefficients → more interpretable
    #     try:
    #         model = pipeline.named_steps.get("classifier")
    #         if hasattr(model, "coef_"):
    #             simplicity = 1 / (1 + model.coef_.size)
    #         elif hasattr(model, "n_estimators"):
    #             simplicity = 1 / (1 + model.n_estimators)
    #         else:
    #             simplicity = 0.5  # default neutral value
    #     except Exception:
    #         simplicity = 0.5
    #
    #     # Weighted average of fidelity and simplicity
    #     interpretability_score = 0.7 * fidelity + 0.3 * simplicity
    #     return interpretability_score

    def calculate_interpretability(
        self, pipeline, X_train, num_samples=25, mode="mixed"
    ):
        """
        Interpretability is operationalized using fidelity, explanation sparsity, and
        cross-instance consistency, reflecting human cognitive constraints rather than
        purely algorithmic properties.
        """
        intrinsic = self.intrinsic_interpretability_score(pipeline)

        fidelity, sparsity, consistency = self.calculate_lime_metrics(
            pipeline, X_train, num_samples=num_samples
        )

        # Human-centered post-hoc interpretability
        posthoc = 0.5 * fidelity + 0.3 * sparsity + 0.2 * consistency

        if mode == "in_situ":
            return intrinsic

        # posthoc = self.calculate_lime_fidelity(
        #     pipeline, X_train, num_samples=num_samples
        # )

        if mode == "post_hoc":
            return posthoc

        # mixed
        return 0.6 * intrinsic + 0.4 * posthoc

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
        # logistic
        if hasattr(clf, "coef_"):
            # sparsity: fraction of near-zero coefficients
            coef = np.ravel(getattr(clf, "coef_", np.array([0.0])))
            sparsity = np.mean(np.isclose(coef, 0.0))
            return 0.5 + 0.5 * sparsity  # [0.5,1.0] favor sparse linear models
        # tree
        if hasattr(clf, "tree_"):
            max_depth = getattr(clf, "max_depth", None)
            if max_depth is None:
                max_depth = 10
            return 1.0 / (1.0 + max_depth)  # smaller depth -> larger score
        # EBM / RuleFit detection (pseudo)
        if clf.__class__.__name__.lower().startswith(
            "explainable"
        ) or clf.__class__.__name__.lower().startswith("rule"):
            return 0.9
        # black-box fallback
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
            stats=None,
            halloffame=pareto_front,
            verbose=True,
        )

        print("✅ AutoML search finished!")
        return pareto_front

    def calculate_lime_metrics(
        self, pipeline, X_train, num_samples=25, max_features=10
    ):
        """
        Computes:
          - fidelity: how well explanation approximates model
          - sparsity: fewer features = better
          - consistency: variance of explanation weights
        Returns values normalized to [0,1]
        """

        predictor = pipeline.predict_proba
        explainer = LimeTextExplainer(class_names=["sci.med", "sci.space"])

        fidelities = []
        sparsities = []
        explanation_vectors = []

        sample_indices = np.random.choice(len(X_train), num_samples, replace=False)

        for i in sample_indices:
            try:
                explanation = explainer.explain_instance(
                    X_train[i], predictor, num_features=max_features
                )

                fidelities.append(explanation.score)

                # Explanation complexity (human cognitive load)
                num_features_used = len(explanation.as_list())
                sparsity_score = 1.0 / (1.0 + num_features_used)
                sparsities.append(sparsity_score)

                # Store explanation weights for consistency
                weights = np.array([abs(w) for _, w in explanation.as_list()])
                explanation_vectors.append(weights)

            except Exception:
                continue

        if not fidelities:
            return 0.0, 0.0, 0.0

        # Consistency across explanations
        if len(explanation_vectors) > 1:
            padded = np.zeros((len(explanation_vectors), max_features))
            for i, v in enumerate(explanation_vectors):
                padded[i, : len(v)] = v
            consistency = 1.0 / (1.0 + np.var(padded))
        else:
            consistency = 0.5

        return (
            float(np.mean(fidelities)),
            float(np.mean(sparsities)),
            float(consistency),
        )

    # def calculate_lime_fidelity(self, pipeline, X_train, num_samples=25):
    #     """
    #     Calculates the average LIME fidelity for a trained pipeline.
    #
    #     Args:
    #         pipeline: The trained scikit-learn pipeline.
    #         X_train (list): The training text data.
    #         num_samples (int): The number of random samples to explain.
    #
    #     Returns:
    #         float: The average fidelity score.
    #     """
    #     print("  Calculating LIME fidelity...")
    #     # LIME needs a function that takes text and returns prediction probabilities
    #     predictor = pipeline.predict_proba
    #
    #     explainer = LimeTextExplainer(
    #         class_names=["sci.med", "sci.space"]
    #     )  # Update with your class names
    #
    #     fidelities = []
    #     # Choose random samples to explain from the training set
    #     sample_indices = np.random.choice(len(X_train), num_samples, replace=False)
    #
    #     for i in sample_indices:
    #         text_instance = X_train[i]
    #         try:
    #             # Generate the explanation
    #             explanation = explainer.explain_instance(
    #                 text_instance, predictor, num_features=10
    #             )
    #             # The 'score' attribute holds the fidelity of the local model
    #             fidelities.append(explanation.score)
    #         except Exception as e:
    #             # Sometimes an explanation can fail, we'll just skip it
    #             # print(f"    LIME explanation failed for sample {i}: {e}")
    #             pass
    #
    #     return np.mean(fidelities) if fidelities else 0.0


# -----------------------------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    # Load data
    categories = ["sci.med", "sci.space"]
    newsgroups_data = fetch_20newsgroups(
        subset="all", categories=categories, shuffle=True, random_state=42
    )
    X, y = newsgroups_data.data, newsgroups_data.target

    # A smaller subset for a quick run
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=0.5, stratify=y, random_state=42
    )

    # 1. Initialize the AutoML system
    automl = HybridAutoML(
        X=X_train, y=y_train, gene_pool=GENE_POOL, param_space=PARAM_SPACE
    )

    pareto_solutions = automl.run(ngen=4, population_size=5)

    print("\n" + "=" * 50)
    print(f"🏆 Found {len(pareto_solutions)} Optimal Trade-off Pipelines 🏆")
    print("=" * 50)

    solution_num = 1
    for solution in pareto_solutions:
        vectorizer = GENE_POOL[0][solution[0]]
        classifier = GENE_POOL[1][solution[1]]
        accuracy = solution.fitness.values[0]
        efficiency = solution.fitness.values[1]
        interpretability = solution.fitness.values[2]

        print(f"\n--- Solution #{solution_num} ---")
        print(f"  - Structure: {vectorizer.__name__} -> {classifier.__name__}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Efficiency: {efficiency:.4f}")
        print(f"  - Interpretability: {interpretability:.4f}")
        solution_num += 1

    print("\n" + "=" * 50)


# if __name__ == "__main__":
#     from user_input_module import collect_user_inputs
#
#     # 1. Collect user inputs interactively
#     X, y, vectorizer_class, classifier_class, objectives = collect_user_inputs()
#
#     # 2. Update the GENE_POOL dynamically based on user choices
#     GENE_POOL = [[vectorizer_class], [classifier_class]]
#
#     # 3. Initialize and run the AutoML system
#     automl = HybridAutoML(X=X, y=y, gene_pool=GENE_POOL, param_space=PARAM_SPACE)
#     pareto_solutions = automl.run(ngen=4, population_size=5)
#
#     print("\n" + "=" * 50)
#     print(f"🏆 Found {len(pareto_solutions)} Optimal Trade-off Pipelines 🏆")
#     print("=" * 50)
#
#     solution_num = 1
#     for solution in pareto_solutions:
#         vectorizer = GENE_POOL[0][solution[0]]
#         classifier = GENE_POOL[1][solution[1]]
#         accuracy = solution.fitness.values[0]
#         efficiency = solution.fitness.values[1]
#         interpretability = solution.fitness.values[2]
#
#         print(f"\n--- Solution #{solution_num} ---")
#         print(f"  - Structure: {vectorizer.__name__} -> {classifier.__name__}")
#         print(f"  - Accuracy: {accuracy:.4f}")
#         print(f"  - Efficiency: {efficiency:.4f}")
#         print(f"  - Interpretability: {interpretability:.4f}")
#         solution_num += 1
#
#     print("\n" + "=" * 50)


# ------------------------------------------------------------------------------------------
# 2. Run the search (Note: smaller ngen/pop_size for a demo)
# best_pipeline_structure = automl.run(ngen=4, population_size=5)

# 3. Decode and print the best pipeline found
# best_vectorizer = GENE_POOL[0][best_pipeline_structure[0]]
# best_classifier = GENE_POOL[1][best_pipeline_structure[1]]

# print("\n" + "=" * 50)
# print("🏆 Overall Best Pipeline Found 🏆")
# print(f"  - Structure: {best_vectorizer.__name__} -> {best_classifier.__name__}")
# print(f"  - Best Tuned Accuracy: {best_pipeline_structure.fitness.values[0]:.4f}")
# print("=" * 50)
