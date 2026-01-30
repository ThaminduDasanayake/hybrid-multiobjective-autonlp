import random
import numpy as np

import ssl
import urllib.request

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
# ===================================================

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from deap import base, creator, tools, algorithms

# -----------------------------------------------------------------------------
# 1. DEFINE THE SEARCH SPACE ("GENES")
# -----------------------------------------------------------------------------
# This dictionary defines all the possible algorithms to choose from.
# The GA will select one item from each list.
SEARCH_SPACE = {
    "vectorizer": [TfidfVectorizer, CountVectorizer],
    "classifier": [LogisticRegression, DecisionTreeClassifier, SVC],
}

# For DEAP, convert to a list of lists of choices to create "genes".
GENE_POOL = [
    SEARCH_SPACE["vectorizer"],
    SEARCH_SPACE["classifier"],
]


class GeneticAlgorithm:
    def __init__(self, X, y, gene_pool):
        self.X = X
        self.y = y
        self.gene_pool = gene_pool
        self.toolbox = base.Toolbox()

        # This sets up the fitness and individual "blueprints" for DEAP.
        # To MAXIMIZE accuracy, use weights=(1.0,).
        # Only create if it doesn't already exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMax)

        self._setup_deap()

    def _evaluate_pipeline(self, individual):
        # 1. Decode the individual's gene into actual algorithms
        try:
            vectorizer_gene = self.gene_pool[0][individual[0]]
            classifier_gene = self.gene_pool[1][individual[1]]

            # 2. Build the scikit-learn pipeline
            # We instantiate the classes here. We are NOT tuning hyperparameters yet.
            vectorizer_instance = vectorizer_gene()

            # Give different models different parameters if needed
            if classifier_gene == SVC:
                classifier_instance = classifier_gene(
                    max_iter=5000,
                )  # Give SVC more iterations
            elif classifier_gene == LogisticRegression:
                classifier_instance = classifier_gene(max_iter=1000)
            else:
                # DecisionTreeClassifier doesn't have max_iter
                classifier_instance = classifier_gene()

            pipeline_steps = [
                ("vectorizer", vectorizer_instance),
                ("classifier", classifier_instance),
            ]

            # pipeline_steps = [
            #     ("vectorizer", vectorizer_gene()),
            #     (
            #         "classifier",
            #         classifier_gene(max_iter=1000),
            #     ),  # Added max_iter for solvers
            # ]
            pipeline = Pipeline(pipeline_steps)

            # 3. Evaluate the pipeline using cross-validation
            # This gives a robust estimate of performance.
            score = np.mean(
                cross_val_score(pipeline, self.X, self.y, cv=3, scoring="accuracy")
            )

            print(
                f"Testing pipeline: {pipeline_steps[0][1].__class__.__name__} -> {pipeline_steps[1][1].__class__.__name__} | Accuracy: {score:.4f}"
            )

            return (score,)  # DEAP requires the fitness to be a tuple
        except Exception as e:
            # If a pipeline fails for any reason, give it the worst possible score.
            print(f"Error evaluating pipeline: {e}")
            return (0.0,)

    def _setup_deap(self):
        for i in range(len(self.gene_pool)):
            self.toolbox.register(
                f"attribute_gene_{i}", random.randint, 0, len(self.gene_pool[i]) - 1
            )

        # "individual" creates a full individual (chromosome) by combining genes.
        gene_tuple = tuple(
            self.toolbox.__getattribute__(f"attribute_gene_{i}")
            for i in range(len(self.gene_pool))
        )
        self.toolbox.register(
            "individual", tools.initCycle, creator.Individual, gene_tuple, n=1
        )

        # "population" creates a list of individuals
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # --- Genetic Operators ---
        self.toolbox.register("evaluate", self._evaluate_pipeline)
        self.toolbox.register("mate", tools.cxTwoPoint)  # Crossover
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=[len(pool) - 1 for pool in self.gene_pool],
            indpb=0.2,
        )  # Mutation
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

    def run(self, ngen=10, pop_size=10, cxpb=0.5, mutpb=0.2):
        print("🚀 Starting Genetic Algorithm Evolution...")

        # Create the initial population
        population = self.toolbox.population(n=pop_size)

        # Keep track of the best individual found
        hof = tools.HallOfFame(1)

        # Run the evolution using a standard DEAP algorithm
        algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=cxpb,  # Crossover probability
            mutpb=mutpb,  # Mutation probability
            ngen=ngen,  # Number of generations
            halloffame=hof,
            verbose=True,
        )

        print("✅ Evolution finished!")

        # Return the single best individual from the Hall of Fame
        return hof[0]
