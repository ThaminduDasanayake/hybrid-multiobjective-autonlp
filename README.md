# T-AutoNLP: Human-Centered Multi-Objective AutoML for NLP

A hybrid Genetic Algorithm + Bayesian Optimization framework for automated NLP pipeline discovery with intrinsic
interpretability.

## Features

- **Multi-objective optimization**: Balances accuracy, efficiency, and interpretability
- **Hybrid GA+BO**: Genetic algorithms for structure search, Bayesian optimization for hyperparameter tuning
- **Intrinsic interpretability**: Optimizes for explainability-by-design using structural model properties
- **Pareto front visualization**: Explore trade-offs between competing objectives
- **Knee-point selection**: Automatic identification of balanced trade-off solutions

## 🚀 Quick Start (Using UV)

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install UV (if not already installed)

https://docs.astral.sh/uv/

```bash
pip install uv
```

### Setup and Run

```bash
uv run streamlit run main.py
```