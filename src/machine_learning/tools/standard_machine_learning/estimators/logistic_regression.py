from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.machine_learning.tools.templates import EstimatorTemplate
from src.utils.metadata import ComponentMetadata


class LogisticRegressionEstimatorTemplate(EstimatorTemplate):
    """Template definition for Logistic Regression classification."""

    metadata = ComponentMetadata(
        name="logistic_regression",
        full_name="Logistic Regression",
        abstract_description=(
            "[Classification] Linear probabilistic classifier that models class "
            "log-odds using a logistic function."
        ),
    )
    estimator_cls = LogisticRegression
    hyperparameter_descriptions = {
        "penalty": "Regularisation penalty used by the optimisation objective.",
        "C": "Inverse regularisation strength (smaller values = stronger regularisation).",
        "solver": "Optimisation algorithm used for fitting.",
        "max_iter": "Maximum optimisation iterations.",
        "class_weight": "Optional class reweighting (`None`, `balanced`, or dict).",
        "random_state": "Random seed used by stochastic solvers.",
    }
