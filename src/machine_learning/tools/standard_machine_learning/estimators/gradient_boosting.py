from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier

from src.machine_learning.tools.templates import EstimatorTemplate
from src.utils.metadata import ComponentMetadata


class GradientBoostingEstimatorTemplate(EstimatorTemplate):
    """Template definition for Gradient Boosting classification."""

    metadata = ComponentMetadata(
        name="gradient_boosting",
        full_name="Gradient Boosting",
        abstract_description=(
            "[Classification] Sequential ensemble that fits weak learners to "
            "correct residual errors of previous learners."
        ),
    )
    estimator_cls = GradientBoostingClassifier
    hyperparameter_descriptions = {
        "loss": "Loss function optimised during boosting.",
        "learning_rate": "Shrinkage applied to each weak learner contribution.",
        "n_estimators": "Number of boosting stages.",
        "subsample": "Fraction of samples used for each base learner.",
        "criterion": "Split criterion for individual regression trees.",
        "min_samples_split": "Minimum samples required to split a node.",
        "min_samples_leaf": "Minimum samples required at leaf nodes.",
        "max_depth": "Maximum depth of individual regression trees.",
        "random_state": "Random seed controlling boosting reproducibility.",
    }
