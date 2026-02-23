from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from src.machine_learning.tools.templates import EstimatorTemplate
from src.utils.metadata import ComponentMetadata


class DecisionTreeEstimatorTemplate(EstimatorTemplate):
    """Template definition for Decision Tree classification."""

    metadata = ComponentMetadata(
        name="decision_tree",
        full_name="Decision Tree",
        abstract_description=(
            "[Classification] Non-linear tree model that recursively splits "
            "features into decision rules to predict classes."
        ),
    )
    estimator_cls = DecisionTreeClassifier
    hyperparameter_descriptions = {
        "criterion": "Function used to measure split quality.",
        "max_depth": "Maximum tree depth.",
        "min_samples_split": "Minimum samples required to split an internal node.",
        "min_samples_leaf": "Minimum samples required at a leaf node.",
        "class_weight": "Optional class reweighting (`None`, `balanced`, or dict).",
        "random_state": "Random seed controlling split randomness.",
    }
