from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

from ldt.machine_learning.tools.templates import EstimatorTemplate
from ldt.utils.metadata import ComponentMetadata


class ExtraTreesEstimatorTemplate(EstimatorTemplate):
    """Template definition for Extra Trees classification."""

    metadata = ComponentMetadata(
        name="extra_trees",
        full_name="Extra Trees",
        abstract_description=(
            "[Classification] Tree ensemble like Random Forest, but with more "
            "randomised split thresholds to reduce variance."
        ),
    )
    estimator_cls = ExtraTreesClassifier
    hyperparameter_descriptions = {
        "n_estimators": "Number of trees in the ensemble.",
        "criterion": "Split-quality criterion.",
        "max_depth": "Maximum depth per tree.",
        "min_samples_split": "Minimum samples required to split a node.",
        "min_samples_leaf": "Minimum samples required at leaf nodes.",
        "max_features": "Feature subset strategy used per split.",
        "class_weight": "Optional class reweighting (`None`, `balanced`, or dict).",
        "random_state": "Random seed controlling tree/split randomness.",
        "n_jobs": "Number of parallel workers (`None` or integer).",
    }
