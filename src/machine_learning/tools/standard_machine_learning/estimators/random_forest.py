from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from src.machine_learning.tools.templates import EstimatorTemplate
from src.utils.metadata import ComponentMetadata


class RandomForestEstimatorTemplate(EstimatorTemplate):
    """Template definition for Random Forest classification."""

    metadata = ComponentMetadata(
        name="random_forest",
        full_name="Random Forest",
        abstract_description=(
            "[Classification] Ensemble of bootstrapped decision trees whose "
            "predictions are aggregated for robust class prediction."
        ),
    )
    estimator_cls = RandomForestClassifier
    hyperparameter_descriptions = {
        "n_estimators": "Number of trees in the forest.",
        "criterion": "Split-quality criterion for each tree.",
        "max_depth": "Maximum depth of each tree.",
        "min_samples_split": "Minimum samples required to split a node.",
        "min_samples_leaf": "Minimum samples required at leaf nodes.",
        "max_features": "Feature subset strategy used per split.",
        "class_weight": "Optional class reweighting (`None`, `balanced`, or dict).",
        "random_state": "Random seed controlling bootstrap/split randomness.",
        "n_jobs": "Number of parallel workers (`None` or integer).",
    }
