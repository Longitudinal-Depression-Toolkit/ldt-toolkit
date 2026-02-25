from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from ldt.machine_learning.tools.templates import EstimatorTemplate
from ldt.utils.metadata import ComponentMetadata


class KNNEstimatorTemplate(EstimatorTemplate):
    """Template definition for K-Nearest Neighbours classification."""

    metadata = ComponentMetadata(
        name="knn",
        full_name="K-Nearest Neighbours",
        abstract_description=(
            "[Classification] Instance-based model that predicts a class from "
            "the majority label among nearest training neighbours."
        ),
    )
    estimator_cls = KNeighborsClassifier
    hyperparameter_descriptions = {
        "n_neighbors": "Number of neighbours considered for prediction.",
        "weights": "Neighbour weighting strategy (`uniform` or `distance`).",
        "algorithm": "Nearest-neighbour search algorithm.",
        "leaf_size": "Leaf size used by tree-based neighbour search.",
        "p": "Power parameter for Minkowski distance.",
        "metric": "Distance metric used for neighbour computation.",
        "n_jobs": "Number of parallel workers (`None` or integer).",
    }
