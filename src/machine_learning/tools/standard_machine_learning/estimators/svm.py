from __future__ import annotations

from sklearn.svm import SVC

from src.machine_learning.tools.templates import EstimatorTemplate
from src.utils.metadata import ComponentMetadata


class SVMEstimatorTemplate(EstimatorTemplate):
    """Template definition for Support Vector Machine classification."""

    metadata = ComponentMetadata(
        name="svm",
        full_name="Support Vector Machine",
        abstract_description=(
            "[Classification] Margin-based classifier that separates classes "
            "using a linear or kernel-transformed decision boundary."
        ),
    )
    estimator_cls = SVC
    hyperparameter_descriptions = {
        "C": "Regularisation strength for margin violations.",
        "kernel": "Kernel type (`linear`, `rbf`, `poly`, `sigmoid`, or callable).",
        "degree": "Polynomial degree when using `poly` kernel.",
        "gamma": "Kernel coefficient for `rbf`, `poly`, and `sigmoid`.",
        "class_weight": "Optional class reweighting (`None`, `balanced`, or dict).",
        "probability": "Whether to enable probability estimates (slower training).",
        "decision_function_shape": "Multiclass strategy (`ovr` or `ovo`).",
        "random_state": "Random seed used when probability estimation is enabled.",
    }
