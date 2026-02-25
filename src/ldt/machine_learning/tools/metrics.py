from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from beartype import beartype
from sklearn.metrics import (
    average_precision_score,
    get_scorer,
    get_scorer_names,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

EstimatorScorer = Callable[[object, object, object], float]


class MetricCompatibilityError(RuntimeError):
    """Raised when an estimator cannot satisfy metric output requirements."""


@beartype
@dataclass(frozen=True)
class MetricDefinition:
    """Typed metadata for discoverable classification metrics.

    Attributes:
        key (str): Metric key used in configuration payloads.
        source (str): Metric provider (`custom` or `scikit-learn`).
        description (str): Human-readable metric summary.
        scorer (EstimatorScorer): Callable scorer used during evaluation.

    """

    key: str
    source: str
    description: str
    scorer: EstimatorScorer


_CUSTOM_METRICS: dict[str, MetricDefinition] = {
    "auroc": MetricDefinition(
        key="auroc",
        source="custom",
        description="Area Under the ROC Curve.",
        scorer=lambda estimator, X, y: _score_auroc(estimator=estimator, X=X, y_true=y),
    ),
    "auprc": MetricDefinition(
        key="auprc",
        source="custom",
        description="Area Under the Precision-Recall Curve.",
        scorer=lambda estimator, X, y: _score_auprc(estimator=estimator, X=X, y_true=y),
    ),
    "geometric_mean": MetricDefinition(
        key="geometric_mean",
        source="custom",
        description="Geometric mean of per-class recall values.",
        scorer=lambda estimator, X, y: _score_geometric_mean(
            estimator=estimator,
            X=X,
            y_true=y,
        ),
    ),
}

_CUSTOM_METRIC_ALIASES: dict[str, str] = {
    "auc_roc": "auroc",
    "gmean": "geometric_mean",
    "geo_mean": "geometric_mean",
}

_SKLEARN_SCORER_KEYS: dict[str, str] = {
    scorer_name.lower(): scorer_name for scorer_name in get_scorer_names()
}


@beartype
def list_supported_metrics() -> tuple[MetricDefinition, ...]:
    """Return all supported classification metrics.

    Supported metric groups:

    | Group | Keys | Notes |
    | --- | --- | --- |
    | `custom` | `auroc`, `auprc`, `geometric_mean` | Implemented in `ldt` for longitudinal workflows and estimator compatibility checks. |
    | `scikit-learn` | All keys returned by `sklearn.metrics.get_scorer_names()` | Includes standard scorers such as `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `roc_auc_ovr`, and many others. |

    Custom aliases accepted by `resolve_metric_definition(...)`:

    | Alias | Resolved key |
    | --- | --- |
    | `auc_roc` | `auroc` |
    | `gmean` | `geometric_mean` |
    | `geo_mean` | `geometric_mean` |

    Glossary:
    - **AUROC**: Area Under the Receiver Operating Characteristic Curve, measuring discrimination ability across thresholds.
    - **AUPRC**: Area Under the Precision-Recall Curve, focusing on performance with imbalanced classes.
    - **Geometric Mean of Recalls**: Aggregates per-class recall values, penalising poor performance on any class, useful for imbalanced datasets.

    Returns:
        tuple[MetricDefinition, ...]: Ordered metric definitions containing
            custom metrics followed by all scikit-learn scorers.
    """

    sklearn_definitions = tuple(
        MetricDefinition(
            key=scorer_name,
            source="scikit-learn",
            description="Scikit-learn built-in scoring metric.",
            scorer=get_scorer(scorer_name),
        )
        for scorer_name in sorted(get_scorer_names())
    )
    custom_definitions = tuple(
        _CUSTOM_METRICS[key] for key in sorted(_CUSTOM_METRICS.keys())
    )
    return custom_definitions + sklearn_definitions


@beartype
def resolve_metric_definition(metric_key: str) -> MetricDefinition:
    """Resolve a metric key into a scorer definition.

    Args:
        metric_key (str): Requested metric key or supported alias.

    Returns:
        MetricDefinition: Resolved metric definition including source,
            description, and scorer callable.

    Raises:
        KeyError: If the key does not match any custom metric, alias, or
            scikit-learn scorer name.
    """

    key_normalised = metric_key.strip().lower()
    if key_normalised in _CUSTOM_METRIC_ALIASES:
        key_normalised = _CUSTOM_METRIC_ALIASES[key_normalised]

    custom_metric = _CUSTOM_METRICS.get(key_normalised)
    if custom_metric is not None:
        return custom_metric

    scorer_key = _SKLEARN_SCORER_KEYS.get(key_normalised)
    if scorer_key is None:
        raise KeyError(f"Unknown metric: {metric_key}")
    return MetricDefinition(
        key=scorer_key,
        source="scikit-learn",
        description="Scikit-learn built-in scoring metric.",
        scorer=get_scorer(scorer_key),
    )


@beartype
def _score_auroc(*, estimator: object, X: object, y_true: object) -> float:
    """Compute AUROC for binary or multiclass classification.

    Args:
        estimator: Fitted estimator.
        X: Feature matrix.
        y_true: Ground-truth labels.

    Returns:
        float: AUROC value.
    """

    y_array = np.asarray(y_true)
    scores, estimator_classes = _extract_scores(
        estimator=estimator,
        X=X,
        require_predict_proba=True,
        metric_key="auroc",
    )
    if estimator_classes is None:
        estimator_classes = np.unique(y_array)

    if len(estimator_classes) <= 2:
        if len(estimator_classes) == 1:
            return float("nan")
        positive_label = estimator_classes[1]
        binary_target = (y_array == positive_label).astype(int)
        score_vector = np.asarray(scores).ravel()
        if score_vector.shape[0] != binary_target.shape[0]:
            return float("nan")
        return float(roc_auc_score(binary_target, score_vector))

    score_matrix = np.asarray(scores)
    if score_matrix.ndim != 2 or score_matrix.shape[1] != len(estimator_classes):
        return float("nan")
    y_binarised = label_binarize(y_array, classes=np.asarray(estimator_classes))
    return float(
        roc_auc_score(
            y_binarised,
            score_matrix,
            average="macro",
            multi_class="ovr",
        )
    )


@beartype
def _score_auprc(*, estimator: object, X: object, y_true: object) -> float:
    """Compute AUPRC for binary or multiclass classification.

    Args:
        estimator: Fitted estimator.
        X: Feature matrix.
        y_true: Ground-truth labels.

    Returns:
        float: AUPRC value.
    """

    y_array = np.asarray(y_true)
    scores, estimator_classes = _extract_scores(
        estimator=estimator,
        X=X,
        require_predict_proba=True,
        metric_key="auprc",
    )
    if estimator_classes is None:
        estimator_classes = np.unique(y_array)

    if len(estimator_classes) <= 2:
        if len(estimator_classes) == 1:
            return float("nan")
        positive_label = estimator_classes[1]
        binary_target = (y_array == positive_label).astype(int)
        score_vector = np.asarray(scores).ravel()
        if score_vector.shape[0] != binary_target.shape[0]:
            return float("nan")
        return float(average_precision_score(binary_target, score_vector))

    score_matrix = np.asarray(scores)
    if score_matrix.ndim != 2 or score_matrix.shape[1] != len(estimator_classes):
        return float("nan")
    y_binarised = label_binarize(y_array, classes=np.asarray(estimator_classes))
    return float(average_precision_score(y_binarised, score_matrix, average="macro"))


@beartype
def _score_geometric_mean(*, estimator: object, X: object, y_true: object) -> float:
    """Compute geometric mean of class recalls.

    Args:
        estimator: Fitted estimator.
        X: Feature matrix.
        y_true: Ground-truth labels.

    Returns:
        float: Geometric mean score.
    """

    y_array = np.asarray(y_true)
    y_pred = np.asarray(estimator.predict(X))
    labels = np.unique(np.concatenate((y_array, y_pred)))
    recalls = recall_score(
        y_array,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    if np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


@beartype
def _extract_scores(
    *,
    estimator: object,
    X: object,
    require_predict_proba: bool = False,
    metric_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract confidence scores from an estimator.

    Args:
        estimator: Fitted estimator.
        X: Feature matrix.
        require_predict_proba: Whether to enforce `predict_proba` support.
        metric_key: Optional metric key used for error messaging.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Score array and estimator classes.
    """

    estimator_classes = getattr(estimator, "classes_", None)

    probabilities = _extract_predict_proba_if_available(estimator=estimator, X=X)
    if require_predict_proba:
        if probabilities is None:
            metric_label = metric_key or "selected"
            raise MetricCompatibilityError(
                f"Metric `{metric_label}` requires `predict_proba`, but "
                f"`{type(estimator).__name__}` does not provide it."
            )
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1], estimator_classes
        return probabilities, estimator_classes

    if probabilities is not None:
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1], estimator_classes
        return probabilities, estimator_classes

    if hasattr(estimator, "decision_function"):
        try:
            decisions = np.asarray(estimator.decision_function(X))
        except (AttributeError, NotImplementedError):
            decisions = None
        if decisions is not None:
            if decisions.ndim == 2 and decisions.shape[1] == 2:
                return decisions[:, 1], estimator_classes
            return decisions, estimator_classes

    predictions = np.asarray(estimator.predict(X))
    return predictions, estimator_classes


@beartype
def _extract_predict_proba_if_available(
    *, estimator: object, X: object
) -> np.ndarray | None:
    """Return estimator probabilities when available and callable."""

    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        return np.asarray(estimator.predict_proba(X))
    except (AttributeError, NotImplementedError):
        return None
