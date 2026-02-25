from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.knn import (
    KNNEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusKNNLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `merwav_time_minus__knn` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__knn",
        full_name="MerWavTimeMinus + K-Nearest Neighbours",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "K-Nearest Neighbours."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "knn"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=KNNEstimatorTemplate,
    )
