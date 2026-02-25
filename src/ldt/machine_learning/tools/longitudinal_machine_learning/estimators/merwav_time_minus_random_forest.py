from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.random_forest import (
    RandomForestEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusRandomForestLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_minus__random_forest` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__random_forest",
        full_name="MerWavTimeMinus + Random Forest",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "Random Forest."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "random_forest"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=RandomForestEstimatorTemplate,
    )
