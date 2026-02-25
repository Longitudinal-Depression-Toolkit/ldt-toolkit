from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.gradient_boosting import (
    GradientBoostingEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusGradientBoostingLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_minus__gradient_boosting` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__gradient_boosting",
        full_name="MerWavTimeMinus + Gradient Boosting",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "Gradient Boosting."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "gradient_boosting"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=GradientBoostingEstimatorTemplate,
    )
