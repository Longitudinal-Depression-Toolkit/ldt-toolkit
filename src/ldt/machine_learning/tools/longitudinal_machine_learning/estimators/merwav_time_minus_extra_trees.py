from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.extra_trees import (
    ExtraTreesEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusExtraTreesLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_minus__extra_trees` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__extra_trees",
        full_name="MerWavTimeMinus + Extra Trees",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "Extra Trees."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "extra_trees"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=ExtraTreesEstimatorTemplate,
    )
