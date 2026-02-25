from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.svm import (
    SVMEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusSVMLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `merwav_time_minus__svm` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__svm",
        full_name="MerWavTimeMinus + Support Vector Machine",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "Support Vector Machine."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "svm"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=SVMEstimatorTemplate,
    )
