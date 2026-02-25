from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.logistic_regression import (
    LogisticRegressionEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusLogisticRegressionLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_minus__logistic_regression` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__logistic_regression",
        full_name="MerWavTimeMinus + Logistic Regression",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "Logistic Regression."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "logistic_regression"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=LogisticRegressionEstimatorTemplate,
    )
