from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.logistic_regression import (
    LogisticRegressionEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingDTLogisticRegressionLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_stacking_dt__logistic_regression` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_dt__logistic_regression",
        full_name="SepWav (Stacking + Decision Tree) + Logistic Regression",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Decision Tree. Base estimator: "
            "Logistic Regression."
        ),
    )
    strategy_key = "sepwav_stacking_dt"
    base_estimator_key = "logistic_regression"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_dt",
        standard_template=LogisticRegressionEstimatorTemplate,
    )
