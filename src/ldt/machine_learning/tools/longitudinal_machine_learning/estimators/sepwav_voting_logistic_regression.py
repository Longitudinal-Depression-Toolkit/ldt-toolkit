from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.logistic_regression import (
    LogisticRegressionEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavVotingLogisticRegressionLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_voting__logistic_regression` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_voting__logistic_regression",
        full_name="SepWav (Voting) + Logistic Regression",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and combine by voting. Base estimator: "
            "Logistic Regression."
        ),
    )
    strategy_key = "sepwav_voting"
    base_estimator_key = "logistic_regression"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_voting",
        standard_template=LogisticRegressionEstimatorTemplate,
    )
