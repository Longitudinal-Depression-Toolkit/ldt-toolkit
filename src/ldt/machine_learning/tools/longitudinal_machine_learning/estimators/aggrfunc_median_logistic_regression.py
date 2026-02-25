from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.logistic_regression import (
    LogisticRegressionEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMedianLogisticRegressionLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_median__logistic_regression` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_median__logistic_regression",
        full_name="AggrFunc (Median) + Logistic Regression",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using median. Base estimator: "
            "Logistic Regression."
        ),
    )
    strategy_key = "aggrfunc_median"
    base_estimator_key = "logistic_regression"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_median",
        standard_template=LogisticRegressionEstimatorTemplate,
    )
