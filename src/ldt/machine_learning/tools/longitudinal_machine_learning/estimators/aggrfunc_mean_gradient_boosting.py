from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.gradient_boosting import (
    GradientBoostingEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMeanGradientBoostingLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_mean__gradient_boosting` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_mean__gradient_boosting",
        full_name="AggrFunc (Mean) + Gradient Boosting",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using mean. Base estimator: "
            "Gradient Boosting."
        ),
    )
    strategy_key = "aggrfunc_mean"
    base_estimator_key = "gradient_boosting"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_mean",
        standard_template=GradientBoostingEstimatorTemplate,
    )
