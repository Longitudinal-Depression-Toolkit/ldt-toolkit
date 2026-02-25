from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.random_forest import (
    RandomForestEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMedianRandomForestLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_median__random_forest` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_median__random_forest",
        full_name="AggrFunc (Median) + Random Forest",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using median. Base estimator: "
            "Random Forest."
        ),
    )
    strategy_key = "aggrfunc_median"
    base_estimator_key = "random_forest"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_median",
        standard_template=RandomForestEstimatorTemplate,
    )
