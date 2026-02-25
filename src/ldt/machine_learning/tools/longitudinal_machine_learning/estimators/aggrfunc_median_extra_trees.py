from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.extra_trees import (
    ExtraTreesEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMedianExtraTreesLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_median__extra_trees` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_median__extra_trees",
        full_name="AggrFunc (Median) + Extra Trees",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using median. Base estimator: "
            "Extra Trees."
        ),
    )
    strategy_key = "aggrfunc_median"
    base_estimator_key = "extra_trees"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_median",
        standard_template=ExtraTreesEstimatorTemplate,
    )
