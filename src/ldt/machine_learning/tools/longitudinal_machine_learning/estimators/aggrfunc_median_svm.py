from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.svm import (
    SVMEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMedianSVMLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `aggrfunc_median__svm` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_median__svm",
        full_name="AggrFunc (Median) + Support Vector Machine",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using median. Base estimator: "
            "Support Vector Machine."
        ),
    )
    strategy_key = "aggrfunc_median"
    base_estimator_key = "svm"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_median",
        standard_template=SVMEstimatorTemplate,
    )
