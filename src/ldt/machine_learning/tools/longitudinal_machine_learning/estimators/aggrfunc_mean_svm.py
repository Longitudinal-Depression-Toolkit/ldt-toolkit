from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.svm import (
    SVMEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMeanSVMLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `aggrfunc_mean__svm` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_mean__svm",
        full_name="AggrFunc (Mean) + Support Vector Machine",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using mean. Base estimator: "
            "Support Vector Machine."
        ),
    )
    strategy_key = "aggrfunc_mean"
    base_estimator_key = "svm"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_mean",
        standard_template=SVMEstimatorTemplate,
    )
