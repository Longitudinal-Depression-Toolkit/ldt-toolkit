from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.decision_tree import (
    DecisionTreeEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMedianDecisionTreeLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_median__decision_tree` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_median__decision_tree",
        full_name="AggrFunc (Median) + Decision Tree",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using median. Base estimator: "
            "Decision Tree."
        ),
    )
    strategy_key = "aggrfunc_median"
    base_estimator_key = "decision_tree"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_median",
        standard_template=DecisionTreeEstimatorTemplate,
    )
