from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.decision_tree import (
    DecisionTreeEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMeanDecisionTreeLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `aggrfunc_mean__decision_tree` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_mean__decision_tree",
        full_name="AggrFunc (Mean) + Decision Tree",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using mean. Base estimator: "
            "Decision Tree."
        ),
    )
    strategy_key = "aggrfunc_mean"
    base_estimator_key = "decision_tree"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_mean",
        standard_template=DecisionTreeEstimatorTemplate,
    )
