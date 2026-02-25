from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.decision_tree import (
    DecisionTreeEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class MerWavTimeMinusDecisionTreeLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_minus__decision_tree` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_minus__decision_tree",
        full_name="MerWavTimeMinus + Decision Tree",
        abstract_description=(
            "[Longitudinal] Flatten waves and discard temporal dependency. Base estimator: "
            "Decision Tree."
        ),
    )
    strategy_key = "merwav_time_minus"
    base_estimator_key = "decision_tree"
    _builder = make_standard_primitive_builder(
        primitive_key="merwav_time_minus",
        standard_template=DecisionTreeEstimatorTemplate,
    )
