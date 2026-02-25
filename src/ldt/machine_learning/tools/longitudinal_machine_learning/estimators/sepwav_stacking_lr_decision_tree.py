from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.decision_tree import (
    DecisionTreeEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingLRDecisionTreeLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_stacking_lr__decision_tree` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_lr__decision_tree",
        full_name="SepWav (Stacking + Logistic Regression) + Decision Tree",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Logistic Regression. Base estimator: "
            "Decision Tree."
        ),
    )
    strategy_key = "sepwav_stacking_lr"
    base_estimator_key = "decision_tree"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_lr",
        standard_template=DecisionTreeEstimatorTemplate,
    )
