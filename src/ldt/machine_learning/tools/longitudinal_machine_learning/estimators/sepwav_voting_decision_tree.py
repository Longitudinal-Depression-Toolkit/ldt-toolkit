from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.decision_tree import (
    DecisionTreeEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavVotingDecisionTreeLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_voting__decision_tree` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_voting__decision_tree",
        full_name="SepWav (Voting) + Decision Tree",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and combine by voting. Base estimator: "
            "Decision Tree."
        ),
    )
    strategy_key = "sepwav_voting"
    base_estimator_key = "decision_tree"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_voting",
        standard_template=DecisionTreeEstimatorTemplate,
    )
