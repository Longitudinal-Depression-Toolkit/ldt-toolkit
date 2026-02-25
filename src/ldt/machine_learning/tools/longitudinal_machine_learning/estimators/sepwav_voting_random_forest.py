from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.random_forest import (
    RandomForestEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavVotingRandomForestLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_voting__random_forest` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_voting__random_forest",
        full_name="SepWav (Voting) + Random Forest",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and combine by voting. Base estimator: "
            "Random Forest."
        ),
    )
    strategy_key = "sepwav_voting"
    base_estimator_key = "random_forest"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_voting",
        standard_template=RandomForestEstimatorTemplate,
    )
