from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.knn import (
    KNNEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavVotingKNNLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `sepwav_voting__knn` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_voting__knn",
        full_name="SepWav (Voting) + K-Nearest Neighbours",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and combine by voting. Base estimator: "
            "K-Nearest Neighbours."
        ),
    )
    strategy_key = "sepwav_voting"
    base_estimator_key = "knn"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_voting",
        standard_template=KNNEstimatorTemplate,
    )
