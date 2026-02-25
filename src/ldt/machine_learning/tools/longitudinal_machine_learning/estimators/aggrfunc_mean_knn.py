from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.knn import (
    KNNEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class AggrFuncMeanKNNLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `aggrfunc_mean__knn` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="aggrfunc_mean__knn",
        full_name="AggrFunc (Mean) + K-Nearest Neighbours",
        abstract_description=(
            "[Longitudinal] Aggregate each feature group over waves using mean. Base estimator: "
            "K-Nearest Neighbours."
        ),
    )
    strategy_key = "aggrfunc_mean"
    base_estimator_key = "knn"
    _builder = make_standard_primitive_builder(
        primitive_key="aggrfunc_mean",
        standard_template=KNNEstimatorTemplate,
    )
