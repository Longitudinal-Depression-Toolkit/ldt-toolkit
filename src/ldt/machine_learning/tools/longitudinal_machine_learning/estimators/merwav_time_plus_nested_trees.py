from __future__ import annotations

from scikit_longitudinal.estimators.ensemble.nested_trees.nested_trees import (
    NestedTreesClassifier,
)

from ldt.utils.metadata import ComponentMetadata

from .base import (
    LongitudinalEstimatorTemplate,
    make_merwav_time_plus_longitudinal_builder,
)


class MerWavTimePlusNestedTreesLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_plus__nested_trees` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_plus__nested_trees",
        full_name="MerWavTimePlus + Nested Trees",
        abstract_description=(
            "[Longitudinal] Preserve temporal dependency and train a "
            "longitudinal-data-aware estimator. Longitudinal algorithm: "
            "Nested Trees. Hierarchical nested-tree model for longitudinal structure."
        ),
    )
    strategy_key = "merwav_time_plus"
    base_estimator_key = "nested_trees"
    _builder = make_merwav_time_plus_longitudinal_builder(
        estimator_cls=NestedTreesClassifier
    )
