from __future__ import annotations

from scikit_longitudinal.estimators.trees.lexicographical.lexico_decision_tree import (
    LexicoDecisionTreeClassifier,
)

from ldt.utils.metadata import ComponentMetadata

from .base import (
    LongitudinalEstimatorTemplate,
    make_merwav_time_plus_longitudinal_builder,
)


class MerWavTimePlusLexicoDecisionTreeLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_plus__lexico_decision_tree` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_plus__lexico_decision_tree",
        full_name="MerWavTimePlus + Lexicographical Decision Tree",
        abstract_description=(
            "[Longitudinal] Preserve temporal dependency and train a "
            "longitudinal-data-aware estimator. Longitudinal algorithm: "
            "Lexicographical Decision Tree. Longitudinal tree that uses lexicographical temporal ordering."
        ),
    )
    strategy_key = "merwav_time_plus"
    base_estimator_key = "lexico_decision_tree"
    _builder = make_merwav_time_plus_longitudinal_builder(
        estimator_cls=LexicoDecisionTreeClassifier
    )
