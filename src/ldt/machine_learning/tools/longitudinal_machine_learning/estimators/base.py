from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from beartype import beartype
from scikit_longitudinal.data_preparation import (
    AggrFunc,
    MerWavTimeMinus,
    MerWavTimePlus,
    SepWav,
)
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest import (
    LexicoDeepForestClassifier,
    LongitudinalClassifierType,
    LongitudinalEstimatorConfig,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
)
from scikit_longitudinal.pipeline import LongitudinalPipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ldt.machine_learning.tools.templates import EstimatorTemplate
from ldt.utils.metadata import ComponentMetadata

EstimatorBuilder = Callable[
    [int, tuple[tuple[int, ...], ...], tuple[int, ...], tuple[str, ...]],
    BaseEstimator,
]


@beartype
@dataclass(frozen=True)
class LongitudinalStrategyDefinition:
    """Declarative description of one longitudinal modelling strategy."""

    key: str
    full_name: str
    description: str
    uses_standard_base_estimator: bool


_LONGITUDINAL_STRATEGIES: tuple[LongitudinalStrategyDefinition, ...] = (
    LongitudinalStrategyDefinition(
        key="merwav_time_minus",
        full_name="MerWavTimeMinus",
        description="Flatten waves and discard temporal dependency.",
        uses_standard_base_estimator=True,
    ),
    LongitudinalStrategyDefinition(
        key="aggrfunc_mean",
        full_name="AggrFunc (Mean)",
        description="Aggregate each feature group over waves using mean.",
        uses_standard_base_estimator=True,
    ),
    LongitudinalStrategyDefinition(
        key="aggrfunc_median",
        full_name="AggrFunc (Median)",
        description="Aggregate each feature group over waves using median.",
        uses_standard_base_estimator=True,
    ),
    LongitudinalStrategyDefinition(
        key="sepwav_voting",
        full_name="SepWav (Voting)",
        description="Train one base estimator per wave and combine by voting.",
        uses_standard_base_estimator=True,
    ),
    LongitudinalStrategyDefinition(
        key="sepwav_stacking_lr",
        full_name="SepWav (Stacking + Logistic Regression)",
        description="Train one base estimator per wave and stack with Logistic Regression.",
        uses_standard_base_estimator=True,
    ),
    LongitudinalStrategyDefinition(
        key="sepwav_stacking_dt",
        full_name="SepWav (Stacking + Decision Tree)",
        description="Train one base estimator per wave and stack with Decision Tree.",
        uses_standard_base_estimator=True,
    ),
    LongitudinalStrategyDefinition(
        key="merwav_time_plus",
        full_name="MerWavTimePlus",
        description="Preserve temporal dependency and train a longitudinal-data-aware estimator.",
        uses_standard_base_estimator=False,
    ),
)


@beartype
def list_longitudinal_strategies() -> tuple[LongitudinalStrategyDefinition, ...]:
    """Return all supported longitudinal modelling strategies."""

    return _LONGITUDINAL_STRATEGIES


@beartype
class LongitudinalEstimatorTemplate:
    """Reusable template describing one discoverable longitudinal estimator."""

    metadata = ComponentMetadata(name="base", full_name="Longitudinal Estimator")
    strategy_key: str = ""
    base_estimator_key: str = ""
    _builder: EstimatorBuilder | None = None

    @classmethod
    @beartype
    def build_estimator(
        cls,
        *,
        random_seed: int,
        feature_groups: tuple[tuple[int, ...], ...],
        non_longitudinal_features: tuple[int, ...],
        feature_list_names: tuple[str, ...],
    ) -> BaseEstimator:
        """Build one configured longitudinal estimator pipeline."""

        builder = cls._builder
        if builder is None:
            raise ValueError(f"{cls.__name__} does not define a builder.")
        return builder(
            random_seed,
            feature_groups,
            non_longitudinal_features,
            feature_list_names,
        )


@beartype
def make_standard_primitive_builder(
    *,
    primitive_key: str,
    standard_template: type[EstimatorTemplate],
) -> EstimatorBuilder:
    """Build one callable creating a primitive + standard-estimator pipeline."""

    @beartype
    def _builder(
        random_seed: int,
        feature_groups: tuple[tuple[int, ...], ...],
        non_longitudinal_features: tuple[int, ...],
        feature_list_names: tuple[str, ...],
    ) -> BaseEstimator:
        primary_estimator = standard_template.build_estimator(
            hyperparameters={},
            random_seed=random_seed,
        )

        if primitive_key == "merwav_time_minus":
            preprocessor = MerWavTimeMinus()
            return _build_pipeline(
                step_name="MerWavTimeMinus",
                preprocessor=preprocessor,
                final_estimator=primary_estimator,
                feature_groups=feature_groups,
                non_longitudinal_features=non_longitudinal_features,
                feature_list_names=feature_list_names,
            )

        if primitive_key == "aggrfunc_mean":
            preprocessor = AggrFunc(aggregation_func="mean")
            return _build_pipeline(
                step_name="AggrFunc",
                preprocessor=preprocessor,
                final_estimator=primary_estimator,
                feature_groups=feature_groups,
                non_longitudinal_features=non_longitudinal_features,
                feature_list_names=feature_list_names,
            )

        if primitive_key == "aggrfunc_median":
            preprocessor = AggrFunc(aggregation_func="median")
            return _build_pipeline(
                step_name="AggrFunc",
                preprocessor=preprocessor,
                final_estimator=primary_estimator,
                feature_groups=feature_groups,
                non_longitudinal_features=non_longitudinal_features,
                feature_list_names=feature_list_names,
            )

        if primitive_key == "sepwav_voting":
            preprocessor = SepWav(
                voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
            )
            return _build_pipeline(
                step_name="SepWav",
                preprocessor=preprocessor,
                final_estimator=primary_estimator,
                feature_groups=feature_groups,
                non_longitudinal_features=non_longitudinal_features,
                feature_list_names=feature_list_names,
            )

        if primitive_key == "sepwav_stacking_lr":
            preprocessor = SepWav(
                voting=LongitudinalEnsemblingStrategy.STACKING,
                stacking_meta_learner=LogisticRegression(random_state=random_seed),
            )
            return _build_pipeline(
                step_name="SepWav",
                preprocessor=preprocessor,
                final_estimator=primary_estimator,
                feature_groups=feature_groups,
                non_longitudinal_features=non_longitudinal_features,
                feature_list_names=feature_list_names,
            )

        if primitive_key == "sepwav_stacking_dt":
            preprocessor = SepWav(
                voting=LongitudinalEnsemblingStrategy.STACKING,
                stacking_meta_learner=DecisionTreeClassifier(random_state=random_seed),
            )
            return _build_pipeline(
                step_name="SepWav",
                preprocessor=preprocessor,
                final_estimator=primary_estimator,
                feature_groups=feature_groups,
                non_longitudinal_features=non_longitudinal_features,
                feature_list_names=feature_list_names,
            )

        raise ValueError(f"Unsupported primitive key: {primitive_key}")

    return _builder


@beartype
def make_merwav_time_plus_longitudinal_builder(
    *,
    estimator_cls: type[BaseEstimator],
) -> EstimatorBuilder:
    """Build one callable creating MerWavTimePlus + longitudinal-estimator."""

    @beartype
    def _builder(
        random_seed: int,
        feature_groups: tuple[tuple[int, ...], ...],
        non_longitudinal_features: tuple[int, ...],
        feature_list_names: tuple[str, ...],
    ) -> BaseEstimator:
        final_estimator = _instantiate_with_random_state(
            estimator_cls=estimator_cls,
            random_seed=random_seed,
        )
        preprocessor = MerWavTimePlus()
        return _build_pipeline(
            step_name="MerWavTimePlus",
            preprocessor=preprocessor,
            final_estimator=final_estimator,
            feature_groups=feature_groups,
            non_longitudinal_features=non_longitudinal_features,
            feature_list_names=feature_list_names,
        )

    return _builder


@beartype
def _build_pipeline(
    *,
    step_name: str,
    preprocessor: object,
    final_estimator: BaseEstimator,
    feature_groups: tuple[tuple[int, ...], ...],
    non_longitudinal_features: tuple[int, ...],
    feature_list_names: tuple[str, ...],
) -> LongitudinalPipeline:
    """Construct a configured scikit-longitudinal pipeline."""

    return LongitudinalPipeline(
        steps=[(step_name, preprocessor), ("classifier", final_estimator)],
        features_group=[list(group) for group in feature_groups],
        non_longitudinal_features=list(non_longitudinal_features),
        feature_list_names=list(feature_list_names),
        update_feature_groups_callback="default",
    )


@beartype
def _instantiate_with_random_state(
    *,
    estimator_cls: type[BaseEstimator],
    random_seed: int,
) -> BaseEstimator:
    """Instantiate estimator while forwarding random_state when supported."""

    signature = inspect.signature(estimator_cls.__init__)
    init_kwargs = {}
    if "random_state" in signature.parameters:
        init_kwargs["random_state"] = random_seed
    if (
        estimator_cls is LexicoDeepForestClassifier
        and "longitudinal_base_estimators" in signature.parameters
    ):
        init_kwargs["longitudinal_base_estimators"] = (
            _default_lexico_deep_forest_base_estimators()
        )
    return estimator_cls(**init_kwargs)


@beartype
def _default_lexico_deep_forest_base_estimators() -> list[LongitudinalEstimatorConfig]:
    """Return default longitudinal base estimators for LexicoDeepForest."""

    return [
        LongitudinalEstimatorConfig(
            classifier_type=LongitudinalClassifierType.LEXICO_RF,
            count=2,
        )
    ]
