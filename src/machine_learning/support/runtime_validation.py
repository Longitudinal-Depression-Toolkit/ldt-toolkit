from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.utils.errors import InputValidationError


def validate_one_vs_rest_cv_folds(
    *,
    cv_folds: int,
    validation_split: float | None,
    scenarios: tuple[Any, ...],
) -> None:
    """Validate split configuration across one-vs-rest scenarios.

    Args:
        cv_folds (int): Requested number of cross-validation folds.
        validation_split (float | None): Optional validation split fraction.
        scenarios (tuple[Any, ...]): One-vs-rest scenario payloads.
    """

    invalid_scenarios: list[str] = []
    for scenario in scenarios:
        try:
            validate_cv_split_feasibility(
                y=scenario.y_binary,
                cv_folds=cv_folds,
                validation_split=validation_split,
            )
        except InputValidationError as exc:
            invalid_scenarios.append(f"{scenario.scenario_key} ({str(exc).strip()})")
    if invalid_scenarios:
        raise InputValidationError(
            "Invalid cross-validation configuration for one-vs-rest scenarios: "
            + ", ".join(invalid_scenarios)
            + "."
        )


def validate_cv_split_feasibility(
    *,
    y: pd.Series,
    cv_folds: int,
    validation_split: float | None,
) -> None:
    """Validate split feasibility for either k-fold or shuffle split.

    Args:
        y (pd.Series): Target labels to validate.
        cv_folds (int): Requested number of cross-validation folds.
        validation_split (float | None): Optional validation split fraction.
    """

    if y.nunique(dropna=True) < 2:
        raise InputValidationError(
            "Classification requires at least two distinct target classes."
        )

    min_class_count = int(y.value_counts(dropna=False).min())
    if validation_split is None:
        if cv_folds > min_class_count:
            raise InputValidationError(
                "Cross-validation folds exceed the smallest class count "
                f"({min_class_count}). Reduce folds or rebalance data."
            )
        return

    class_count = int(y.nunique(dropna=True))
    sample_count = len(y)
    validation_sample_count = int(np.ceil(validation_split * sample_count))
    train_sample_count = sample_count - validation_sample_count
    if validation_sample_count < class_count:
        raise InputValidationError(
            "Validation split is too small for stratification "
            f"(needs at least {class_count} validation samples per split)."
        )
    if train_sample_count < class_count:
        raise InputValidationError(
            "Validation split is too large for stratification "
            f"(needs at least {class_count} training samples per split)."
        )
    if min_class_count < 2:
        raise InputValidationError(
            "Validation split requires at least 2 samples in each class."
        )


def resolve_estimator_init_kwargs(
    *,
    estimator_template: Any,
    hyperparameter_overrides: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Resolve estimator constructor kwargs used by library notebook helpers.

    Args:
        estimator_template (Any): Template exposing an `estimator_cls` constructor.
        hyperparameter_overrides (Mapping[str, Any]): User-provided parameter overrides.
        random_seed (int): Random seed for reproducibility.

    Returns:
        dict[str, Any]: Constructor arguments accepted by the estimator class.
    """

    signature = inspect.signature(estimator_template.estimator_cls.__init__)
    init_kwargs: dict[str, Any] = {}
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in hyperparameter_overrides:
            init_kwargs[parameter.name] = hyperparameter_overrides[parameter.name]
        elif parameter.name == "random_state":
            init_kwargs[parameter.name] = random_seed
    return init_kwargs


def public_estimator_module(*, estimator_template: Any) -> str:
    """Resolve a public scikit-learn import module for estimator classes.

    Args:
        estimator_template (Any): Template exposing an `estimator_cls` type.

    Returns:
        str: Public import module path.
    """

    module_name = estimator_template.estimator_cls.__module__
    if module_name.startswith("sklearn.") and "._" in module_name:
        return module_name.split("._", maxsplit=1)[0]
    return module_name


__all__ = [
    "public_estimator_module",
    "resolve_estimator_init_kwargs",
    "validate_cv_split_feasibility",
    "validate_one_vs_rest_cv_folds",
]
