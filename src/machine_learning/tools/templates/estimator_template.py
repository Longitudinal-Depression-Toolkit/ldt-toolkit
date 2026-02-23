from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from beartype import beartype
from sklearn.base import BaseEstimator

from src.utils.metadata import ComponentMetadata


@beartype
@dataclass(frozen=True)
class EstimatorHyperparameter:
    """Typed hyperparameter metadata for estimator templates.

    Attributes:
        name (str): Name used for name.
        description (str): Human-readable description.
        default (Any): Default.
        required (bool): Whether to required.

    """

    name: str
    description: str
    default: Any
    required: bool


@beartype
class EstimatorTemplate:
    """Reusable template describing a discoverable ML estimator.

    Attributes:
        estimator_cls (type[BaseEstimator]): Estimator cls.
        hyperparameter_descriptions (dict[str, str]): Hyperparameter descriptions.

    """

    metadata = ComponentMetadata(name="base", full_name="Estimator")
    estimator_cls: type[BaseEstimator] = BaseEstimator
    hyperparameter_descriptions: dict[str, str] = {}

    @classmethod
    @beartype
    def list_hyperparameters(cls) -> tuple[EstimatorHyperparameter, ...]:
        """Return discoverable constructor hyperparameters.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Returns:
            tuple[EstimatorHyperparameter, ...]: Tuple of resolved values.
        """

        signature = inspect.signature(cls.estimator_cls.__init__)
        specs: list[EstimatorHyperparameter] = []
        for parameter in list(signature.parameters.values())[1:]:
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            default_value = parameter.default
            required = default_value is inspect.Parameter.empty
            description = cls.hyperparameter_descriptions.get(
                parameter.name,
                "Scikit-learn constructor hyperparameter.",
            )
            specs.append(
                EstimatorHyperparameter(
                    name=parameter.name,
                    description=description,
                    default=None if required else default_value,
                    required=required,
                )
            )
        return tuple(specs)

    @classmethod
    @beartype
    def build_estimator(
        cls,
        *,
        hyperparameters: Mapping[str, Any],
        random_seed: int,
    ) -> BaseEstimator:
        """Build the estimator instance from hyperparameters.

        Args:
            hyperparameters (Mapping[str, Any]): Hyperparameters.
            random_seed (int): Random seed for reproducibility.

        Returns:
            BaseEstimator: Instantiated estimator object.
        """

        init_kwargs = dict(hyperparameters)
        signature = inspect.signature(cls.estimator_cls.__init__)
        if "random_state" in signature.parameters and "random_state" not in init_kwargs:
            init_kwargs["random_state"] = random_seed
        return cls.estimator_cls(**init_kwargs)

    @classmethod
    @beartype
    def supports_hyperparameter(cls, hyperparameter: str) -> bool:
        """Check whether estimator exposes a constructor hyperparameter.

        Args:
            hyperparameter (str): Hyperparameter.

        Returns:
            bool: Parsed boolean value.
        """

        signature = inspect.signature(cls.estimator_cls.__init__)
        return hyperparameter in signature.parameters

    @classmethod
    @beartype
    def resolve_effective_hyperparameter_value(
        cls,
        *,
        hyperparameter: str,
        overrides: Mapping[str, Any],
        random_seed: int,
    ) -> Any:
        """Resolve the final constructor value for one hyperparameter.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            hyperparameter (str): Hyperparameter.
            overrides (Mapping[str, Any]): Overrides.
            random_seed (int): Random seed for reproducibility.

        Returns:
            Any: Imported symbol or computed value.
        """

        if hyperparameter in overrides:
            return overrides[hyperparameter]

        if hyperparameter == "random_state" and cls.supports_hyperparameter(
            "random_state"
        ):
            return random_seed

        for spec in cls.list_hyperparameters():
            if spec.name == hyperparameter:
                return spec.default
        return None
