from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from beartype import beartype

from src.utils.metadata import ComponentMetadata, resolve_component_metadata

synthetic_generators_package_name = (
    "ldt.data_preparation.tools.synthetic_data_generation.generators"
)


@beartype
@dataclass(frozen=True)
class SyntheticGenerationConfig:
    """Configuration payload for synthetic longitudinal data generation.

    Use this model when a caller wants to pass one normalized payload into a
    generator implementation.

    Attributes:
        output_path (str): Path where the generated CSV should be written.
        n_samples (int): Number of synthetic participants to generate.
        n_waves (int): Number of repeated measurements per participant.
        random_state (int | None): Random seed for reproducibility.
        feature_cols (list[str]): Feature columns to synthesize at each wave.
    """

    output_path: str
    n_samples: int
    n_waves: int
    random_state: int | None
    feature_cols: list[str]


@beartype
class Synthesis:
    """Base interface for synthetic longitudinal data generators.

    Concrete generators should implement `generate` and return a long-format
    dataframe with `subject_id`, `wave`, and one or more feature columns.
    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Synthetic Generator",
    )

    def generate(self, *, config: SyntheticGenerationConfig) -> pd.DataFrame:
        """Generate synthetic data from one normalized configuration object.

        Args:
            config (SyntheticGenerationConfig): Generation configuration.

        Returns:
            pd.DataFrame: Generated long-format synthetic dataset.
        """

        raise NotImplementedError


@beartype
def discover_synthetic_generators() -> dict[str, type[Synthesis]]:
    """Discover synthetic generator classes from the generators package.

    Classes are keyed by `metadata.name`, allowing tool runners to resolve
    stable technique identifiers from catalog entries.

    Returns:
        dict[str, type[Synthesis]]: Mapping from technique key to generator class.
    """

    generators: dict[str, type[Synthesis]] = {}
    package = importlib.import_module(synthetic_generators_package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Synthesis) and obj is not Synthesis:
                label = resolve_component_metadata(obj).name
                generators[label] = obj
    return dict(sorted(generators.items(), key=lambda item: item[0].lower()))


def write_generated_csv(*, data: pd.DataFrame, output_path: str) -> Path:
    """Persist one generated dataframe to CSV and return the destination.

    Args:
        data (pd.DataFrame): Input dataset.
        output_path (str): Destination CSV path.

    Returns:
        Path: Resolved filesystem path.
    """

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return destination
