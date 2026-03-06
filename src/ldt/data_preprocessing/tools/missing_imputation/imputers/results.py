from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype


@beartype
@dataclass(frozen=True)
class MissingImputationResult:
    """Structured result shared by all missing-data imputers."""

    output_path: Path
    row_count: int
    column_count: int
    missing_before: int
    missing_after: int


@beartype
@dataclass(frozen=True)
class SimpleImputationResult(MissingImputationResult):
    """Structured result for column-wise simple missing-data imputation."""

    strategy: str
    eligible_columns: tuple[str, ...]
    imputed_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...] = ()
    fill_value: Any = None


@beartype
@dataclass(frozen=True)
class MICEImputationResult(MissingImputationResult):
    """Structured result for MICE missing-data imputation."""

    iterations: int
    num_datasets: int
    dataset_index: int
