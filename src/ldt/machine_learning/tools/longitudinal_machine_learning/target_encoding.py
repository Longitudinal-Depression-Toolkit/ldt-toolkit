from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from beartype import beartype
from rich.console import Console
from rich.table import Table

console = Console()


@beartype
@dataclass(frozen=True)
class LongitudinalTargetEncodingResult:
    """Result payload for optional longitudinal target-label encoding.

    Attributes:
        encoded_target (pd.Series): Encoded target.
        was_encoded (bool): Whether to was encoded.
        label_mapping (tuple[tuple[int, str], ...]): Label mapping.

    """

    encoded_target: pd.Series
    was_encoded: bool
    label_mapping: tuple[tuple[int, str], ...]


@beartype
class LongitudinalTargetEncoder:
    """Encode non-numeric targets for scikit-longitudinal compatibility."""

    @staticmethod
    @beartype
    def encode_if_needed(*, y: pd.Series) -> LongitudinalTargetEncodingResult:
        """Encode target labels to integers when target dtype is non-numeric.

        Args:
            y (pd.Series): Target labels.

        Returns:
            LongitudinalTargetEncodingResult: Result object for this operation.
        """

        if pd.api.types.is_numeric_dtype(y):
            return LongitudinalTargetEncodingResult(
                encoded_target=y.copy(),
                was_encoded=False,
                label_mapping=tuple(),
            )

        encoded_values, unique_values = pd.factorize(y, sort=True)
        encoded_target = pd.Series(
            encoded_values.astype("int64"),
            index=y.index,
            name=y.name,
        )
        label_mapping = tuple(
            (index, str(label)) for index, label in enumerate(unique_values.tolist())
        )
        return LongitudinalTargetEncodingResult(
            encoded_target=encoded_target,
            was_encoded=True,
            label_mapping=label_mapping,
        )

    @staticmethod
    @beartype
    def print_encoding_notice(
        *,
        target_column: str,
        result: LongitudinalTargetEncodingResult,
    ) -> None:
        """Print target-encoding details when automatic conversion happened.

        Args:
            target_column (str): Column name for target column.
            result (LongitudinalTargetEncodingResult): Result object used by this workflow.
        """

        if not result.was_encoded:
            return

        console.print(
            "[yellow]Notice:[/yellow] "
            f"Target column `{target_column}` is non-numeric and was encoded to "
            "integer labels for longitudinal-ml compatibility."
        )
        mapping_table = Table(title="Target Label Mapping")
        mapping_table.add_column("Encoded label", style="bold cyan")
        mapping_table.add_column("Original label", style="white")
        for encoded_label, original_label in result.label_mapping:
            mapping_table.add_row(str(encoded_label), original_label)
        console.print(mapping_table)
