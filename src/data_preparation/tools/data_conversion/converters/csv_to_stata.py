from __future__ import annotations

from pathlib import Path

import pandas as pd
from beartype import beartype

from src.utils.metadata import ComponentMetadata

from ..data_conversion import Conversion
from ._tabular_converter import TabularFileConverterMixin


@beartype
class CsvToStata(TabularFileConverterMixin, Conversion):
    """Convert CSV files (`.csv`) to Stata (`.dta`)."""

    metadata = ComponentMetadata(
        name="csv_to_stata",
        full_name="CSV to Stata (.dta)",
        abstract_description=(
            "Convert one CSV file or every CSV file in a folder to Stata."
        ),
    )
    input_extension = ".csv"
    output_extension = ".dta"
    input_format_label = "CSV"
    output_format_label = "Stata"
    batch_progress_description = "Converting CSV files to Stata..."

    @beartype
    def _read_table(self, *, input_path: Path) -> pd.DataFrame:
        """Read CSV input data.

        Args:
            input_path: Path to the source CSV file.

        Returns:
            pd.DataFrame: Loaded tabular data.
        """
        return pd.read_csv(input_path)

    @beartype
    def _write_table(self, *, data: pd.DataFrame, output_path: Path) -> None:
        """Write dataframe as Stata (`.dta`).

        Args:
            data: Dataframe to serialise.
            output_path: Destination Stata file path.

        Returns:
            None.
        """
        data.to_stata(output_path, write_index=False)
