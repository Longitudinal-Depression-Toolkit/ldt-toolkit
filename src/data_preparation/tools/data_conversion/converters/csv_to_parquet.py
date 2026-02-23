from __future__ import annotations

from pathlib import Path

import pandas as pd
from beartype import beartype

from src.utils.metadata import ComponentMetadata

from ..data_conversion import Conversion
from ._tabular_converter import TabularFileConverterMixin


@beartype
class CsvToParquet(TabularFileConverterMixin, Conversion):
    """Convert CSV files (`.csv`) to Parquet (`.parquet`)."""

    metadata = ComponentMetadata(
        name="csv_to_parquet",
        full_name="CSV to Parquet",
        abstract_description=(
            "Convert one CSV file or every CSV file in a folder to Parquet."
        ),
    )
    input_extension = ".csv"
    output_extension = ".parquet"
    input_format_label = "CSV"
    output_format_label = "Parquet"
    batch_progress_description = "Converting CSV files to Parquet..."

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
        """Write dataframe as Parquet.

        Args:
            data: Dataframe to serialise.
            output_path: Destination Parquet file path.

        Returns:
            None.
        """
        self._write_parquet(data=data, output_path=output_path)
