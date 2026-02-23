from __future__ import annotations

from pathlib import Path

import pandas as pd
from beartype import beartype

from src.utils.metadata import ComponentMetadata

from ..data_conversion import Conversion
from ._tabular_converter import TabularFileConverterMixin


@beartype
class ParquetToCsv(TabularFileConverterMixin, Conversion):
    """Convert Parquet files (`.parquet`) to CSV (`.csv`)."""

    metadata = ComponentMetadata(
        name="parquet_to_csv",
        full_name="Parquet to CSV",
        abstract_description=(
            "Convert one Parquet file or every Parquet file in a folder to CSV."
        ),
    )
    input_extension = ".parquet"
    output_extension = ".csv"
    input_format_label = "Parquet"
    output_format_label = "CSV"
    batch_progress_description = "Converting Parquet files to CSV..."

    @beartype
    def _read_table(self, *, input_path: Path) -> pd.DataFrame:
        """Read Parquet input data.

        Args:
            input_path: Path to the source Parquet file.

        Returns:
            pd.DataFrame: Loaded tabular data.
        """
        return self._read_parquet(input_path)

    @beartype
    def _write_table(self, *, data: pd.DataFrame, output_path: Path) -> None:
        """Write dataframe as CSV.

        Args:
            data: Dataframe to serialise.
            output_path: Destination CSV file path.

        Returns:
            None.
        """
        data.to_csv(output_path, index=False)
