from __future__ import annotations

from pathlib import Path

import pandas as pd
from beartype import beartype

from ldt.utils.metadata import ComponentMetadata

from ._tabular_converter import TabularConverterTool


@beartype
class ParquetToStata(TabularConverterTool):
    """Convert Parquet files (`.parquet`) to Stata (`.dta`)."""

    metadata = ComponentMetadata(
        name="parquet_to_stata",
        full_name="Parquet to Stata (.dta)",
        abstract_description=(
            "Convert one Parquet file or every Parquet file in a folder to Stata."
        ),
    )
    input_extension = ".parquet"
    output_extension = ".dta"
    input_format_label = "Parquet"
    output_format_label = "Stata"
    batch_progress_description = "Converting Parquet files to Stata..."

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
        """Write dataframe as Stata (`.dta`).

        Args:
            data: Dataframe to serialise.
            output_path: Destination Stata file path.

        Returns:
            None.
        """
        data.to_stata(output_path, write_index=False)
