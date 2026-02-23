from __future__ import annotations

from pathlib import Path

import pandas as pd
from beartype import beartype

from src.utils.metadata import ComponentMetadata

from ..data_conversion import Conversion
from ._tabular_converter import TabularFileConverterMixin


@beartype
class StataToParquet(TabularFileConverterMixin, Conversion):
    """Convert Stata files (`.dta`) to Parquet (`.parquet`)."""

    metadata = ComponentMetadata(
        name="stata_to_parquet",
        full_name="Stata (.dta) to Parquet",
        abstract_description=(
            "Convert one Stata file or every Stata file in a folder to Parquet."
        ),
    )
    input_extension = ".dta"
    output_extension = ".parquet"
    input_format_label = "Stata"
    output_format_label = "Parquet"
    batch_progress_description = "Converting Stata files to Parquet..."

    @beartype
    def _read_table(self, *, input_path: Path) -> pd.DataFrame:
        """Read Stata input data.

        Args:
            input_path: Path to the source Stata file.

        Returns:
            pd.DataFrame: Loaded tabular data.
        """
        return self._read_stata(input_path)

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
