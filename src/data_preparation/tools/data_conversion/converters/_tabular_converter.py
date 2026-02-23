from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd
from beartype import beartype

from src.utils.errors import InputValidationError


@beartype
class TabularFileConverterMixin:
    """Reusable single-file and folder-wise tabular converter behaviour.

    Attributes:
        input_extension (str): Input extension.
        output_extension (str): Output extension.
        input_format_label (str): Input format label.
        output_format_label (str): Output format label.
        batch_progress_description (str): Batch progress description.

    """

    input_extension: str = ""
    output_extension: str = ""
    input_format_label: str = "input"
    output_format_label: str = "output"
    batch_progress_description: str = "Converting files..."

    @beartype
    def _convert_file(self, *, input_path: Path, output_path: Path) -> tuple[int, int]:
        """Convert one file from input format to output format."""

        data = self._read_table(input_path=input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_table(data=data, output_path=output_path)
        return len(data), int(data.shape[1])

    @beartype
    def _build_batch_output_path(
        self,
        *,
        input_folder: Path,
        output_folder: Path,
        input_file: Path,
        include_subfolders: bool,
    ) -> Path:
        """Build the output path for one batch-converted file."""

        if include_subfolders:
            relative_parent = input_file.parent.relative_to(input_folder)
            destination_parent = output_folder / relative_parent
        else:
            destination_parent = output_folder
        destination_parent.mkdir(parents=True, exist_ok=True)
        return destination_parent / f"{input_file.stem}{self.output_extension}"

    @staticmethod
    @beartype
    def _normalise_run_mode(raw_mode: str) -> str:
        """Normalise free-text mode input to supported run modes."""

        normalised = raw_mode.strip().lower().replace("-", "_")
        single_aliases = {"single", "file", "single_file", "one"}
        folder_aliases = {"folder", "batch", "directory", "dir"}
        if normalised in single_aliases:
            return "single"
        if normalised in folder_aliases:
            return "folder"
        raise InputValidationError(
            "Unsupported run mode. Choose either `single` or `folder`."
        )

    @beartype
    def _validate_input_file_path(self, path: Path) -> None:
        """Validate that the source path exists and matches the input extension."""

        if not path.exists() or not path.is_file():
            raise InputValidationError(f"Input path does not exist: {path}")
        if path.suffix.lower() != self.input_extension:
            raise InputValidationError(
                f"Input path must point to a `{self.input_extension}` file."
            )

    @staticmethod
    @beartype
    def _read_parquet(input_path: Path) -> pd.DataFrame:
        """Read a Parquet file with optional dependency handling."""

        try:
            return pd.read_parquet(input_path)
        except ImportError as exc:
            raise InputValidationError(
                "Parquet support requires `pyarrow` or `fastparquet`."
            ) from exc

    @staticmethod
    @beartype
    def _read_stata(input_path: Path) -> pd.DataFrame:
        """Read a Stata file with validation for truncated/invalid payloads."""

        try:
            file_size = input_path.stat().st_size
        except OSError as exc:
            raise InputValidationError(
                f"Could not access Stata file: {input_path.resolve()}"
            ) from exc

        if file_size == 0:
            raise InputValidationError(
                "Input Stata file is empty (0 bytes): "
                f"{input_path.resolve()}. "
                "Re-download or re-extract the source data and try again."
            )

        try:
            return pd.read_stata(input_path)
        except (OSError, ValueError, struct.error) as exc:
            raise InputValidationError(
                "Could not parse Stata file: "
                f"{input_path.resolve()}. "
                "Ensure it is a valid, non-corrupted `.dta` file."
            ) from exc

    @staticmethod
    @beartype
    def _write_parquet(*, data: pd.DataFrame, output_path: Path) -> None:
        """Write a dataframe to Parquet with optional dependency handling."""

        try:
            data.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise InputValidationError(
                "Parquet support requires `pyarrow` or `fastparquet`."
            ) from exc

    @beartype
    def _read_table(self, *, input_path: Path) -> pd.DataFrame:
        """Read input data from `input_path`."""

        raise NotImplementedError

    @beartype
    def _write_table(self, *, data: pd.DataFrame, output_path: Path) -> None:
        """Write converted dataframe to `output_path`."""

        raise NotImplementedError
