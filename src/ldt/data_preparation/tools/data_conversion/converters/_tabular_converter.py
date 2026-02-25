from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.templates.tools.data_preparation import DataPreparationTool


@beartype
@dataclass(frozen=True)
class ConversionFileResult:
    """Structured result for one file conversion."""

    input_path: Path
    output_path: Path
    row_count: int
    column_count: int


@beartype
@dataclass(frozen=True)
class ConversionBatchResult:
    """Structured result for one folder conversion batch."""

    input_folder: Path
    output_folder: Path
    total_files: int
    converted_files: int
    failed_files: int
    failures: tuple[str, ...]


@beartype
class TabularConverterTool(DataPreparationTool):
    """Reusable base class for one tabular converter tool.

    Subclasses define format-specific behaviour through class attributes and
    `_read_table(...)` / `_write_table(...)`.

    Class attributes:
        input_extension (str): Source file extension handled by this converter.
        output_extension (str): Destination file extension produced.
        input_format_label (str): Human-readable input format label.
        output_format_label (str): Human-readable output format label.
        batch_progress_description (str): Batch conversion progress message.

    """

    input_extension: str = ""
    output_extension: str = ""
    input_format_label: str = "input"
    output_format_label: str = "output"
    batch_progress_description: str = "Converting files..."

    @beartype
    def prepare(self, **kwargs: Any) -> ConversionFileResult | ConversionBatchResult:
        """Run conversion in `single` or `folder` mode.

        Args:
            **kwargs (Any): Conversion runtime keys:
                - `run_mode` (str): `single` or `folder`.
                - `input_path` (str | Path): Required in `single` mode.
                - `output_path` (str | Path | None): Optional in `single` mode.
                - `input_folder` (str | Path): Required in `folder` mode.
                - `output_folder` (str | Path | None): Optional in `folder` mode.
                - `include_subfolders` (bool): Recurse into subfolders when
                  running in `folder` mode.
                - `fail_fast` (bool): Stop on first failed file in `folder`
                  mode when `True`.

        Returns:
            ConversionFileResult | ConversionBatchResult: Typed conversion
            summary for a single file or a whole folder batch.
        """

        run_mode = self._normalise_run_mode(kwargs.get("run_mode", "single"))
        if run_mode == "single":
            input_path = kwargs.get("input_path")
            if input_path is None:
                raise InputValidationError("Missing required parameter: input_path")
            return self.convert_file(
                input_path=Path(str(input_path)),
                output_path=(
                    Path(str(kwargs["output_path"]))
                    if kwargs.get("output_path") is not None
                    else None
                ),
            )

        input_folder = kwargs.get("input_folder")
        if input_folder is None:
            raise InputValidationError("Missing required parameter: input_folder")
        return self.convert_folder(
            input_folder=Path(str(input_folder)),
            output_folder=(
                Path(str(kwargs["output_folder"]))
                if kwargs.get("output_folder") is not None
                else None
            ),
            include_subfolders=bool(kwargs.get("include_subfolders", False)),
            fail_fast=bool(kwargs.get("fail_fast", False)),
        )

    @beartype
    def convert_file(
        self,
        *,
        input_path: Path | str,
        output_path: Path | str | None = None,
    ) -> ConversionFileResult:
        """Convert one file and return typed conversion metadata.

        Args:
            input_path (Path | str): Source path in `input_extension` format.
            output_path (Path | str | None): Optional destination path.

        Returns:
            ConversionFileResult: Converted file metadata.
        """

        source = Path(input_path).expanduser()
        self._validate_input_file_path(source)

        destination = (
            Path(output_path).expanduser()
            if output_path is not None
            else source.with_suffix(self.output_extension)
        )
        if destination.resolve() == source.resolve():
            raise InputValidationError("Output path must be different from input path.")

        row_count, column_count = self._convert_file(
            input_path=source,
            output_path=destination,
        )
        return ConversionFileResult(
            input_path=source.resolve(),
            output_path=destination.resolve(),
            row_count=int(row_count),
            column_count=int(column_count),
        )

    @beartype
    def convert_folder(
        self,
        *,
        input_folder: Path | str,
        output_folder: Path | str | None = None,
        include_subfolders: bool = False,
        fail_fast: bool = False,
    ) -> ConversionBatchResult:
        """Convert all matching files from one folder.

        Args:
            input_folder (Path | str): Source folder.
            output_folder (Path | str | None): Destination root folder.
            include_subfolders (bool): Whether to recurse into subfolders.
            fail_fast (bool): Raise immediately on first conversion failure.

        Returns:
            ConversionBatchResult: Batch conversion summary.
        """

        source_folder = Path(input_folder).expanduser()
        if not source_folder.exists() or not source_folder.is_dir():
            raise InputValidationError(
                f"Input folder does not exist: {source_folder.resolve()}"
            )

        destination_folder = (
            Path(output_folder).expanduser()
            if output_folder is not None
            else source_folder
        )
        destination_folder.mkdir(parents=True, exist_ok=True)

        pattern = (
            f"**/*{self.input_extension}"
            if include_subfolders
            else f"*{self.input_extension}"
        )
        input_files = sorted(
            path for path in source_folder.glob(pattern) if path.is_file()
        )
        if not input_files:
            raise InputValidationError(
                f"No `{self.input_extension}` files found in folder: "
                f"{source_folder.resolve()}"
            )

        converted_count = 0
        failure_messages: list[str] = []
        for source_file in input_files:
            destination = self._build_batch_output_path(
                input_folder=source_folder,
                output_folder=destination_folder,
                input_file=source_file,
                include_subfolders=include_subfolders,
            )
            try:
                self._convert_file(
                    input_path=source_file,
                    output_path=destination,
                )
                converted_count += 1
            except Exception as exc:  # noqa: BLE001
                message = f"{source_file.resolve()}: {exc}"
                if fail_fast:
                    raise InputValidationError(message) from exc
                failure_messages.append(message)

        if converted_count == 0:
            raise InputValidationError(
                "No files were converted. Inspect conversion failures."
            )

        return ConversionBatchResult(
            input_folder=source_folder.resolve(),
            output_folder=destination_folder.resolve(),
            total_files=len(input_files),
            converted_files=converted_count,
            failed_files=len(failure_messages),
            failures=tuple(failure_messages),
        )

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
    def _normalise_run_mode(raw_mode: Any) -> str:
        """Normalise free-text mode input to supported run modes."""

        if not isinstance(raw_mode, str):
            raise InputValidationError(
                "Unsupported run mode. Choose either `single` or `folder`."
            )
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
