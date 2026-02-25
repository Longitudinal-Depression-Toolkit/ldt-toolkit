from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ldt.data_preparation.catalog import (
    list_data_conversion_techniques,
    resolve_technique_with_defaults,
)
from ldt.utils.errors import InputValidationError

from .converters import (
    ConversionBatchResult,
    ConversionFileResult,
    CsvToParquet,
    CsvToStata,
    ParquetToCsv,
    ParquetToStata,
    StataToCsv,
    StataToParquet,
    TabularConverterTool,
)


def run_data_conversion(*, technique: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run one data-conversion technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call converter classes directly from `ldt.data_preparation`.

    Args:
        technique (str): Conversion technique key from the catalogue, for
            example `csv_to_parquet` or `stata_to_csv`.
        params (Mapping[str, Any]): Wrapper parameters, where supported keys
            are:
            - `run_mode` (`single` or `folder`)
            - `input_path`, `output_path` for `single`
            - `input_folder`, `output_folder` for `folder`
            - `include_subfolders` (bool)
            - `fail_fast` (bool, currently fixed to `False` by the wrapper)

    Returns:
        dict[str, Any]: Serialised conversion summary payload for the CLI.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="data_conversion",
        technique_id=technique,
        provided_params=dict(params),
    )
    converter = _create_converter(technique=technique)
    result = converter.prepare(
        run_mode=str(resolved.get("run_mode", "single")),
        input_path=_as_optional_string(resolved.get("input_path")),
        output_path=_as_optional_string(resolved.get("output_path")),
        input_folder=_as_optional_string(resolved.get("input_folder")),
        output_folder=_as_optional_string(resolved.get("output_folder")),
        include_subfolders=_as_bool(resolved.get("include_subfolders", False)),
        fail_fast=False,
    )

    if isinstance(result, ConversionFileResult):
        return {
            "technique": _converter_key(converter),
            "mode": "single",
            "input_path": str(result.input_path),
            "output_path": str(result.output_path),
            "row_count": int(result.row_count),
            "column_count": int(result.column_count),
        }

    if isinstance(result, ConversionBatchResult):
        return {
            "technique": _converter_key(converter),
            "mode": "folder",
            "input_folder": str(result.input_folder),
            "output_folder": str(result.output_folder),
            "total_files": int(result.total_files),
            "converted_files": int(result.converted_files),
            "failed_files": int(result.failed_files),
            "failures": list(result.failures),
        }

    raise InputValidationError("Unsupported conversion result payload.")


def _as_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        parsed = value.strip().lower()
        if parsed in {"1", "true", "yes", "y"}:
            return True
        if parsed in {"0", "false", "no", "n"}:
            return False
    raise InputValidationError("include_subfolders must be a boolean value.")


def _create_converter(*, technique: str) -> TabularConverterTool:
    converter_class = _resolve_converter_class(technique)
    return converter_class()


def _resolve_converter_class(technique: str) -> type[TabularConverterTool]:
    converter_class = _CONVERTER_REGISTRY.get(_normalise_key(technique))
    if converter_class is None:
        raise InputValidationError(f"Unknown data conversion technique: {technique}")
    return converter_class


def _converter_key(converter: Any) -> str:
    metadata = getattr(converter, "metadata", None)
    if metadata is None:
        return converter.__class__.__name__
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return converter.__class__.__name__


def _normalise_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


_CONVERTER_REGISTRY: dict[str, type[TabularConverterTool]] = {
    "csv_to_parquet": CsvToParquet,
    "csv_to_stata": CsvToStata,
    "parquet_to_csv": ParquetToCsv,
    "parquet_to_stata": ParquetToStata,
    "stata_to_csv": StataToCsv,
    "stata_to_parquet": StataToParquet,
}


def list_data_conversion_catalog() -> list[dict[str, Any]]:
    """Return catalogue entries for data-conversion techniques."""

    return list_data_conversion_techniques()
