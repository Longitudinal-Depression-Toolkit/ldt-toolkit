from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.data_preparation.catalog import (
    list_data_conversion_techniques,
    resolve_technique_with_defaults,
)
from src.data_preparation.tools.data_conversion.data_conversion import (
    discover_converters,
)
from src.utils.errors import InputValidationError


def run_data_conversion(*, technique: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run a data-conversion technique in single-file or folder mode.

    The function supports both one-file conversion and batch folder conversion.
    It resolves defaults from the catalog, validates paths, executes the
    selected converter, and returns conversion metadata.

    Args:
        technique (str): Converter key from the data-conversion catalog.
        params (Mapping[str, Any]): Conversion parameters and file paths.

    Returns:
        dict[str, Any]: Conversion summary including output paths and counts.

    Examples:
        ```python
        from ldt.data_preparation.tools.data_conversion.run import run_data_conversion
        result = run_data_conversion(
            technique="csv_to_parquet",
            params={
                "run_mode": "single",
                "input_path": "./wave_1.csv",
                "output_path": "./wave_1.parquet",
            },
        )
        ```
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="data_conversion",
        technique_id=technique,
        provided_params=dict(params),
    )

    converter = _resolve_converter(technique)
    mode = _normalise_run_mode(resolved.get("run_mode", "single"))

    if mode == "single":
        return _run_single_conversion(converter=converter, params=resolved)

    return _run_folder_conversion(converter=converter, params=resolved)


def _run_single_conversion(
    *, converter: Any, params: Mapping[str, Any]
) -> dict[str, Any]:
    input_path = Path(_as_required_string(params, "input_path")).expanduser()

    try:
        converter._validate_input_file_path(input_path)
    except Exception as exc:  # noqa: BLE001
        raise InputValidationError(str(exc)) from exc

    output_raw = params.get("output_path")
    if isinstance(output_raw, str) and output_raw.strip():
        output_path = Path(output_raw.strip()).expanduser()
    else:
        output_path = input_path.with_suffix(converter.output_extension)

    if output_path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")

    try:
        row_count, column_count = converter._convert_file(
            input_path=input_path,
            output_path=output_path,
        )
    except Exception as exc:  # noqa: BLE001
        raise InputValidationError(str(exc)) from exc

    return {
        "technique": _converter_key(converter),
        "mode": "single",
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "row_count": int(row_count),
        "column_count": int(column_count),
    }


def _run_folder_conversion(
    *, converter: Any, params: Mapping[str, Any]
) -> dict[str, Any]:
    input_folder = Path(_as_required_string(params, "input_folder")).expanduser()
    if not input_folder.exists() or not input_folder.is_dir():
        raise InputValidationError(f"Input folder does not exist: {input_folder}")

    output_raw = params.get("output_folder")
    if isinstance(output_raw, str) and output_raw.strip():
        output_folder = Path(output_raw.strip()).expanduser()
    else:
        output_folder = input_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    include_subfolders = _as_bool(params.get("include_subfolders", False))
    pattern = (
        f"**/*{converter.input_extension}"
        if include_subfolders
        else f"*{converter.input_extension}"
    )
    input_files = sorted(path for path in input_folder.glob(pattern) if path.is_file())
    if not input_files:
        raise InputValidationError(
            f"No `{converter.input_extension}` files found in folder: {input_folder.resolve()}"
        )

    converted_count = 0
    failure_messages: list[str] = []
    for input_file in input_files:
        destination = converter._build_batch_output_path(
            input_folder=input_folder,
            output_folder=output_folder,
            input_file=input_file,
            include_subfolders=include_subfolders,
        )
        try:
            converter._convert_file(
                input_path=input_file,
                output_path=destination,
            )
            converted_count += 1
        except Exception as exc:  # noqa: BLE001
            failure_messages.append(f"{input_file.resolve()}: {exc}")

    if converted_count == 0:
        raise InputValidationError(
            "No files were converted. Inspect failure details in the response."
        )

    return {
        "technique": _converter_key(converter),
        "mode": "folder",
        "input_folder": str(input_folder.resolve()),
        "output_folder": str(output_folder.resolve()),
        "total_files": len(input_files),
        "converted_files": converted_count,
        "failed_files": len(failure_messages),
        "failures": failure_messages,
    }


def _resolve_converter(technique: str) -> Any:
    converters = discover_converters()
    target = _normalise_key(technique)
    for name, converter_class in converters.items():
        if _normalise_key(name) == target:
            return converter_class()
    raise InputValidationError(f"Unknown data conversion technique: {technique}")


def _normalise_run_mode(raw_mode: Any) -> str:
    if not isinstance(raw_mode, str):
        raise InputValidationError("run_mode must be either `single` or `folder`.")
    normalised = raw_mode.strip().lower().replace("-", "_")
    if normalised in {"single", "file", "single_file", "one"}:
        return "single"
    if normalised in {"folder", "batch", "directory", "dir"}:
        return "folder"
    raise InputValidationError("run_mode must be either `single` or `folder`.")


def _as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


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


def _normalise_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


def _converter_key(converter: Any) -> str:
    metadata = getattr(converter, "metadata", None)
    if metadata is None:
        return converter.__class__.__name__
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return converter.__class__.__name__


def list_data_conversion_catalog() -> list[dict[str, Any]]:
    """Return catalog entries for data-conversion techniques.

    Returns:
        list[dict[str, Any]]: Catalog rows describing available converters.
    """

    return list_data_conversion_techniques()
