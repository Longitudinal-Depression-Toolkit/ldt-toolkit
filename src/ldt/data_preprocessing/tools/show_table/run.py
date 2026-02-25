from __future__ import annotations

import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.data_preprocessing.support.skrub_compat import import_skrub_symbol
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

TableReport = import_skrub_symbol("TableReport")


@beartype
@dataclass(frozen=True)
class ShowTableResult:
    """Structured output from table-report generation."""

    output_html: Path
    row_count: int
    column_count: int
    opened_browser: bool


@beartype
class ShowTable(DataPreprocessingTool):
    """Generate a browser-viewable `skrub.TableReport` for a CSV dataset.

    Runtime parameters:
        - `input_path`: Input CSV path.
        - `output_html`: Optional output HTML path. Defaults to
          `<input_stem>_skrub_report.html` beside the input file.
        - `open_browser`: Whether to open the generated report in the default
          browser.

    Examples:
        ```python
        from ldt.data_preprocessing import ShowTable

        tool = ShowTable()
        result = tool.fit_preprocess(
            input_path="./data/cleaned.csv",
            output_html="./outputs/cleaned_report.html",
            open_browser=False,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="show_table",
        full_name="Show Table",
        abstract_description="Generate an HTML table profile report for a CSV dataset.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, **kwargs: Any) -> ShowTable:
        """Validate and store report-generation configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `input_path` (str | Path): Input CSV path.
                - `output_html` (str | Path | None): Output HTML path.
                - `open_browser` (bool): Whether to open the report after
                  generation.

        Returns:
            ShowTable: The fitted tool instance.
        """

        input_path_raw = kwargs.get("input_path")
        if input_path_raw is None:
            raise InputValidationError("Missing required parameter: input_path")

        input_path = Path(str(input_path_raw)).expanduser()
        _validate_csv_path(input_path)

        output_html_raw = kwargs.get("output_html")
        output_html = (
            Path(str(output_html_raw)).expanduser()
            if output_html_raw is not None
            else input_path.with_name(f"{input_path.stem}_skrub_report.html")
        )

        open_browser = _as_bool(
            kwargs.get("open_browser", False),
            field_name="open_browser",
        )

        self._config = {
            "input_path": input_path,
            "output_html": output_html,
            "open_browser": open_browser,
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> ShowTableResult:
        """Generate the HTML report and optionally open it in a browser.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            ShowTableResult: Typed summary of the generated report output.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._config is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        input_path = _as_path(self._config.get("input_path"), key="input_path")
        output_html = _as_path(self._config.get("output_html"), key="output_html")
        open_browser = _as_bool(
            self._config.get("open_browser", False),
            field_name="open_browser",
        )

        data = pd.read_csv(input_path)
        report = TableReport(
            data,
            title=f"skrub TableReport - {input_path.name}",
            verbose=0,
        )

        output_html.parent.mkdir(parents=True, exist_ok=True)
        report.write_html(output_html)

        if open_browser:
            webbrowser.open(output_html.resolve().as_uri())

        return ShowTableResult(
            output_html=output_html.resolve(),
            row_count=len(data),
            column_count=int(data.shape[1]),
            opened_browser=bool(open_browser),
        )


@beartype
def run_show_table(*, technique: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run one show-table technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `ShowTable` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Show-table technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `ShowTable.fit_preprocess(...)`: `input_path`, optional
            `output_html`, and `open_browser`.

    Returns:
        dict[str, Any]: Serialised report-generation summary for the Go CLI
        bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="show_table",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(_as_required_string(resolved, "input_path")).expanduser()
    output_html_raw = _as_optional_string(resolved.get("output_html"))
    output_html = (
        Path(output_html_raw).expanduser()
        if output_html_raw
        else input_path.with_name(f"{input_path.stem}_skrub_report.html")
    )

    result = ShowTable().fit_preprocess(
        input_path=input_path,
        output_html=output_html,
        open_browser=_as_bool(
            resolved.get("open_browser", False),
            field_name="open_browser",
        ),
    )

    return {
        "output_html": str(result.output_html),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "opened_browser": result.opened_browser,
    }


@beartype
def _as_required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    raise InputValidationError("Expected a string value.")


@beartype
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


@beartype
def _as_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise InputValidationError(f"`{field_name}` must be a boolean value.")


@beartype
def _validate_csv_path(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"CSV path does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"CSV path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("CSV path must point to a .csv file.")
