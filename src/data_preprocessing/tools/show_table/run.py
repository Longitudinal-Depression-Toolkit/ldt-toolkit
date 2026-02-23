from __future__ import annotations

import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_bool,
    as_optional_string,
    as_required_string,
    run_with_validation,
)
from src.data_preprocessing.support.skrub_compat import import_skrub_symbol
from src.utils.errors import InputValidationError

TableReport = import_skrub_symbol("TableReport")


@dataclass(frozen=True)
class ShowTableRequest:
    """Request payload for table reporting.

    Attributes:
        input_path (Path): Input CSV path.
        output_html (Path | None): Output HTML path for the table report.
        open_browser (bool): Whether to open the generated HTML in a browser.
    """

    input_path: Path
    output_html: Path | None
    open_browser: bool


@dataclass(frozen=True)
class ShowTableResult:
    """Result payload from table report generation.

    Attributes:
        output_html (Path): Generated report HTML path.
        row_count (int): Number of rows in source dataset.
        column_count (int): Number of columns in source dataset.
        opened_browser (bool): Whether a browser-open request was performed.
    """

    output_html: Path
    row_count: int
    column_count: int
    opened_browser: bool


def run_show_table_report(request: ShowTableRequest) -> ShowTableResult:
    """Generate a browser-viewable table profile report using `skrub.TableReport`.

    The generated HTML includes column-level statistics and data quality
    summaries that are useful for quick exploratory inspection of a dataset.

    Args:
        request (ShowTableRequest): Input CSV path and report output options.

    Returns:
        ShowTableResult: Output HTML path, dataset shape summary, and whether
            the default browser was opened.
    """

    _validate_csv_path(request.input_path)

    output_html = request.output_html
    if output_html is None:
        output_html = request.input_path.with_name(
            f"{request.input_path.stem}_skrub_report.html"
        )

    data = pd.read_csv(request.input_path)
    report = TableReport(
        data,
        title=f"skrub TableReport - {request.input_path.name}",
        verbose=0,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    report.write_html(output_html)

    opened = bool(request.open_browser)
    if opened:
        webbrowser.open(output_html.resolve().as_uri())

    return ShowTableResult(
        output_html=output_html.resolve(),
        row_count=len(data),
        column_count=int(data.shape[1]),
        opened_browser=opened,
    )


def run_show_table_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run table-report generation from catalog payloads.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `show_table` | Builds an HTML `skrub` profile report for a CSV dataset. |

    Args:
        technique (str): Technique identifier in the show-table catalog.
        params (Mapping[str, Any]): Parameters mapped to `ShowTableRequest`.

    Returns:
        dict[str, Any]: Output HTML path, row/column counts, and browser-open
            flag.
    """

    return run_with_validation(
        lambda: _run_show_table_tool(technique=technique, params=params)
    )


def _run_show_table_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="show_table",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_html_raw = as_optional_string(resolved, "output_html")
    output_html = (
        Path(output_html_raw).expanduser()
        if output_html_raw
        else input_path.with_name(f"{input_path.stem}_skrub_report.html")
    )
    open_browser = as_bool(
        resolved.get("open_browser", False),
        field_name="open_browser",
    )

    result = run_show_table_report(
        ShowTableRequest(
            input_path=input_path,
            output_html=output_html,
            open_browser=open_browser,
        )
    )
    return {
        "output_html": str(result.output_html),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "opened_browser": result.opened_browser,
    }


def _validate_csv_path(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"CSV path does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"CSV path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("CSV path must point to a .csv file.")
