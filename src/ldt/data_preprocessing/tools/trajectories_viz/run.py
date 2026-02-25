from __future__ import annotations

import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

from .techniques import TECHNIQUE_RUNNERS


@beartype
@dataclass(frozen=True)
class TrajectoriesVizResult:
    """Structured output from trajectory-visualisation generation."""

    technique: str
    output_html: Path
    row_count: int
    column_count: int
    title: str
    opened_browser: bool


@beartype
class TrajectoriesViz(DataPreprocessingTool):
    """Generate interactive visualisations for trajectory assignments.

    Each visualisation technique is implemented in its own module under
    `tools/trajectories_viz/techniques`.

    Supported techniques:
        - `mean_profiles`
        - `class_spaghetti`
        - `trajectory_sizes`

    Runtime parameters:
        - `technique`: One of the supported technique keys. (See above.)
        - `input_path`: Input CSV path.
        - `output_html`: Output HTML path.
        - `title`: Figure title.
        - `open_browser`: Whether to open the output in the default browser.
        - additional technique-specific keys are passed to the selected
          technique runner through `params`.
    """

    metadata = ComponentMetadata(
        name="trajectories_viz",
        full_name="Trajectories Visualisation",
        abstract_description="Generate interactive trajectory visualisations in HTML.",
    )

    def __init__(self) -> None:
        self._technique: str | None = None
        self._input_path: Path | None = None
        self._output_html: Path | None = None
        self._title: str | None = None
        self._open_browser: bool = False
        self._technique_params: dict[str, Any] = {}

    @beartype
    def fit(self, **kwargs: Any) -> TrajectoriesViz:
        """Validate and store trajectory-visualisation configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `technique` (str): Visualisation technique key. Which are: `mean_profiles`, `class_spaghetti`, `trajectory_sizes`.
                - `input_path` (str | Path): Input CSV path.
                - `output_html` (str | Path): Output HTML path.
                - `title` (str): Chart title.
                - `open_browser` (bool): Open output in browser.
                - any additional keys are forwarded to the selected technique
                  runner as technique parameters.

        Returns:
            TrajectoriesViz: The fitted tool instance.
        """

        technique = _normalise_technique(kwargs.get("technique"))

        input_path = _as_path(kwargs.get("input_path"), key="input_path")
        output_html = _as_path(kwargs.get("output_html"), key="output_html")
        title = _as_required_string(kwargs.get("title"), field_name="title")

        _validate_csv_path(input_path)
        _validate_html_path(output_html)

        self._technique = technique
        self._input_path = input_path
        self._output_html = output_html
        self._title = title
        self._open_browser = _as_bool(
            kwargs.get("open_browser", False),
            field_name="open_browser",
        )
        self._technique_params = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "technique",
                "input_path",
                "output_html",
                "title",
                "open_browser",
            }
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> TrajectoriesVizResult:
        """Build the configured visualisation and write HTML output.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            TrajectoriesVizResult: Typed summary of generated visual output.
        """

        if kwargs:
            self.fit(**kwargs)
        if (
            self._technique is None
            or self._input_path is None
            or self._output_html is None
            or self._title is None
        ):
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        data = pd.read_csv(self._input_path)
        runner = TECHNIQUE_RUNNERS[self._technique]
        figure = runner(
            data=data,
            title=self._title,
            params=dict(self._technique_params),
        )

        self._output_html.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(
            str(self._output_html), include_plotlyjs="cdn", full_html=True
        )
        if self._open_browser:
            webbrowser.open(self._output_html.resolve().as_uri())

        return TrajectoriesVizResult(
            technique=self._technique,
            output_html=self._output_html.resolve(),
            row_count=int(len(data)),
            column_count=int(data.shape[1]),
            title=self._title,
            opened_browser=bool(self._open_browser),
        )


@beartype
def run_trajectories_viz(
    *, technique: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one trajectory-visualisation technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `TrajectoriesViz` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Visualisation technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `TrajectoriesViz.fit_preprocess(...)`, including shared keys
            (`input_path`, `output_html`, `title`, `open_browser`) plus
            technique-specific parameters.

    Returns:
        dict[str, Any]: Serialised visualisation summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="trajectories_viz",
        technique_id=technique,
        provided_params=dict(params),
    )

    payload = dict(resolved)
    payload["technique"] = technique

    result = TrajectoriesViz().fit_preprocess(**payload)
    return {
        "technique": result.technique,
        "output_html": str(result.output_html),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "title": result.title,
        "opened_browser": result.opened_browser,
    }


@beartype
def _normalise_technique(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise InputValidationError("Missing required visualisation technique.")

    token = raw.strip().lower().replace("-", "_")
    if token not in TECHNIQUE_RUNNERS:
        allowed = ", ".join(sorted(TECHNIQUE_RUNNERS.keys()))
        raise InputValidationError(
            f"Unsupported trajectories-viz technique: {raw}. Supported: {allowed}"
        )
    return token


@beartype
def _as_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {field_name}")
    return value.strip()


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
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


@beartype
def _validate_csv_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"Input CSV path does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Input path must point to a .csv file.")


@beartype
def _validate_html_path(path: Path) -> None:
    if path.suffix.lower() != ".html":
        raise InputValidationError("output_html must point to a .html file.")
