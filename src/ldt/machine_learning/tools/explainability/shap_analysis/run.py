from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.machine_learning.catalog import resolve_technique_with_defaults
from ldt.machine_learning.support.inputs import (
    as_optional_string,
    as_required_int,
    as_required_string,
    parse_class_index,
    run_with_validation,
)
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.machine_learning import MachineLearningTool

from .service import SHAPAnalysisRunner


@beartype
class SHAPAnalysis(MachineLearningTool):
    """Run SHAP explainability analysis for trained classification models.

    SHAP (SHapley Additive exPlanations) attributes each prediction to feature
    contributions using Shapley-value principles from cooperative game theory.
    This tool computes local and global explanations from a saved model and
    dataset, then exports artefacts for inspection.

    Supported model prediction interfaces:

    | Interface | What it provides |
    | --- | --- |
    | `predict_proba` | Class probability outputs. Preferred when available. |
    | `decision_function` | Continuous decision scores for classification. |
    | `predict` | Label predictions when probability/score APIs are unavailable. |

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `run_analysis` | Loads a saved model, computes SHAP values, and writes explainability artefacts. |

    Output artefacts written by the analysis:

    | Artefact | Description |
    | --- | --- |
    | `shap_analysis_report.txt` | Text summary of the analysis run and configuration. |
    | `shap_feature_importance.csv` | Global feature-importance ranking from absolute SHAP magnitudes. |
    | `shap_summary_dot.png` | SHAP summary dot plot (distribution of signed contributions). |
    | `shap_summary_bar.png` | SHAP summary bar plot (mean absolute contribution per feature). |

    Examples:
        ```python
        from ldt.machine_learning import SHAPAnalysis

        tool = SHAPAnalysis()
        result = tool.fit_predict(
            technique="run_analysis",
            model_path="./outputs/standard_ml/random_forest/random_forest_20260101_120000_model.pkl",
            input_path="./data/millennium.csv",
            target_column="depression_status",
            feature_columns="sleep_score,anxiety_score,sex,income",
            max_rows_to_explain=500,
            background_rows=200,
            class_index="auto",
        )
        ```
    """

    metadata = ComponentMetadata(
        name="shap_analysis",
        full_name="SHAP Analysis",
        abstract_description=(
            "Run SHAP explainability analysis and export reproducible interpretation artefacts."
        ),
    )

    def __init__(self) -> None:
        self._technique: str | None = None
        self._params: dict[str, Any] = {}

    @beartype
    def fit(self, **kwargs: Any) -> SHAPAnalysis:
        """Validate and store one explainability execution payload.

        Args:
            **kwargs (Any): Configuration keys:
                - `technique` (str): Must be `run_analysis`.
                - `params` (Mapping[str, Any] | None): Optional parameter object.
                - for `run_analysis`, expected keys include:
                  `model_path` (saved `.pkl` model path), `input_path`,
                  `target_column`,
                  `feature_columns`, `max_rows_to_explain`,
                  `background_rows`, `class_index`, and optional `output_dir`.
                - any additional keys are merged into `params` as direct
                  library-friendly shorthand.

        Returns:
            SHAPAnalysis: The fitted tool instance.
        """

        technique_raw = kwargs.get("technique", self._technique)
        if not isinstance(technique_raw, str) or not technique_raw.strip():
            raise InputValidationError("Missing required string parameter: technique")

        params_payload: dict[str, Any] = {}
        params_raw = kwargs.get("params")
        if params_raw is not None:
            if not isinstance(params_raw, Mapping):
                raise InputValidationError("`params` must be an object.")
            params_payload.update(dict(params_raw))
        params_payload.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"technique", "params"}
            }
        )

        self._technique = technique_raw.strip()
        self._params = params_payload
        return self

    @beartype
    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the configured SHAP-analysis workflow.

        Args:
            **kwargs (Any): Optional keys identical to `fit(...)`. If provided,
                they override the existing fitted configuration. Expected keys:
                `technique` and optional `params`, or direct shorthand keys for
                `run_analysis`:
                `model_path` (saved `.pkl` model path), `input_path`,
                `target_column`,
                `feature_columns`, `max_rows_to_explain`,
                `background_rows`, `class_index`, and `output_dir`.

        Returns:
            dict[str, Any]: Output directory and generated artefact paths.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._technique is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `predict(...)`."
            )

        return run_with_validation(
            lambda: _run_shap_analysis_tool(
                technique=self._technique or "",
                params=self._params,
            )
        )

    @beartype
    def predict_proba(self, **kwargs: Any) -> dict[str, Any]:
        """SHAP-analysis tool-level probability payloads are not exposed.

        Args:
            **kwargs (Any): Ignored. Present for template contract compliance.

        Returns:
            dict[str, Any]: Never returned.

        Raises:
            InputValidationError: Always raised because this tool exports
                explainability artefacts rather than raw probability arrays.
        """

        _ = kwargs
        raise InputValidationError(
            "`SHAPAnalysis` does not expose tool-level `predict_proba(...)`."
        )


def run_shap_analysis_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one SHAP-analysis technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not
        be treated as the Python library API. In Python scripts or notebooks,
        import and call `SHAPAnalysis` directly from `ldt.machine_learning`.

    Args:
        technique (str): Analysis mode to run. Must be `run_analysis`.
        params (Mapping[str, Any]): Analysis configuration. Expected keys:
            `model_path` (saved `.pkl` model path), `input_path`,
            `feature_columns`,
            `max_rows_to_explain`, `background_rows`; optional keys:
            `target_column`, `class_index`, and `output_dir`.

    Returns:
        dict[str, Any]: Output directory and generated artefact paths.
    """

    return SHAPAnalysis().fit_predict(
        technique=technique,
        params=params,
    )


def _run_shap_analysis_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="shap_analysis",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = technique.strip().lower().replace("-", "_")
    if mode != "run_analysis":
        raise InputValidationError(f"Unsupported SHAP-analysis technique: {technique}")

    model_path = Path(as_required_string(resolved, "model_path")).expanduser()
    input_path = Path(as_required_string(resolved, "input_path")).expanduser()

    target_raw = as_optional_string(resolved, "target_column") or "none"
    target_column = None if target_raw.strip().lower() in {"", "none"} else target_raw

    feature_columns = as_required_string(resolved, "feature_columns")
    max_rows_to_explain = as_required_int(resolved, "max_rows_to_explain", minimum=1)
    background_rows = as_required_int(resolved, "background_rows", minimum=1)

    class_index = parse_class_index(
        as_optional_string(resolved, "class_index") or "auto"
    )

    output_dir_raw = as_optional_string(resolved, "output_dir")
    output_dir = (
        Path(output_dir_raw).expanduser()
        if output_dir_raw is not None
        else model_path.with_name(f"{model_path.stem}_shap_analysis")
    )

    runner = SHAPAnalysisRunner()
    runner.run(
        model_path=model_path,
        input_csv_path=input_path,
        target_column=target_column,
        feature_columns_raw=feature_columns,
        max_rows_to_explain=max_rows_to_explain,
        background_rows=background_rows,
        class_index=class_index,
        output_dir=output_dir,
    )

    report_path = output_dir / "shap_analysis_report.txt"
    importance_path = output_dir / "shap_feature_importance.csv"
    summary_dot_path = output_dir / "shap_summary_dot.png"
    summary_bar_path = output_dir / "shap_summary_bar.png"

    return {
        "output_dir": str(output_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "importance_csv": str(importance_path.resolve()),
        "summary_dot_plot": str(summary_dot_path.resolve()),
        "summary_bar_plot": str(summary_bar_path.resolve()),
    }
