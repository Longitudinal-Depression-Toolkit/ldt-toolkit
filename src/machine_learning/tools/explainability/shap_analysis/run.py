from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.machine_learning.catalog import resolve_technique_with_defaults
from src.machine_learning.support.inputs import (
    as_optional_string,
    as_required_int,
    as_required_string,
    parse_class_index,
    run_with_validation,
)
from src.utils.errors import InputValidationError

from .service import SHAPAnalysisRunner


def run_shap_analysis_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run SHAP explainability analysis for a trained classification model.

    SHAP (SHapley Additive exPlanations) attributes each prediction to feature
    contributions using Shapley-value principles from cooperative game theory.
    This tool computes both local and global explanations from a saved model and
    dataset, then exports human-readable artifacts for review.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `run_analysis` | Loads a saved model, computes SHAP values, and writes explainability artifacts. |

    Output artifacts written by the analysis:

    | Artifact | Description |
    | --- | --- |
    | `shap_analysis_report.txt` | Text summary of the analysis run and configuration. |
    | `shap_feature_importance.csv` | Global feature-importance ranking from absolute SHAP magnitudes. |
    | `shap_summary_dot.png` | SHAP summary dot plot (distribution of signed contributions). |
    | `shap_summary_bar.png` | SHAP summary bar plot (mean absolute contribution per feature). |

    Multi-class note:
        For multi-class models, `class_index` controls which class slice is
        analysed (`auto` selects the default class handling in the runner).

    Args:
        technique (str): Analysis mode to run. Must be `run_analysis`.
        params (Mapping[str, Any]): Analysis configuration, including
            `model_path`, `input_path`, `target_column` (optional),
            `feature_columns`, `max_rows_to_explain`, `background_rows`,
            `class_index`, and `output_dir` (optional).

    Returns:
        dict[str, Any]: Output directory and generated artifact paths.

    Examples:
        ```python
        from ldt.machine_learning.tools.explainability.shap_analysis.run import run_shap_analysis_tool

        result = run_shap_analysis_tool(
            technique="run_analysis",
            params={
                "model_path": "./outputs/standard_ml/random_forest/model.joblib",
                "input_path": "./data/millennium.csv",
                "target_column": "depression_status",
                "feature_columns": "sleep_score,anxiety_score,sex,income",
                "max_rows_to_explain": 500,
                "background_rows": 200,
                "class_index": "auto",
            },
        )
        ```
    """

    return run_with_validation(
        lambda: _run_shap_analysis_tool(technique=technique, params=params)
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
