from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from beartype import beartype
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.utils.errors import InputValidationError


@beartype
class SHAPAnalysisRunner:
    """Library runner for SHAP explainability on saved model artifacts."""

    @beartype
    def run(
        self,
        *,
        model_path: Path,
        input_csv_path: Path,
        target_column: str | None,
        feature_columns_raw: str,
        max_rows_to_explain: int,
        background_rows: int,
        class_index: int | None,
        output_dir: Path,
    ) -> None:
        """Execute SHAP analysis and export figures + report artifacts.

        Args:
            model_path (Path): Filesystem path used by the workflow.
            input_csv_path (Path): Filesystem path used by the workflow.
            target_column (str | None): Column name for target column.
            feature_columns_raw (str): Comma-separated feature column names.
            max_rows_to_explain (int): Max rows to explain.
            background_rows (int): Background rows.
            class_index (int | None): Optional class index to explain.
            output_dir (Path): Filesystem location for output dir.
        """

        self._validate_model_path(model_path)
        self._validate_csv_path(input_csv_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Loading model and dataset...", total=6)
            with model_path.open("rb") as file:
                model = pickle.load(file)
            data = pd.read_csv(input_csv_path)
            progress.update(
                task_id, advance=1, description="Preparing feature matrix..."
            )

            feature_columns = self._resolve_feature_columns(
                data=data,
                feature_columns_raw=feature_columns_raw,
                target_column=target_column,
            )
            X = data[feature_columns].copy()
            if X.empty:
                raise InputValidationError("No rows available for SHAP analysis.")

            explain_rows = min(max_rows_to_explain, len(X))
            explain_data = (
                X.sample(n=explain_rows, random_state=42)
                if explain_rows < len(X)
                else X.copy()
            )
            background_size = min(background_rows, len(explain_data))
            background_data = (
                explain_data.sample(n=background_size, random_state=42)
                if background_size < len(explain_data)
                else explain_data.copy()
            )
            progress.update(
                task_id, advance=1, description="Computing SHAP explanations..."
            )

            prediction_function, prediction_mode = self._resolve_prediction_function(
                model=model
            )
            try:
                explainer = shap.Explainer(prediction_function, background_data)
                explanation = explainer(explain_data)
            except Exception as exc:  # pragma: no cover - external library failure
                raise InputValidationError(
                    "SHAP explanation failed. Ensure the saved model can predict "
                    "from the provided dataset and feature columns."
                ) from exc
            progress.update(
                task_id, advance=1, description="Summarising SHAP values..."
            )

            shap_values = np.asarray(explanation.values)
            plot_values, selected_class_index = self._resolve_plot_values(
                shap_values=shap_values,
                class_index=class_index,
            )
            mean_abs_values = self._mean_abs_feature_importance(shap_values=shap_values)
            feature_importance = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "mean_abs_shap": mean_abs_values,
                }
            ).sort_values(by="mean_abs_shap", ascending=False)
            progress.update(task_id, advance=1, description="Rendering SHAP figures...")

            output_dir.mkdir(parents=True, exist_ok=True)
            summary_dot_path = output_dir / "shap_summary_dot.png"
            summary_bar_path = output_dir / "shap_summary_bar.png"
            self._save_summary_plot(
                shap_values=plot_values,
                features=explain_data,
                output_path=summary_dot_path,
                plot_type="dot",
            )
            self._save_summary_plot(
                shap_values=plot_values,
                features=explain_data,
                output_path=summary_bar_path,
                plot_type="bar",
            )
            progress.update(task_id, advance=1, description="Writing SHAP report...")

            importance_path = output_dir / "shap_feature_importance.csv"
            report_path = output_dir / "shap_analysis_report.txt"
            feature_importance.to_csv(importance_path, index=False)
            report_path.write_text(
                self._build_report_text(
                    model_path=model_path,
                    input_csv_path=input_csv_path,
                    output_dir=output_dir,
                    explain_rows=len(explain_data),
                    background_rows=len(background_data),
                    prediction_mode=prediction_mode,
                    explainer_name=type(explainer).__name__,
                    selected_class_index=selected_class_index,
                    feature_importance=feature_importance,
                    summary_dot_path=summary_dot_path,
                    summary_bar_path=summary_bar_path,
                    importance_path=importance_path,
                )
            )
            progress.update(task_id, advance=1, description="Done")

    @staticmethod
    @beartype
    def _resolve_feature_columns(
        *,
        data: pd.DataFrame,
        feature_columns_raw: str,
        target_column: str | None,
    ) -> list[str]:
        """Resolve and validate feature columns for SHAP analysis."""

        if feature_columns_raw.lower() == "auto":
            excluded = {target_column} if target_column else set()
            selected = [column for column in data.columns if column not in excluded]
        else:
            selected = [
                item.strip() for item in feature_columns_raw.split(",") if item.strip()
            ]

        if not selected:
            raise InputValidationError("At least one feature column is required.")
        missing = [column for column in selected if column not in data.columns]
        if missing:
            raise InputValidationError(
                f"Missing requested feature columns: {', '.join(missing)}"
            )
        if target_column is not None and target_column in selected:
            raise InputValidationError(
                "Target column cannot also be part of SHAP feature columns."
            )
        return selected

    @staticmethod
    @beartype
    def _resolve_prediction_function(*, model: Any) -> tuple[Any, str]:
        """Resolve the most informative prediction callable from a model."""

        if hasattr(model, "predict_proba"):
            return model.predict_proba, "predict_proba"
        if hasattr(model, "decision_function"):
            return model.decision_function, "decision_function"
        if hasattr(model, "predict"):
            return model.predict, "predict"
        raise InputValidationError(
            "Model does not expose predict_proba, decision_function, or predict."
        )

    @staticmethod
    @beartype
    def _resolve_plot_values(
        *,
        shap_values: np.ndarray,
        class_index: int | None,
    ) -> tuple[np.ndarray, int | None]:
        """Resolve SHAP values to a 2D matrix for summary plots."""

        if shap_values.ndim == 2:
            return shap_values, None
        if shap_values.ndim != 3:
            raise InputValidationError(
                "Unexpected SHAP value shape. Expected 2D or 3D arrays."
            )

        n_classes = shap_values.shape[2]
        selected_class_index = 0 if class_index is None else class_index
        if selected_class_index < 0 or selected_class_index >= n_classes:
            raise InputValidationError("Class index is out of range for model outputs.")
        return shap_values[:, :, selected_class_index], selected_class_index

    @staticmethod
    @beartype
    def _mean_abs_feature_importance(*, shap_values: np.ndarray) -> np.ndarray:
        """Compute mean absolute SHAP values per feature."""

        if shap_values.ndim == 2:
            return np.mean(np.abs(shap_values), axis=0)
        if shap_values.ndim == 3:
            return np.mean(np.abs(shap_values), axis=(0, 2))
        raise InputValidationError("Unexpected SHAP value shape.")

    @staticmethod
    @beartype
    def _save_summary_plot(
        *,
        shap_values: np.ndarray,
        features: pd.DataFrame,
        output_path: Path,
        plot_type: str,
    ) -> None:
        """Render and save one SHAP summary plot."""

        shap.summary_plot(shap_values, features, show=False, plot_type=plot_type)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close()

    @staticmethod
    @beartype
    def _build_report_text(
        *,
        model_path: Path,
        input_csv_path: Path,
        output_dir: Path,
        explain_rows: int,
        background_rows: int,
        prediction_mode: str,
        explainer_name: str,
        selected_class_index: int | None,
        feature_importance: pd.DataFrame,
        summary_dot_path: Path,
        summary_bar_path: Path,
        importance_path: Path,
    ) -> str:
        """Build human-readable SHAP analysis report text."""

        lines = [
            "SHAP Analysis Report",
            "====================",
            f"Model artifact: {model_path.resolve()}",
            f"Input CSV: {input_csv_path.resolve()}",
            f"Output directory: {output_dir.resolve()}",
            f"Explained rows: {explain_rows}",
            f"Background rows: {background_rows}",
            f"Prediction mode: {prediction_mode}",
            f"SHAP explainer: {explainer_name}",
        ]
        if selected_class_index is not None:
            lines.append(f"Summary plots class index: {selected_class_index}")
        lines.extend(
            [
                "",
                "Generated artifacts",
                "-------------------",
                f"- SHAP summary dot plot: {summary_dot_path.resolve()}",
                f"- SHAP summary bar plot: {summary_bar_path.resolve()}",
                f"- SHAP feature importance CSV: {importance_path.resolve()}",
                "",
                "Top features by mean absolute SHAP value",
                "-----------------------------------------",
            ]
        )
        top = feature_importance.head(20).reset_index(drop=True)
        for idx, row in top.iterrows():
            lines.append(
                f"{idx + 1}. {row['feature']}: {float(row['mean_abs_shap']):.6f}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    @beartype
    def _validate_model_path(path: Path) -> None:
        """Validate model artifact path."""

        if not path.exists() or not path.is_file():
            raise InputValidationError(f"Model artifact path does not exist: {path}")
        if path.suffix.lower() != ".pkl":
            raise InputValidationError(
                "Model artifact path must point to a `.pkl` file."
            )

    @staticmethod
    @beartype
    def _validate_csv_path(path: Path) -> None:
        """Validate CSV path exists and points to a file."""

        if not path.exists() or not path.is_file():
            raise InputValidationError(f"Input CSV path does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise InputValidationError("Input path must point to a .csv file.")
