from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype


@beartype
@dataclass(frozen=True)
class ClassificationNotebookConfig:
    """Typed configuration for reproducibility notebook generation.

    Attributes:
        notebook_path (Path): Path to the generated notebook file.
        dataset_path (Path): Path to the source dataset file.
        output_dir (Path): Filesystem location for output dir.
        estimator_module (str): Estimator module.
        estimator_class_name (str): Name for estimator class.
        estimator_name (str): Name for estimator.
        estimator_init_kwargs (Mapping[str, Any]): Estimator init kwargs.
        target_column (str): Column name for target column.
        feature_columns (tuple[str, ...]): Column names for feature columns.
        metric_keys (tuple[str, ...]): Metric keys.
        cv_folds (int): Cv folds.
        validation_split (float | None): Validation split.
        random_seed (int): Random seed for reproducibility.
        silent_training_output (bool): Whether to silent training output.

    """

    notebook_path: Path
    dataset_path: Path
    output_dir: Path
    estimator_module: str
    estimator_class_name: str
    estimator_name: str
    estimator_init_kwargs: Mapping[str, Any]
    target_column: str
    feature_columns: tuple[str, ...]
    metric_keys: tuple[str, ...]
    cv_folds: int
    validation_split: float | None
    random_seed: int
    silent_training_output: bool


@beartype
class ClassificationExperimentNotebookTemplate:
    """Template generator for standard ML experiment reproducibility notebooks."""

    @beartype
    def write_notebook(self, *, config: ClassificationNotebookConfig) -> Path:
        """Write a reproducibility notebook with three code cells.

        Args:
            config (ClassificationNotebookConfig): Config object used by this workflow.

        Returns:
            Path: Resolved filesystem path.
        """

        destination = config.notebook_path.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        notebook_payload = {
            "cells": [
                self._make_code_cell(self._build_cell_one(config=config)),
                self._make_code_cell(self._build_cell_two(config=config)),
                self._make_code_cell(self._build_cell_three()),
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        destination.write_text(json.dumps(notebook_payload, indent=2))
        return destination

    @staticmethod
    @beartype
    def _make_code_cell(source_text: str) -> dict[str, object]:
        """Build one notebook code-cell payload."""

        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_text.splitlines(keepends=True),
        }

    @staticmethod
    @beartype
    def _build_cell_one(*, config: ClassificationNotebookConfig) -> str:
        """Build notebook cell defining dataset and experiment constants."""

        return (
            "# Cell 1: paths and experiment inputs\n"
            "from pathlib import Path\n\n"
            f"DATA_PATH = Path({repr(str(config.dataset_path.resolve()))})\n"
            f"OUTPUT_DIR = Path({repr(str(config.output_dir.resolve()))})\n"
            f"TARGET_COLUMN = {repr(config.target_column)}\n"
            f"FEATURE_COLUMNS = {repr(list(config.feature_columns))}\n"
            f"METRIC_KEYS = {repr(list(config.metric_keys))}\n"
            f"CV_FOLDS = {config.cv_folds}\n"
            f"VALIDATION_SPLIT = {repr(config.validation_split)}\n"
            f"RANDOM_SEED = {config.random_seed}\n"
            f"SILENT_TRAINING_OUTPUT = {config.silent_training_output}\n"
        )

    @beartype
    def _build_cell_two(self, *, config: ClassificationNotebookConfig) -> str:
        """Build notebook cell running the full experiment."""

        init_lines = self._format_estimator_init_kwargs(
            estimator_class_name=config.estimator_class_name,
            estimator_init_kwargs=config.estimator_init_kwargs,
        )
        return (
            "# Cell 2: imports, estimator instantiation, and experiment run\n"
            "import pandas as pd\n"
            f"from {config.estimator_module} import {config.estimator_class_name}\n"
            "from ldt.machine_learning.tools.templates import ClassificationExperimentTemplate\n\n"
            "data = pd.read_csv(DATA_PATH)\n"
            "modelling_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(subset=[TARGET_COLUMN])\n"
            "X = modelling_data[FEATURE_COLUMNS].copy()\n"
            "y = modelling_data[TARGET_COLUMN].copy()\n\n"
            "if y.nunique(dropna=True) < 2:\n"
            "    raise ValueError('Classification requires at least two distinct target classes.')\n\n"
            f"{init_lines}\n\n"
            "experiment = ClassificationExperimentTemplate()\n"
            "result = experiment.run(\n"
            "    estimator=estimator,\n"
            f"    estimator_name={repr(config.estimator_name)},\n"
            "    X=X,\n"
            "    y=y,\n"
            "    metric_keys=tuple(METRIC_KEYS),\n"
            "    output_dir=OUTPUT_DIR,\n"
            "    cv_folds=CV_FOLDS,\n"
            "    validation_split=VALIDATION_SPLIT,\n"
            "    random_seed=RANDOM_SEED,\n"
            "    silent_training_output=SILENT_TRAINING_OUTPUT,\n"
            ")\n"
        )

    @staticmethod
    @beartype
    def _build_cell_three() -> str:
        """Build notebook cell rendering experiment outputs."""

        return (
            "# Cell 3: results\n"
            "print('1) Classification report')\n"
            "print(result.classification_report_text)\n"
            "print('2) Metrics of interest')\n"
            "print('Ranking metric:', result.metric_key)\n"
            "for metric_key in result.metric_keys:\n"
            "    metric_summary = result.metric_summaries[metric_key]\n"
            "    print(\n"
            '        f"{metric_key} mean ± std: "\n'
            '        f"{metric_summary.mean_score:.4f} ± {metric_summary.std_score:.4f}"\n'
            "    )\n"
            "    print(\n"
            "        'Fold scores:',\n"
            "        ', '.join(f\"{score:.4f}\" for score in metric_summary.fold_scores),\n"
            "    )\n"
            "print('3) Saved summary files and locations')\n"
            "print('Output directory:', OUTPUT_DIR.resolve())\n"
            "print('Model artefact:', result.artifacts.model_path.resolve())\n"
            "print('Summary artefact:', result.artifacts.summary_path.resolve())\n"
            "print('Classification report file:', result.artifacts.report_path.resolve())\n"
        )

    @staticmethod
    @beartype
    def _format_estimator_init_kwargs(
        *, estimator_class_name: str, estimator_init_kwargs: Mapping[str, Any]
    ) -> str:
        """Format estimator constructor code from init kwargs."""

        if not estimator_init_kwargs:
            return f"estimator = {estimator_class_name}()"

        lines = [f"estimator = {estimator_class_name}("]
        for key, value in estimator_init_kwargs.items():
            lines.append(f"    {key}={repr(value)},")
        lines.append(")")
        return "\n".join(lines)
