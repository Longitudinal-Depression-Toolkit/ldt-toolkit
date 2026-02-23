from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype


@beartype
@dataclass(frozen=True)
class ClassificationBenchmarkNotebookConfig:
    """Typed config for benchmark reproducibility notebook generation.

    Attributes:
        notebook_path (Path): Path to the generated notebook file.
        dataset_path (Path): Path to the source dataset file.
        output_root_dir (Path): Filesystem location for output root dir.
        benchmark_name (str): Name for benchmark.
        excluded_estimators (tuple[str, ...]): Excluded estimators.
        target_column (str): Column name for target column.
        feature_columns (tuple[str, ...]): Column names for feature columns.
        metric_keys (tuple[str, ...]): Metric keys.
        cv_folds (int): Cv folds.
        validation_split (float | None): Validation split.
        random_seed (int): Random seed for reproducibility.
        silent_training_output (bool): Whether to silent training output.
        include_pipeline_profiler_at_end (bool): Whether to include pipeline profiler at end.
        pipeline_profiler_max_pipelines (int): Pipeline profiler max pipelines.

    """

    notebook_path: Path
    dataset_path: Path
    output_root_dir: Path
    benchmark_name: str
    excluded_estimators: tuple[str, ...]
    target_column: str
    feature_columns: tuple[str, ...]
    metric_keys: tuple[str, ...]
    cv_folds: int
    validation_split: float | None
    random_seed: int
    silent_training_output: bool
    include_pipeline_profiler_at_end: bool
    pipeline_profiler_max_pipelines: int = 5


@beartype
class ClassificationBenchmarkNotebookTemplate:
    """Template generator for benchmark reproducibility notebooks."""

    @beartype
    def write_notebook(self, *, config: ClassificationBenchmarkNotebookConfig) -> Path:
        """Write benchmark reproducibility notebook cells.

        Args:
            config (ClassificationBenchmarkNotebookConfig): Config object used by this workflow.

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
                self._make_code_cell(self._build_cell_four()),
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
    def _build_cell_one(*, config: ClassificationBenchmarkNotebookConfig) -> str:
        """Build notebook cell defining paths and benchmark constants."""

        return (
            "# Cell 1: benchmark inputs\n"
            "from pathlib import Path\n\n"
            f"DATA_PATH = Path({repr(str(config.dataset_path.resolve()))})\n"
            f"OUTPUT_ROOT_DIR = Path({repr(str(config.output_root_dir.resolve()))})\n"
            f"BENCHMARK_NAME = {repr(config.benchmark_name)}\n"
            f"EXCLUDED_ESTIMATORS = {repr(list(config.excluded_estimators))}\n"
            f"TARGET_COLUMN = {repr(config.target_column)}\n"
            f"FEATURE_COLUMNS = {repr(list(config.feature_columns))}\n"
            f"METRIC_KEYS = {repr(list(config.metric_keys))}\n"
            f"CV_FOLDS = {config.cv_folds}\n"
            f"VALIDATION_SPLIT = {repr(config.validation_split)}\n"
            f"RANDOM_SEED = {config.random_seed}\n"
            f"SILENT_TRAINING_OUTPUT = {config.silent_training_output}\n"
            "INCLUDE_PIPELINE_PROFILER_AT_END = "
            f"{config.include_pipeline_profiler_at_end}\n"
            "PIPELINE_PROFILER_MAX_PIPELINES = "
            f"{config.pipeline_profiler_max_pipelines}\n"
        )

    @staticmethod
    @beartype
    def _build_cell_two(*, config: ClassificationBenchmarkNotebookConfig) -> str:
        """Build notebook cell running benchmark execution."""

        return (
            "# Cell 2: benchmark execution\n"
            "import pandas as pd\n"
            "from ldt.machine_learning.tools.standard_machine_learning."
            "discovery import (\n"
            "    discover_standard_estimators,\n"
            ")\n"
            "from ldt.machine_learning.tools.templates import (\n"
            "    BenchmarkEstimatorSpec,\n"
            "    ClassificationBenchmarkTemplate,\n"
            ")\n"
            "from ldt.utils.metadata import resolve_component_metadata\n\n"
            "data = pd.read_csv(DATA_PATH)\n"
            "modelling_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(\n"
            "    subset=[TARGET_COLUMN]\n"
            ")\n"
            "X = modelling_data[FEATURE_COLUMNS].copy()\n"
            "y = modelling_data[TARGET_COLUMN].copy()\n\n"
            "if y.nunique(dropna=True) < 2:\n"
            "    raise ValueError('Classification requires at least two classes.')\n\n"
            "available_estimators = discover_standard_estimators()\n"
            "selected_estimator_keys = [\n"
            "    key\n"
            "    for key in available_estimators.keys()\n"
            "    if key not in EXCLUDED_ESTIMATORS\n"
            "]\n"
            "if not selected_estimator_keys:\n"
            "    raise ValueError(\n"
            "        'All estimators were excluded in this notebook config.'\n"
            "    )\n\n"
            "estimator_specs = []\n"
            "for estimator_key in selected_estimator_keys:\n"
            "    estimator_template = available_estimators[estimator_key]\n"
            "    estimator_specs.append(\n"
            "        BenchmarkEstimatorSpec(\n"
            "            estimator_key=estimator_key,\n"
            "            estimator_name=resolve_component_metadata(\n"
            "                estimator_template\n"
            "            ).full_name,\n"
            "            estimator=estimator_template.build_estimator(\n"
            "                hyperparameters={},\n"
            "                random_seed=RANDOM_SEED,\n"
            "            ),\n"
            "        )\n"
            "    )\n\n"
            "benchmark_result = ClassificationBenchmarkTemplate().run(\n"
            f"    benchmark_name={repr(config.benchmark_name)},\n"
            "    estimator_specs=tuple(estimator_specs),\n"
            "    X=X,\n"
            "    y=y,\n"
            "    metric_keys=tuple(METRIC_KEYS),\n"
            "    output_root_dir=OUTPUT_ROOT_DIR,\n"
            "    cv_folds=CV_FOLDS,\n"
            "    validation_split=VALIDATION_SPLIT,\n"
            "    random_seed=RANDOM_SEED,\n"
            "    silent_training_output=SILENT_TRAINING_OUTPUT,\n"
            "    generate_pipeline_profiler_html=False,\n"
            ")\n"
        )

    @staticmethod
    @beartype
    def _build_cell_three() -> str:
        """Build notebook cell rendering benchmark results."""

        return (
            "# Cell 3: benchmark results\n"
            "from IPython.display import display\n\n"
            "ranking_rows = []\n"
            "for rank, estimator in enumerate(\n"
            "    benchmark_result.ranked_estimators,\n"
            "    start=1,\n"
            "):\n"
            "    ranking_rows.append(\n"
            "        {\n"
            "            'rank': rank,\n"
            "            'estimator_key': estimator.estimator_key,\n"
            "            'estimator_name': estimator.estimator_name,\n"
            "        }\n"
            "    )\n\n"
            "    for metric_key in estimator.metric_keys:\n"
            "        metric_summary = estimator.metric_summaries[metric_key]\n"
            "        ranking_rows[-1][f'{metric_key}_mean'] = metric_summary.mean_score\n"
            "        ranking_rows[-1][f'{metric_key}_std'] = metric_summary.std_score\n"
            "        ranking_rows[-1][f'{metric_key}_fold_scores'] = ', '.join(\n"
            "            f'{score:.4f}' for score in metric_summary.fold_scores\n"
            "        )\n\n"
            "ranking_table = pd.DataFrame(ranking_rows)\n"
            "print('1) Ranked benchmark results')\n"
            "print('Ranking metric:', benchmark_result.metric_key)\n"
            "print('Metrics evaluated:', ', '.join(benchmark_result.metric_keys))\n"
            "display(ranking_table)\n"
            "print('2) Saved benchmark artifacts')\n"
            "print(\n"
            "    'Benchmark output directory:',\n"
            "    benchmark_result.artifacts.benchmark_output_dir.resolve(),\n"
            ")\n"
            "print('Ranking CSV:', benchmark_result.artifacts.ranking_path.resolve())\n"
            "print(\n"
            "    'Summary JSON:',\n"
            "    benchmark_result.artifacts.summary_path.resolve(),\n"
            ")\n"
            "print(\n"
            "    'Benchmark report:',\n"
            "    benchmark_result.artifacts.report_path.resolve(),\n"
            ")\n"
            "print(\n"
            "    'PipelineProfiler input JSON:',\n"
            "    benchmark_result.artifacts.pipeline_profiler.input_json_path."
            "resolve(),\n"
            ")\n"
        )

    @staticmethod
    @beartype
    def _build_cell_four() -> str:
        """Build notebook cell optionally plotting PipelineProfiler inline."""

        return (
            "# Cell 4: optional PipelineProfiler view (end-of-notebook)\n"
            "if INCLUDE_PIPELINE_PROFILER_AT_END:\n"
            "    import json\n"
            "    import PipelineProfiler\n"
            "    from PipelineProfiler import _plot_pipeline_matrix as _ppm\n"
            "    def _use_global_pipeline_indices() -> None:\n"
            "        def _rename_with_global_index(pipelines):\n"
            "            for index, pipeline in enumerate(pipelines, start=1):\n"
            "                source_info = pipeline.setdefault('pipeline_source', {})\n"
            "                source = source_info.get('name', 'pipeline')\n"
            "                source_info['name'] = f'{source} #{index}'\n"
            "        _ppm.rename_pipelines = _rename_with_global_index\n"
            "    _use_global_pipeline_indices()\n"
            "    payload_path = (\n"
            "        benchmark_result.artifacts.pipeline_profiler.input_json_path\n"
            "    )\n"
            "    with payload_path.open('r') as file:\n"
            "        pipelines = json.load(file)\n"
            "    if not pipelines:\n"
            "        raise ValueError(\n"
            "            'PipelineProfiler payload is empty. Re-run benchmark first.'\n"
            "        )\n"
            "    pipelines_to_plot = pipelines[:PIPELINE_PROFILER_MAX_PIPELINES]\n"
            "    print(\n"
            "        'Plotting pipelines in notebook:',\n"
            "        len(pipelines_to_plot),\n"
            "    )\n"
            "    PipelineProfiler.plot_pipeline_matrix(pipelines_to_plot)\n"
            "else:\n"
            "    print(\n"
            "        'Skipped PipelineProfiler notebook view '\n"
            "        '(INCLUDE_PIPELINE_PROFILER_AT_END=False).'\n"
            "    )\n"
        )


@beartype
@dataclass(frozen=True)
class LongitudinalBenchmarkNotebookConfig:
    """Typed config for longitudinal benchmark reproducibility notebook.

    Attributes:
        notebook_path (Path): Path to the generated notebook file.
        dataset_path (Path): Path to the source dataset file.
        output_root_dir (Path): Filesystem location for output root dir.
        benchmark_name (str): Name for benchmark.
        excluded_estimators (tuple[str, ...]): Excluded estimators.
        target_column (str): Column name for target column.
        feature_columns (tuple[str, ...]): Column names for feature columns.
        feature_groups (tuple[tuple[int, ...], ...]): Feature groups.
        non_longitudinal_features (tuple[int, ...]): Column names for non longitudinal features.
        metric_keys (tuple[str, ...]): Metric keys.
        cv_folds (int): Cv folds.
        validation_split (float | None): Validation split.
        random_seed (int): Random seed for reproducibility.
        silent_training_output (bool): Whether to silent training output.
        include_pipeline_profiler_at_end (bool): Whether to include pipeline profiler at end.
        pipeline_profiler_max_pipelines (int): Pipeline profiler max pipelines.

    """

    notebook_path: Path
    dataset_path: Path
    output_root_dir: Path
    benchmark_name: str
    excluded_estimators: tuple[str, ...]
    target_column: str
    feature_columns: tuple[str, ...]
    feature_groups: tuple[tuple[int, ...], ...]
    non_longitudinal_features: tuple[int, ...]
    metric_keys: tuple[str, ...]
    cv_folds: int
    validation_split: float | None
    random_seed: int
    silent_training_output: bool
    include_pipeline_profiler_at_end: bool
    pipeline_profiler_max_pipelines: int = 5


@beartype
class LongitudinalBenchmarkNotebookTemplate:
    """Template generator for longitudinal benchmark reproducibility notebooks."""

    @beartype
    def write_notebook(self, *, config: LongitudinalBenchmarkNotebookConfig) -> Path:
        """Write longitudinal benchmark reproducibility notebook cells.

        Args:
            config (LongitudinalBenchmarkNotebookConfig): Config object used by this workflow.

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
                self._make_code_cell(self._build_cell_four()),
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
    def _build_cell_one(*, config: LongitudinalBenchmarkNotebookConfig) -> str:
        """Build notebook cell defining paths and benchmark constants."""

        return (
            "# Cell 1: longitudinal benchmark inputs\n"
            "from pathlib import Path\n\n"
            f"DATA_PATH = Path({repr(str(config.dataset_path.resolve()))})\n"
            f"OUTPUT_ROOT_DIR = Path({repr(str(config.output_root_dir.resolve()))})\n"
            f"BENCHMARK_NAME = {repr(config.benchmark_name)}\n"
            f"EXCLUDED_ESTIMATORS = {repr(list(config.excluded_estimators))}\n"
            f"TARGET_COLUMN = {repr(config.target_column)}\n"
            f"FEATURE_COLUMNS = {repr(list(config.feature_columns))}\n"
            f"FEATURE_GROUPS = {repr([list(group) for group in config.feature_groups])}\n"
            "NON_LONGITUDINAL_FEATURES = "
            f"{repr(list(config.non_longitudinal_features))}\n"
            f"METRIC_KEYS = {repr(list(config.metric_keys))}\n"
            f"CV_FOLDS = {config.cv_folds}\n"
            f"VALIDATION_SPLIT = {repr(config.validation_split)}\n"
            f"RANDOM_SEED = {config.random_seed}\n"
            f"SILENT_TRAINING_OUTPUT = {config.silent_training_output}\n"
            "INCLUDE_PIPELINE_PROFILER_AT_END = "
            f"{config.include_pipeline_profiler_at_end}\n"
            "PIPELINE_PROFILER_MAX_PIPELINES = "
            f"{config.pipeline_profiler_max_pipelines}\n"
        )

    @staticmethod
    @beartype
    def _build_cell_two(*, config: LongitudinalBenchmarkNotebookConfig) -> str:
        """Build notebook cell running longitudinal benchmark execution."""

        return (
            "# Cell 2: longitudinal benchmark execution\n"
            "import pandas as pd\n"
            "from ldt.machine_learning.tools.longitudinal_machine_learning.discovery import (\n"
            "    discover_longitudinal_estimators,\n"
            ")\n"
            "from ldt.machine_learning.tools.longitudinal_machine_learning.target_encoding import (\n"
            "    LongitudinalTargetEncoder,\n"
            ")\n"
            "from ldt.machine_learning.tools.target_scenarios import TargetScenarioPlanner\n"
            "from ldt.machine_learning.tools.templates import (\n"
            "    BenchmarkEstimatorSpec,\n"
            "    ClassificationBenchmarkTemplate,\n"
            ")\n"
            "from ldt.utils.metadata import resolve_component_metadata\n\n"
            "def _build_estimator_specs(\n"
            "    *,\n"
            "    random_seed: int,\n"
            "    feature_groups: tuple[tuple[int, ...], ...],\n"
            "    non_longitudinal_features: tuple[int, ...],\n"
            "    feature_columns: list[str],\n"
            ") -> tuple[BenchmarkEstimatorSpec, ...]:\n"
            "    available_estimators = discover_longitudinal_estimators()\n"
            "    selected_estimator_keys = [\n"
            "        key\n"
            "        for key in available_estimators.keys()\n"
            "        if key not in EXCLUDED_ESTIMATORS\n"
            "    ]\n"
            "    if not selected_estimator_keys:\n"
            "        raise ValueError(\n"
            "            'All estimators were excluded in this notebook config.'\n"
            "        )\n"
            "    estimator_specs = []\n"
            "    for estimator_key in selected_estimator_keys:\n"
            "        estimator_template = available_estimators[estimator_key]\n"
            "        estimator_specs.append(\n"
            "            BenchmarkEstimatorSpec(\n"
            "                estimator_key=estimator_key,\n"
            "                estimator_name=resolve_component_metadata(\n"
            "                    estimator_template\n"
            "                ).full_name,\n"
            "                estimator=estimator_template.build_estimator(\n"
            "                    random_seed=random_seed,\n"
            "                    feature_groups=feature_groups,\n"
            "                    non_longitudinal_features=non_longitudinal_features,\n"
            "                    feature_list_names=tuple(feature_columns),\n"
            "                ),\n"
            "            )\n"
            "        )\n"
            "    return tuple(estimator_specs)\n\n"
            "data = pd.read_csv(DATA_PATH)\n"
            "modelling_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(\n"
            "    subset=[TARGET_COLUMN]\n"
            ")\n"
            "X = modelling_data[FEATURE_COLUMNS].copy()\n"
            "y_raw = modelling_data[TARGET_COLUMN].copy()\n\n"
            "if y_raw.nunique(dropna=True) < 2:\n"
            "    raise ValueError('Classification requires at least two classes.')\n\n"
            "feature_groups = tuple(tuple(group) for group in FEATURE_GROUPS)\n"
            "non_longitudinal_features = tuple(NON_LONGITUDINAL_FEATURES)\n\n"
            "scenario_runs = []\n"
            "if y_raw.nunique(dropna=True) > 2:\n"
            "    scenarios = TargetScenarioPlanner.build_one_vs_rest_scenarios(y=y_raw)\n"
            "    for scenario in scenarios:\n"
            "        estimator_specs = _build_estimator_specs(\n"
            "            random_seed=RANDOM_SEED,\n"
            "            feature_groups=feature_groups,\n"
            "            non_longitudinal_features=non_longitudinal_features,\n"
            "            feature_columns=FEATURE_COLUMNS,\n"
            "        )\n"
            "        scenario_output_root_dir = OUTPUT_ROOT_DIR / scenario.scenario_key\n"
            "        benchmark_result = ClassificationBenchmarkTemplate().run(\n"
            "            benchmark_name=BENCHMARK_NAME,\n"
            "            estimator_specs=estimator_specs,\n"
            "            X=X,\n"
            "            y=scenario.y_binary,\n"
            "            metric_keys=tuple(METRIC_KEYS),\n"
            "            output_root_dir=scenario_output_root_dir,\n"
            "            cv_folds=CV_FOLDS,\n"
            "            validation_split=VALIDATION_SPLIT,\n"
            "            random_seed=RANDOM_SEED,\n"
            "            silent_training_output=SILENT_TRAINING_OUTPUT,\n"
            "            generate_pipeline_profiler_html=False,\n"
            "        )\n"
            "        scenario_runs.append(\n"
            "            {\n"
            "                'scenario_key': scenario.scenario_key,\n"
            "                'scenario_label': scenario.scenario_label,\n"
            "                'benchmark_result': benchmark_result,\n"
            "            }\n"
            "        )\n"
            "else:\n"
            "    target_encoding_result = LongitudinalTargetEncoder.encode_if_needed(\n"
            "        y=y_raw\n"
            "    )\n"
            "    LongitudinalTargetEncoder.print_encoding_notice(\n"
            "        target_column=TARGET_COLUMN,\n"
            "        result=target_encoding_result,\n"
            "    )\n"
            "    estimator_specs = _build_estimator_specs(\n"
            "        random_seed=RANDOM_SEED,\n"
            "        feature_groups=feature_groups,\n"
            "        non_longitudinal_features=non_longitudinal_features,\n"
            "        feature_columns=FEATURE_COLUMNS,\n"
            "    )\n"
            "    benchmark_result = ClassificationBenchmarkTemplate().run(\n"
            "        benchmark_name=BENCHMARK_NAME,\n"
            "        estimator_specs=estimator_specs,\n"
            "        X=X,\n"
            "        y=target_encoding_result.encoded_target,\n"
            "        metric_keys=tuple(METRIC_KEYS),\n"
            "        output_root_dir=OUTPUT_ROOT_DIR,\n"
            "        cv_folds=CV_FOLDS,\n"
            "        validation_split=VALIDATION_SPLIT,\n"
            "        random_seed=RANDOM_SEED,\n"
            "        silent_training_output=SILENT_TRAINING_OUTPUT,\n"
            "        generate_pipeline_profiler_html=False,\n"
            "    )\n"
            "    scenario_runs.append(\n"
            "        {\n"
            "            'scenario_key': 'single_run',\n"
            "            'scenario_label': 'Single run',\n"
            "            'benchmark_result': benchmark_result,\n"
            "        }\n"
            "    )\n"
        )

    @staticmethod
    @beartype
    def _build_cell_three() -> str:
        """Build notebook cell rendering per-scenario benchmark results."""

        return (
            "# Cell 3: per-scenario benchmark results\n"
            "from IPython.display import Markdown, display\n\n"
            "for scenario_run in scenario_runs:\n"
            "    benchmark_result = scenario_run['benchmark_result']\n"
            "    display(Markdown(f\"## Scenario: {scenario_run['scenario_label']}\"))\n"
            "    ranking_rows = []\n"
            "    for rank, estimator in enumerate(\n"
            "        benchmark_result.ranked_estimators,\n"
            "        start=1,\n"
            "    ):\n"
            "        ranking_rows.append(\n"
            "            {\n"
            "                'rank': rank,\n"
            "                'estimator_key': estimator.estimator_key,\n"
            "                'estimator_name': estimator.estimator_name,\n"
            "            }\n"
            "        )\n"
            "        for metric_key in estimator.metric_keys:\n"
            "            metric_summary = estimator.metric_summaries[metric_key]\n"
            "            ranking_rows[-1][f'{metric_key}_mean'] = (\n"
            "                metric_summary.mean_score\n"
            "            )\n"
            "            ranking_rows[-1][f'{metric_key}_std'] = (\n"
            "                metric_summary.std_score\n"
            "            )\n"
            "            ranking_rows[-1][f'{metric_key}_fold_scores'] = ', '.join(\n"
            "                f'{score:.4f}' for score in metric_summary.fold_scores\n"
            "            )\n"
            "    ranking_table = pd.DataFrame(ranking_rows)\n"
            "    print('Ranking metric:', benchmark_result.metric_key)\n"
            "    print('Metrics evaluated:', ', '.join(benchmark_result.metric_keys))\n"
            "    display(ranking_table)\n"
            "    print('Benchmark output directory:', benchmark_result.artifacts.benchmark_output_dir.resolve())\n"
            "    print('Ranking CSV:', benchmark_result.artifacts.ranking_path.resolve())\n"
            "    print('Summary JSON:', benchmark_result.artifacts.summary_path.resolve())\n"
            "    print('Benchmark report:', benchmark_result.artifacts.report_path.resolve())\n"
            "    print('PipelineProfiler input JSON:', benchmark_result.artifacts.pipeline_profiler.input_json_path.resolve())\n"
        )

    @staticmethod
    @beartype
    def _build_cell_four() -> str:
        """Build notebook cell plotting per-scenario PipelineProfiler sections."""

        return (
            "# Cell 4: per-scenario PipelineProfiler (end-of-notebook)\n"
            "if INCLUDE_PIPELINE_PROFILER_AT_END:\n"
            "    import json\n"
            "    import PipelineProfiler\n"
            "    from PipelineProfiler import _plot_pipeline_matrix as _ppm\n"
            "    from IPython.display import Markdown, display\n\n"
            "    def _use_global_pipeline_indices() -> None:\n"
            "        def _rename_with_global_index(pipelines):\n"
            "            for index, pipeline in enumerate(pipelines, start=1):\n"
            "                source_info = pipeline.setdefault('pipeline_source', {})\n"
            "                source = source_info.get('name', 'pipeline')\n"
            "                source_info['name'] = f'{source} #{index}'\n"
            "        _ppm.rename_pipelines = _rename_with_global_index\n"
            "    _use_global_pipeline_indices()\n\n"
            "    for scenario_run in scenario_runs:\n"
            "        benchmark_result = scenario_run['benchmark_result']\n"
            "        display(\n"
            "            Markdown(\n"
            "                f\"### PipelineProfiler: {scenario_run['scenario_label']}\"\n"
            "            )\n"
            "        )\n"
            "        payload_path = benchmark_result.artifacts.pipeline_profiler.input_json_path\n"
            "        with payload_path.open('r') as file:\n"
            "            pipelines = json.load(file)\n"
            "        if not pipelines:\n"
            "            print('Skipped: PipelineProfiler payload is empty.')\n"
            "            continue\n"
            "        pipelines_to_plot = pipelines[:min(PIPELINE_PROFILER_MAX_PIPELINES, len(pipelines))]\n"
            "        print('Plotting pipelines in notebook:', len(pipelines_to_plot))\n"
            "        PipelineProfiler.plot_pipeline_matrix(pipelines_to_plot)\n"
            "else:\n"
            "    print(\n"
            "        'Skipped PipelineProfiler notebook view '\n"
            "        '(INCLUDE_PIPELINE_PROFILER_AT_END=False).'\n"
            "    )\n"
        )


@beartype
@dataclass(frozen=True)
class MixedBenchmarkNotebookConfig:
    """Typed config for mixed standard+longitudinal benchmark notebooks.

    Attributes:
        notebook_path (Path): Path to the generated notebook file.
        dataset_path (Path): Path to the source dataset file.
        output_root_dir (Path): Filesystem location for output root dir.
        benchmark_name (str): Name for benchmark.
        excluded_estimators (tuple[str, ...]): Excluded estimators.
        target_column (str): Column name for target column.
        feature_columns (tuple[str, ...]): Column names for feature columns.
        feature_groups (tuple[tuple[int, ...], ...]): Feature groups.
        non_longitudinal_features (tuple[int, ...]): Column names for non longitudinal features.
        metric_keys (tuple[str, ...]): Metric keys.
        cv_folds (int): Cv folds.
        validation_split (float | None): Validation split.
        random_seed (int): Random seed for reproducibility.
        silent_training_output (bool): Whether to silent training output.
        include_pipeline_profiler_at_end (bool): Whether to include pipeline profiler at end.
        pipeline_profiler_max_pipelines (int): Pipeline profiler max pipelines.

    """

    notebook_path: Path
    dataset_path: Path
    output_root_dir: Path
    benchmark_name: str
    excluded_estimators: tuple[str, ...]
    target_column: str
    feature_columns: tuple[str, ...]
    feature_groups: tuple[tuple[int, ...], ...]
    non_longitudinal_features: tuple[int, ...]
    metric_keys: tuple[str, ...]
    cv_folds: int
    validation_split: float | None
    random_seed: int
    silent_training_output: bool
    include_pipeline_profiler_at_end: bool
    pipeline_profiler_max_pipelines: int = 5


@beartype
class MixedBenchmarkNotebookTemplate:
    """Template generator for standard+longitudinal benchmark notebooks."""

    @beartype
    def write_notebook(self, *, config: MixedBenchmarkNotebookConfig) -> Path:
        """Write mixed benchmark reproducibility notebook cells.

        Args:
            config (MixedBenchmarkNotebookConfig): Config object used by this workflow.

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
                self._make_code_cell(self._build_cell_four()),
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
    def _build_cell_one(*, config: MixedBenchmarkNotebookConfig) -> str:
        """Build notebook cell defining paths and benchmark constants."""

        return (
            "# Cell 1: mixed benchmark inputs\n"
            "from pathlib import Path\n\n"
            f"DATA_PATH = Path({repr(str(config.dataset_path.resolve()))})\n"
            f"OUTPUT_ROOT_DIR = Path({repr(str(config.output_root_dir.resolve()))})\n"
            f"BENCHMARK_NAME = {repr(config.benchmark_name)}\n"
            f"EXCLUDED_ESTIMATORS = {repr(list(config.excluded_estimators))}\n"
            f"TARGET_COLUMN = {repr(config.target_column)}\n"
            f"FEATURE_COLUMNS = {repr(list(config.feature_columns))}\n"
            f"FEATURE_GROUPS = {repr([list(group) for group in config.feature_groups])}\n"
            "NON_LONGITUDINAL_FEATURES = "
            f"{repr(list(config.non_longitudinal_features))}\n"
            f"METRIC_KEYS = {repr(list(config.metric_keys))}\n"
            f"CV_FOLDS = {config.cv_folds}\n"
            f"VALIDATION_SPLIT = {repr(config.validation_split)}\n"
            f"RANDOM_SEED = {config.random_seed}\n"
            f"SILENT_TRAINING_OUTPUT = {config.silent_training_output}\n"
            "INCLUDE_PIPELINE_PROFILER_AT_END = "
            f"{config.include_pipeline_profiler_at_end}\n"
            "PIPELINE_PROFILER_MAX_PIPELINES = "
            f"{config.pipeline_profiler_max_pipelines}\n"
        )

    @staticmethod
    @beartype
    def _build_cell_two(*, config: MixedBenchmarkNotebookConfig) -> str:
        """Build notebook cell running mixed benchmark execution."""

        return (
            "# Cell 2: mixed benchmark execution\n"
            "import pandas as pd\n"
            "from ldt.machine_learning.tools.longitudinal_machine_learning.discovery import (\n"
            "    discover_longitudinal_estimators,\n"
            ")\n"
            "from ldt.machine_learning.tools.longitudinal_machine_learning.target_encoding import (\n"
            "    LongitudinalTargetEncoder,\n"
            ")\n"
            "from ldt.machine_learning.tools.standard_machine_learning.discovery import (\n"
            "    discover_standard_estimators,\n"
            ")\n"
            "from ldt.machine_learning.tools.target_scenarios import TargetScenarioPlanner\n"
            "from ldt.machine_learning.tools.templates import (\n"
            "    BenchmarkEstimatorSpec,\n"
            "    ClassificationBenchmarkTemplate,\n"
            ")\n"
            "from ldt.utils.metadata import resolve_component_metadata\n\n"
            "def _build_estimator_specs(\n"
            "    *,\n"
            "    random_seed: int,\n"
            "    feature_groups: tuple[tuple[int, ...], ...],\n"
            "    non_longitudinal_features: tuple[int, ...],\n"
            "    feature_columns: list[str],\n"
            ") -> tuple[BenchmarkEstimatorSpec, ...]:\n"
            "    standard_estimators = discover_standard_estimators()\n"
            "    longitudinal_estimators = discover_longitudinal_estimators()\n"
            "    selected_standard = {\n"
            "        key: template\n"
            "        for key, template in standard_estimators.items()\n"
            "        if key not in EXCLUDED_ESTIMATORS\n"
            "    }\n"
            "    selected_longitudinal = {\n"
            "        key: template\n"
            "        for key, template in longitudinal_estimators.items()\n"
            "        if key not in EXCLUDED_ESTIMATORS\n"
            "    }\n"
            "    if not selected_standard and not selected_longitudinal:\n"
            "        raise ValueError('All estimators are excluded in this notebook config.')\n"
            "    estimator_specs = []\n"
            "    for estimator_key, estimator_template in selected_standard.items():\n"
            "        metadata = resolve_component_metadata(estimator_template)\n"
            "        estimator_specs.append(\n"
            "            BenchmarkEstimatorSpec(\n"
            "                estimator_key=estimator_key,\n"
            '                estimator_name=f"{metadata.full_name} [standard-ml]",\n'
            "                estimator=estimator_template.build_estimator(\n"
            "                    hyperparameters={},\n"
            "                    random_seed=random_seed,\n"
            "                ),\n"
            "            )\n"
            "        )\n"
            "    for estimator_key, estimator_template in selected_longitudinal.items():\n"
            "        metadata = resolve_component_metadata(estimator_template)\n"
            "        estimator_specs.append(\n"
            "            BenchmarkEstimatorSpec(\n"
            "                estimator_key=estimator_key,\n"
            '                estimator_name=f"{metadata.full_name} [longitudinal-ml]",\n'
            "                estimator=estimator_template.build_estimator(\n"
            "                    random_seed=random_seed,\n"
            "                    feature_groups=feature_groups,\n"
            "                    non_longitudinal_features=non_longitudinal_features,\n"
            "                    feature_list_names=tuple(feature_columns),\n"
            "                ),\n"
            "            )\n"
            "        )\n"
            "    return tuple(estimator_specs)\n\n"
            "data = pd.read_csv(DATA_PATH)\n"
            "modelling_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna(\n"
            "    subset=[TARGET_COLUMN]\n"
            ")\n"
            "X = modelling_data[FEATURE_COLUMNS].copy()\n"
            "y_raw = modelling_data[TARGET_COLUMN].copy()\n\n"
            "if y_raw.nunique(dropna=True) < 2:\n"
            "    raise ValueError('Classification requires at least two classes.')\n\n"
            "feature_groups = tuple(tuple(group) for group in FEATURE_GROUPS)\n"
            "non_longitudinal_features = tuple(NON_LONGITUDINAL_FEATURES)\n\n"
            "scenario_runs = []\n"
            "if y_raw.nunique(dropna=True) > 2:\n"
            "    scenarios = TargetScenarioPlanner.build_one_vs_rest_scenarios(y=y_raw)\n"
            "    for scenario in scenarios:\n"
            "        estimator_specs = _build_estimator_specs(\n"
            "            random_seed=RANDOM_SEED,\n"
            "            feature_groups=feature_groups,\n"
            "            non_longitudinal_features=non_longitudinal_features,\n"
            "            feature_columns=FEATURE_COLUMNS,\n"
            "        )\n"
            "        scenario_output_root_dir = OUTPUT_ROOT_DIR / scenario.scenario_key\n"
            "        benchmark_result = ClassificationBenchmarkTemplate().run(\n"
            "            benchmark_name=BENCHMARK_NAME,\n"
            "            estimator_specs=estimator_specs,\n"
            "            X=X,\n"
            "            y=scenario.y_binary,\n"
            "            metric_keys=tuple(METRIC_KEYS),\n"
            "            output_root_dir=scenario_output_root_dir,\n"
            "            cv_folds=CV_FOLDS,\n"
            "            validation_split=VALIDATION_SPLIT,\n"
            "            random_seed=RANDOM_SEED,\n"
            "            silent_training_output=SILENT_TRAINING_OUTPUT,\n"
            "            generate_pipeline_profiler_html=False,\n"
            "        )\n"
            "        scenario_runs.append(\n"
            "            {\n"
            "                'scenario_key': scenario.scenario_key,\n"
            "                'scenario_label': scenario.scenario_label,\n"
            "                'benchmark_result': benchmark_result,\n"
            "            }\n"
            "        )\n"
            "else:\n"
            "    target_encoding_result = LongitudinalTargetEncoder.encode_if_needed(\n"
            "        y=y_raw\n"
            "    )\n"
            "    LongitudinalTargetEncoder.print_encoding_notice(\n"
            "        target_column=TARGET_COLUMN,\n"
            "        result=target_encoding_result,\n"
            "    )\n"
            "    estimator_specs = _build_estimator_specs(\n"
            "        random_seed=RANDOM_SEED,\n"
            "        feature_groups=feature_groups,\n"
            "        non_longitudinal_features=non_longitudinal_features,\n"
            "        feature_columns=FEATURE_COLUMNS,\n"
            "    )\n"
            "    benchmark_result = ClassificationBenchmarkTemplate().run(\n"
            "        benchmark_name=BENCHMARK_NAME,\n"
            "        estimator_specs=estimator_specs,\n"
            "        X=X,\n"
            "        y=target_encoding_result.encoded_target,\n"
            "        metric_keys=tuple(METRIC_KEYS),\n"
            "        output_root_dir=OUTPUT_ROOT_DIR,\n"
            "        cv_folds=CV_FOLDS,\n"
            "        validation_split=VALIDATION_SPLIT,\n"
            "        random_seed=RANDOM_SEED,\n"
            "        silent_training_output=SILENT_TRAINING_OUTPUT,\n"
            "        generate_pipeline_profiler_html=False,\n"
            "    )\n"
            "    scenario_runs.append(\n"
            "        {\n"
            "            'scenario_key': 'single_run',\n"
            "            'scenario_label': 'Single run',\n"
            "            'benchmark_result': benchmark_result,\n"
            "        }\n"
            "    )\n"
        )

    @staticmethod
    @beartype
    def _build_cell_three() -> str:
        """Build notebook cell rendering per-scenario benchmark results."""

        return (
            "# Cell 3: per-scenario benchmark results\n"
            "from IPython.display import Markdown, display\n\n"
            "for scenario_run in scenario_runs:\n"
            "    benchmark_result = scenario_run['benchmark_result']\n"
            "    display(Markdown(f\"## Scenario: {scenario_run['scenario_label']}\"))\n"
            "    ranking_rows = []\n"
            "    for rank, estimator in enumerate(\n"
            "        benchmark_result.ranked_estimators,\n"
            "        start=1,\n"
            "    ):\n"
            "        ranking_rows.append(\n"
            "            {\n"
            "                'rank': rank,\n"
            "                'estimator_key': estimator.estimator_key,\n"
            "                'estimator_name': estimator.estimator_name,\n"
            "            }\n"
            "        )\n"
            "        for metric_key in estimator.metric_keys:\n"
            "            metric_summary = estimator.metric_summaries[metric_key]\n"
            "            ranking_rows[-1][f'{metric_key}_mean'] = metric_summary.mean_score\n"
            "            ranking_rows[-1][f'{metric_key}_std'] = metric_summary.std_score\n"
            "            ranking_rows[-1][f'{metric_key}_fold_scores'] = ', '.join(\n"
            "                f'{score:.4f}' for score in metric_summary.fold_scores\n"
            "            )\n"
            "    ranking_table = pd.DataFrame(ranking_rows)\n"
            "    print('Ranking metric:', benchmark_result.metric_key)\n"
            "    print('Metrics evaluated:', ', '.join(benchmark_result.metric_keys))\n"
            "    display(ranking_table)\n"
            "    print('Benchmark output directory:', benchmark_result.artifacts.benchmark_output_dir.resolve())\n"
            "    print('Ranking CSV:', benchmark_result.artifacts.ranking_path.resolve())\n"
            "    print('Summary JSON:', benchmark_result.artifacts.summary_path.resolve())\n"
            "    print('Benchmark report:', benchmark_result.artifacts.report_path.resolve())\n"
            "    print('PipelineProfiler input JSON:', benchmark_result.artifacts.pipeline_profiler.input_json_path.resolve())\n"
        )

    @staticmethod
    @beartype
    def _build_cell_four() -> str:
        """Build notebook cell plotting per-scenario PipelineProfiler sections."""

        return (
            "# Cell 4: per-scenario PipelineProfiler (end-of-notebook)\n"
            "if INCLUDE_PIPELINE_PROFILER_AT_END:\n"
            "    import json\n"
            "    import PipelineProfiler\n"
            "    from PipelineProfiler import _plot_pipeline_matrix as _ppm\n"
            "    from IPython.display import Markdown, display\n\n"
            "    def _use_global_pipeline_indices() -> None:\n"
            "        def _rename_with_global_index(pipelines):\n"
            "            for index, pipeline in enumerate(pipelines, start=1):\n"
            "                source_info = pipeline.setdefault('pipeline_source', {})\n"
            "                source = source_info.get('name', 'pipeline')\n"
            "                source_info['name'] = f'{source} #{index}'\n"
            "        _ppm.rename_pipelines = _rename_with_global_index\n"
            "    _use_global_pipeline_indices()\n\n"
            "    for scenario_run in scenario_runs:\n"
            "        benchmark_result = scenario_run['benchmark_result']\n"
            "        display(\n"
            "            Markdown(\n"
            "                f\"### PipelineProfiler: {scenario_run['scenario_label']}\"\n"
            "            )\n"
            "        )\n"
            "        payload_path = benchmark_result.artifacts.pipeline_profiler.input_json_path\n"
            "        with payload_path.open('r') as file:\n"
            "            pipelines = json.load(file)\n"
            "        if not pipelines:\n"
            "            print('Skipped: PipelineProfiler payload is empty.')\n"
            "            continue\n"
            "        pipelines_to_plot = pipelines[:min(PIPELINE_PROFILER_MAX_PIPELINES, len(pipelines))]\n"
            "        print('Plotting pipelines in notebook:', len(pipelines_to_plot))\n"
            "        PipelineProfiler.plot_pipeline_matrix(pipelines_to_plot)\n"
            "else:\n"
            "    print(\n"
            "        'Skipped PipelineProfiler notebook view '\n"
            "        '(INCLUDE_PIPELINE_PROFILER_AT_END=False).'\n"
            "    )\n"
        )


# Backward-compatible aliases.
StandardMLBenchmarkNotebookConfig = ClassificationBenchmarkNotebookConfig
StandardMLBenchmarkNotebookTemplate = ClassificationBenchmarkNotebookTemplate
