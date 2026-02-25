from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype


@beartype
@dataclass(frozen=True)
class PipelineProfilerNotebookConfig:
    """Typed configuration for PipelineProfiler notebook generation.

    Attributes:
        notebook_path (Path): Path to the generated notebook file.
        pipeline_payload_path (Path): Path for pipeline payload path.
        max_pipelines (int): Max pipelines.

    """

    notebook_path: Path
    pipeline_payload_path: Path
    max_pipelines: int


@beartype
class PipelineProfilerNotebookTemplate:
    """Template generator for PipelineProfiler notebook visualisations."""

    @beartype
    def write_notebook(self, *, config: PipelineProfilerNotebookConfig) -> Path:
        """Write a notebook that plots pipelines with PipelineProfiler.

        Args:
            config (PipelineProfilerNotebookConfig): Config object used by this workflow.

        Returns:
            Path: Resolved filesystem path.
        """

        destination = config.notebook_path.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        notebook_payload = {
            "cells": [
                self._make_code_cell(self._build_cell_one(config=config)),
                self._make_code_cell(self._build_cell_two(config=config)),
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
    def _build_cell_one(*, config: PipelineProfilerNotebookConfig) -> str:
        """Build notebook cell with pipeline-profiler constants."""

        return (
            "# Cell 1: PipelineProfiler inputs\n"
            "from pathlib import Path\n\n"
            "PIPELINE_PAYLOAD_PATH = "
            f"Path({repr(str(config.pipeline_payload_path.resolve()))})\n"
            f"MAX_PIPELINES = {config.max_pipelines}\n"
        )

    @staticmethod
    @beartype
    def _build_cell_two(*, config: PipelineProfilerNotebookConfig) -> str:
        """Build notebook cell that loads payload and plots pipelines."""

        return (
            "# Cell 2: plot PipelineProfiler matrix in notebook\n"
            "import json\n"
            "import PipelineProfiler\n\n"
            "from PipelineProfiler import _plot_pipeline_matrix as _ppm\n\n"
            "def _use_global_pipeline_indices() -> None:\n"
            "    def _rename_with_global_index(pipelines):\n"
            "        for index, pipeline in enumerate(pipelines, start=1):\n"
            "            source_info = pipeline.setdefault('pipeline_source', {})\n"
            "            source = source_info.get('name', 'pipeline')\n"
            "            source_info['name'] = f'{source} #{index}'\n"
            "    _ppm.rename_pipelines = _rename_with_global_index\n\n"
            "_use_global_pipeline_indices()\n\n"
            "with PIPELINE_PAYLOAD_PATH.open('r') as file:\n"
            "    pipelines = json.load(file)\n\n"
            "if not pipelines:\n"
            "    raise ValueError('Pipeline payload is empty.')\n"
            "pipelines_to_plot = pipelines[:MAX_PIPELINES]\n"
            "print('Plotting pipelines:', len(pipelines_to_plot))\n"
            "PipelineProfiler.plot_pipeline_matrix(pipelines_to_plot)\n"
        )
