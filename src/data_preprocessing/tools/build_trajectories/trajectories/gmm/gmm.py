from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from rich.console import Console

from src.utils.metadata import ComponentMetadata

from ...trajectory import (
    TrajectoryModel,
    TrajectoryResult,
    build_trajectory_dataset,
    normalise_trajectory_names,
)

console = Console()


@dataclass(frozen=True)
class _GMMBridgeOutput:
    """Container for parsed GMM bridge outputs."""

    assignments: pd.DataFrame
    posterior_probabilities: pd.DataFrame | None
    class_parameters: pd.DataFrame | None
    centroids: np.ndarray | None


@beartype
class GMM(TrajectoryModel):
    """Estimate growth-mixture trajectories via R `lcmm::hlme`.

    GMM (Growth Mixture Model) fits latent trajectory classes while allowing
    class-specific random-effects structure around longitudinal trends. This can
    capture heterogeneous growth patterns that are not well represented by a
    single population trajectory.

    High-level algorithm (pseudocode):

    ```text
    1) Validate one-outcome longitudinal input (id, time, value).
    2) Optionally verify R + lcmm availability.
    3) Call the internal R bridge (`fit_gmm_lcmm.R`) with model hyperparameters.
    4) Parse bridge outputs: assignments, posterior probabilities, parameters.
    5) Map numeric class IDs to trajectory names and return standard outputs.
    ```

    Args:
        n_trajectories (int): Number of latent trajectory classes.
        random_state (int | None): Optional random seed forwarded to R.
        max_iter (int): Maximum optimisation iterations.
        n_init (int): Number of multi-start initialisations.
        random_effects (str): Random-effects structure (`intercept` or
            `intercept_slope`).
        diagonal_random_effect_covariance (bool): Whether random-effect
            covariance is diagonal (`idiag` in `lcmm`).
        class_specific_random_effect_scale (bool): Whether class-specific
            random-effect scale is enabled (`nwg` in `lcmm`).
        warn_on_small_classes (bool): Whether to warn when estimated classes are
            very small.
        min_class_proportion (float): Small-class warning threshold.
        check_r_dependencies (bool): Whether to validate R dependencies before
            fitting.
        rscript_path (str | None): Optional explicit path to `Rscript`.
        trajectory_names (Sequence[str] | None): Optional custom class names.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.build_trajectories.trajectories.gmm.gmm import GMM

        model = GMM(n_trajectories=4, random_state=42, max_iter=200, n_init=20)
        ```
    """

    metadata = ComponentMetadata(
        name="GMM",
        full_name="Growth Mixture Model (GMM)",
        abstract_description=(
            "Growth mixture modelling using R lcmm::hlme with an internal "
            "Python-R bridge."
        ),
        tutorial_goal=(
            "Estimate trajectory classes while allowing class-specific random "
            "effects around growth patterns."
        ),
        tutorial_how_it_works=(
            "Fits growth-mixture trajectories in R (`lcmm::hlme`) with "
            "mixture-model flexibility and assigns class memberships."
        ),
    )
    max_value_cols = 1

    def __init__(
        self,
        n_trajectories: int,
        random_state: int | None = None,
        *,
        max_iter: int = 200,
        n_init: int = 20,
        random_effects: str = "intercept_slope",
        diagonal_random_effect_covariance: bool = False,
        class_specific_random_effect_scale: bool = False,
        warn_on_small_classes: bool = True,
        min_class_proportion: float = 0.03,
        check_r_dependencies: bool = True,
        rscript_path: str | None = None,
        trajectory_names: Sequence[str] | None = None,
    ) -> None:
        """Initialise GMM builder (R/lcmm backend).

        Args:
            n_trajectories: Number of latent classes.
            random_state: Optional random seed forwarded to R.
            max_iter: Maximum number of optimisation iterations.
            n_init: Number of random-start attempts via `gridsearch`.
            random_effects:
                Random-effects structure (`intercept` or `intercept_slope`).
            diagonal_random_effect_covariance:
                Whether random-effect covariance is constrained diagonal (`idiag`).
            class_specific_random_effect_scale:
                Whether class-specific random-effect scale is enabled (`nwg`).
            warn_on_small_classes: Whether to warn on tiny classes.
            min_class_proportion: Threshold for tiny-class warnings.
            check_r_dependencies: Validate R + `lcmm` availability before fitting.
            rscript_path: Optional explicit `Rscript` path.
            trajectory_names: Optional custom trajectory labels.
        """
        self.n_trajectories = n_trajectories
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_effects = random_effects
        self.diagonal_random_effect_covariance = diagonal_random_effect_covariance
        self.class_specific_random_effect_scale = class_specific_random_effect_scale
        self.warn_on_small_classes = warn_on_small_classes
        self.min_class_proportion = min_class_proportion
        self.check_r_dependencies = check_r_dependencies
        self.rscript_path = rscript_path
        self.trajectory_names = (
            tuple(trajectory_names) if trajectory_names is not None else None
        )

    def fit_(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> TrajectoryResult:
        """Fit GMM via R and return standardised trajectory outputs.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Single outcome column used for growth
                mixture modelling.

        Returns:
            TrajectoryResult: Trajectory assignments and optional posterior/class
                outputs parsed from the R bridge.
        """
        value_col = self._single_value_col(value_cols)
        if self.check_r_dependencies:
            self._assert_r_dependencies()

        bridge_output = self._run_bridge(
            data=data[[id_col, time_col, value_col]].copy(),
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
        )
        assignments = bridge_output.assignments.copy()
        labels = assignments["trajectory_id"].to_numpy(dtype=int)
        self._warn_small_classes(labels)

        trajectory_names = normalise_trajectory_names(
            self.n_trajectories,
            self.trajectory_names,
        )
        assignments["trajectory_name"] = [trajectory_names[idx] for idx in labels]

        return TrajectoryResult(
            n_trajectories=self.n_trajectories,
            trajectory_names=trajectory_names,
            assignments=assignments,
            centroids=bridge_output.centroids,
            posterior_probabilities=bridge_output.posterior_probabilities,
            class_parameters=bridge_output.class_parameters,
            class_covariances=None,
        )

    def transform_(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> pd.DataFrame:
        """Transform fitted GMM trajectories into a standard output dataset.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Single outcome column used for growth
                mixture modelling.

        Returns:
            pd.DataFrame: Transformed dataset as a pandas DataFrame.
        """
        _ = data, id_col, time_col, value_cols
        return build_trajectory_dataset(self._last_result)

    def verify_input(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> None:
        """Validate input assumptions and hyperparameter ranges.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Single outcome column used for growth
                mixture modelling.
        """
        value_col = self._single_value_col(value_cols)
        required = [id_col, time_col, value_col]
        self._ensure_required_columns(data, required)
        self._ensure_non_empty(data)
        self._ensure_time_numeric(data[time_col])
        if not pd.api.types.is_numeric_dtype(data[value_col]):
            raise ValueError("GMM requires a numeric outcome column.")
        if data[required].isna().any().any():
            raise ValueError(
                "GMM requires complete data with no missing values in id, time, or "
                "value columns."
            )
        if data[time_col].nunique() < 2:
            raise ValueError("GMM requires at least two distinct time points.")
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories must be a positive integer.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.n_init <= 0:
            raise ValueError("n_init must be a positive integer.")
        if self.random_effects not in {"intercept", "intercept_slope"}:
            raise ValueError("random_effects must be 'intercept' or 'intercept_slope'.")
        if not (0.0 <= self.min_class_proportion <= 1.0):
            raise ValueError("min_class_proportion must be in [0, 1].")

    @staticmethod
    def _single_value_col(value_cols: Iterable[str]) -> str:
        """Validate and extract the single outcome column."""
        cols = TrajectoryModel._ensure_value_cols(value_cols)
        if len(cols) != 1:
            raise ValueError(
                "GMM currently supports exactly one longitudinal outcome column."
            )
        return cols[0]

    def _assert_r_dependencies(self) -> None:
        """Validate R runtime and required package availability."""
        rscript = self._resolve_rscript()
        check = subprocess.run(
            [
                rscript,
                "--vanilla",
                "-e",
                "if (!requireNamespace('lcmm', quietly = TRUE)) quit(status = 42)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if check.returncode == 42:
            raise RuntimeError(
                "R package 'lcmm' is not installed. Install it in your R "
                "environment: install.packages('lcmm')"
            )
        if check.returncode != 0:
            stderr = (check.stderr or "").strip()
            raise RuntimeError(
                "Failed to validate R dependencies before GMM fitting. "
                f"R stderr: {stderr}"
            )

    def _resolve_rscript(self) -> str:
        """Resolve the `Rscript` executable path."""
        if self.rscript_path:
            return self.rscript_path
        resolved = shutil.which("Rscript")
        if resolved is None:
            raise RuntimeError(
                "Rscript was not found in PATH. Install R or set `rscript_path`."
            )
        return resolved

    def _run_bridge(
        self,
        *,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        value_col: str,
    ) -> _GMMBridgeOutput:
        """Execute R bridge script and parse output artefacts."""
        script_path = Path(__file__).resolve().parent / "fit_gmm_lcmm.R"
        if not script_path.exists():
            raise RuntimeError(f"GMM bridge script is missing: {script_path}")

        with tempfile.TemporaryDirectory(prefix="gmm_bridge_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input_long.csv"
            output_dir = tmpdir_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(input_path, index=False)

            command = [
                self._resolve_rscript(),
                "--vanilla",
                str(script_path),
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--id-col",
                id_col,
                "--time-col",
                time_col,
                "--value-col",
                value_col,
                "--n-trajectories",
                str(self.n_trajectories),
                "--max-iter",
                str(self.max_iter),
                "--n-init",
                str(self.n_init),
                "--random-effects",
                self.random_effects,
                "--idiag",
                str(self.diagonal_random_effect_covariance).lower(),
                "--nwg",
                str(self.class_specific_random_effect_scale).lower(),
            ]
            if self.random_state is not None:
                command.extend(["--seed", str(self.random_state)])

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                raise RuntimeError(
                    "GMM bridge execution failed. "
                    f"Exit code: {completed.returncode}. "
                    f"STDERR: {stderr}. STDOUT: {stdout}"
                )

            assignments = pd.read_csv(output_dir / "assignments.csv")
            expected_assignment_cols = {"subject_id", "trajectory_id"}
            if not expected_assignment_cols.issubset(assignments.columns):
                raise RuntimeError(
                    "GMM bridge output is missing required assignment columns."
                )
            assignments = assignments[["subject_id", "trajectory_id"]].copy()
            assignments["trajectory_id"] = assignments["trajectory_id"].astype(int)

            invalid = assignments["trajectory_id"].isin(
                list(range(self.n_trajectories))
            )
            if not invalid.all():
                raise RuntimeError(
                    "GMM returned trajectory IDs outside expected range."
                )

            posterior_path = output_dir / "posterior_probabilities.csv"
            posterior: pd.DataFrame | None = None
            if posterior_path.exists():
                posterior = pd.read_csv(posterior_path)

            class_params_path = output_dir / "class_parameters.csv"
            class_params: pd.DataFrame | None = None
            if class_params_path.exists():
                class_params = pd.read_csv(class_params_path)

            centroids_path = output_dir / "centroids.csv"
            centroids: np.ndarray | None = None
            if centroids_path.exists():
                centroids_df = pd.read_csv(centroids_path)
                centroids = self._parse_centroids(centroids_df)

            return _GMMBridgeOutput(
                assignments=assignments,
                posterior_probabilities=posterior,
                class_parameters=class_params,
                centroids=centroids,
            )

    def _parse_centroids(self, centroids_df: pd.DataFrame) -> np.ndarray:
        """Convert long centroid rows to `(n_classes, n_times)` matrix."""
        required = {"trajectory_id", "time", "value"}
        if not required.issubset(centroids_df.columns):
            raise RuntimeError("GMM centroid output is missing required columns.")
        pivot = (
            centroids_df.pivot_table(
                index="trajectory_id",
                columns="time",
                values="value",
                aggfunc="first",
            )
            .reindex(range(self.n_trajectories))
            .sort_index(axis=1)
        )
        if pivot.isna().any().any():
            raise RuntimeError("GMM centroid matrix contains missing values.")
        return pivot.to_numpy(dtype=float)

    def _warn_small_classes(self, labels: np.ndarray) -> None:
        """Warn when any class has proportion below threshold."""
        if not self.warn_on_small_classes:
            return
        counts = np.bincount(labels, minlength=self.n_trajectories)
        proportions = counts / counts.sum()
        flagged = [
            (idx, int(count), float(prop))
            for idx, (count, prop) in enumerate(zip(counts, proportions, strict=False))
            if prop < self.min_class_proportion
        ]
        if not flagged:
            return
        flagged_text = ", ".join(
            f"class_{idx}: n={count}, p={prop:.3f}" for idx, count, prop in flagged
        )
        console.print(
            "[yellow]Warning:[/yellow] GMM estimated one or more very small "
            f"classes (threshold={self.min_class_proportion:.3f}): {flagged_text}."
        )
