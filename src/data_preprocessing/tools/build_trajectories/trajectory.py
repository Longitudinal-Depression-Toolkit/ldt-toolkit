from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from beartype import beartype

from src.utils.metadata import ComponentMetadata


@beartype
@dataclass(frozen=True)
class TrajectoryResult:
    """Structured result produced by fitted trajectory models.

    Attributes:
        n_trajectories (int): Number of trajectories.
        trajectory_names (tuple[str, ...]): Trajectory names.
        assignments (pd.DataFrame): Assignments.
        centroids (np.ndarray | None): Centroids.
        posterior_probabilities (pd.DataFrame | None): Posterior probabilities.
        class_parameters (pd.DataFrame | None): Class parameters.
        class_covariances (np.ndarray | None): Class covariances.

    """

    n_trajectories: int
    trajectory_names: tuple[str, ...]
    assignments: pd.DataFrame
    centroids: np.ndarray | None = None
    posterior_probabilities: pd.DataFrame | None = None
    class_parameters: pd.DataFrame | None = None
    class_covariances: np.ndarray | None = None


TModel = TypeVar("TModel", bound="TrajectoryModel")


@beartype
class TrajectoryModel:
    """Base interface for trajectory models.

    Attributes:
        max_value_cols (int | None): Max value cols.

    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Trajectory Model",
    )
    max_value_cols: int | None = None

    def save(self, path: str | Path) -> None:
        """Serialise the trajectory model instance to disk.

        Args:
            path (str | Path): Filesystem path used by the workflow.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            pickle.dump(self, file)

    def fit_(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> TrajectoryResult:
        """Fit trajectories (core hook).

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Column names used by this workflow.

        Returns:
            TrajectoryResult: Result object for this operation.
        """
        raise NotImplementedError

    def verify_input(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> None:
        """Validate inputs prior to fitting.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Column names used by this workflow.
        """
        _ = data, id_col, time_col, value_cols

    def transform_(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> pd.DataFrame:
        """Transform fitted trajectories (core hook).

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Column names used by this workflow.

        Returns:
            pd.DataFrame: Transformed dataset as a pandas DataFrame.
        """
        raise NotImplementedError

    def fit(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> None:
        """Fit trajectories and store the fitted result.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Column names used by this workflow.
        """
        self.verify_input(data, id_col=id_col, time_col=time_col, value_cols=value_cols)
        result = self.fit_(
            data, id_col=id_col, time_col=time_col, value_cols=value_cols
        )
        self._last_result = result
        return None

    def transform(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> pd.DataFrame:
        """Transform trajectories into an assignments dataset.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Column names used by this workflow.

        Returns:
            pd.DataFrame: Transformed dataset as a pandas DataFrame.
        """
        if not hasattr(self, "_last_result"):
            raise RuntimeError("fit must be called before transform.")
        return self.transform_(
            data, id_col=id_col, time_col=time_col, value_cols=value_cols
        )

    @classmethod
    def load(cls: type[TModel], path: str | Path) -> TModel:
        """Load a previously serialised trajectory model instance.

        Args:
            path (str | Path): Filesystem path used by the workflow.

        Returns:
            TModel: Result object for this operation.
        """
        source = Path(path)
        with source.open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, TrajectoryModel):
            raise TypeError("Serialised object is not a TrajectoryModel instance.")
        if not isinstance(loaded, cls):
            raise TypeError(
                "Serialised model is "
                f"{type(loaded).__name__}, expected {cls.__name__}."
            )
        return loaded

    @staticmethod
    def _ensure_value_cols(value_cols: Iterable[str]) -> list[str]:
        """Validate that at least one value column was provided.

        Args:
            value_cols: Candidate value-column names.

        Returns:
            list[str]: Normalised list of value-column names.
        """
        value_cols = list(value_cols)
        if not value_cols:
            raise ValueError("value_cols must contain at least one column.")
        return value_cols

    @staticmethod
    def _ensure_required_columns(data: pd.DataFrame, required: Iterable[str]) -> None:
        """Validate required columns exist in the input dataframe.

        Args:
            data: Input dataframe.
            required: Required column names.

        Returns:
            None.
        """
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    @staticmethod
    def _ensure_non_empty(data: pd.DataFrame) -> None:
        """Validate that the input dataframe is not empty.

        Args:
            data: Input dataframe.

        Returns:
            None.
        """
        if data.empty:
            raise ValueError("Input data is empty.")

    @staticmethod
    def _ensure_time_numeric(time_values: pd.Series) -> None:
        """Validate that time values can be parsed as numeric.

        Args:
            time_values: Series of time-axis values.

        Returns:
            None.
        """
        coerced = pd.to_numeric(time_values, errors="coerce")
        if coerced.isna().any():
            raise ValueError("time_col must contain numeric values only.")


@beartype
def normalise_trajectory_names(
    n_trajectories: int, trajectory_names: Sequence[str] | None
) -> tuple[str, ...]:
    """Normalise trajectory names.

    Args:
        n_trajectories (int): Number of trajectories.
        trajectory_names (Sequence[str] | None): Trajectory names.

    Returns:
        tuple[str, ...]: Tuple of resolved values.
    """
    if trajectory_names is None:
        return tuple(f"cluster_{idx}" for idx in range(n_trajectories))
    if len(trajectory_names) != n_trajectories:
        raise ValueError("trajectory_names length must match n_trajectories")
    return tuple(trajectory_names)


@beartype
def build_trajectory_dataset(result: TrajectoryResult) -> pd.DataFrame:
    """Build a trajectory dataset.

    Args:
        result (TrajectoryResult): Result object used by this workflow.

    Returns:
        pd.DataFrame: Transformed dataset as a pandas DataFrame.
    """
    if result.posterior_probabilities is None:
        return result.assignments.copy()
    if "subject_id" in result.posterior_probabilities.columns:
        return result.assignments.merge(
            result.posterior_probabilities,
            on="subject_id",
            how="left",
            suffixes=("", "_posterior"),
        )
    return result.assignments.join(
        result.posterior_probabilities, how="left", suffixes=("", "_posterior")
    )
