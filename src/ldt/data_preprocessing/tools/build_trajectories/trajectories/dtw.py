from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from beartype import beartype
from tslearn.clustering import TimeSeriesKMeans

from ldt.utils.metadata import ComponentMetadata

from ..trajectory import (
    TrajectoryModel,
    TrajectoryResult,
    build_trajectory_dataset,
    normalise_trajectory_names,
)


@beartype
class DTWKMeans(TrajectoryModel):
    """Cluster trajectories with Dynamic Time Warping (DTW) k-means.

    DTW-based clustering groups subjects by longitudinal shape similarity rather
    than strict pointwise alignment, which is useful when subjects follow
    similar trends with slight timing shifts.

    High-level algorithm (pseudocode):

    ```text
    1) Pivot long data to a 3D tensor: subjects x timepoints x features.
    2) Compute DTW-aware cluster assignments with tslearn TimeSeriesKMeans.
    3) Build subject-level assignments (trajectory_id, trajectory_name).
    4) Store cluster centroids and expose unified trajectory outputs.
    ```

    Args:
        n_trajectories (int): Number of trajectory clusters to estimate.
        random_state (int | None): Optional random seed for reproducibility.
        trajectory_names (Sequence[str] | None): Optional custom class names;
            defaults to `cluster_0`, `cluster_1`, ...

    Examples:
        ```python
        from ldt.data_preprocessing.tools.build_trajectories.trajectories.dtw import DTWKMeans

        model = DTWKMeans(n_trajectories=4, random_state=42)
        ```
    """

    metadata = ComponentMetadata(
        name="dtw_kmeans",
        full_name="Dynamic Time Warping k-means",
        abstract_description=(
            "Distance-based clustering with DTW: align longitudinal sequences and "
            "apply k-means to group similar trajectory shapes."
        ),
        tutorial_goal=(
            "Group subjects with similarly shaped longitudinal trajectories "
            "without parametric growth assumptions."
        ),
        tutorial_how_it_works=(
            "Computes shape-aware distances with Dynamic Time Warping and "
            "groups subjects via k-means on trajectory similarity."
        ),
    )

    def __init__(
        self,
        n_trajectories: int,
        random_state: int | None = None,
        *,
        trajectory_names: Sequence[str] | None = None,
    ) -> None:
        """Initialise a DTW k-means trajectory model.

        Args:
            n_trajectories: Number of trajectory clusters.
            random_state: Optional random seed.
            trajectory_names: Optional custom trajectory names.
        """
        self.n_trajectories = n_trajectories
        self.random_state = random_state
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
        """Fit DTW k-means and return standardised trajectory results.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Outcome columns used to build
                trajectory sequences.

        Returns:
            TrajectoryResult: Trajectory assignments and DTW cluster centroids.
        """
        values, _, subject_ids = self._pivot_longitudinal(
            data, id_col=id_col, time_col=time_col, value_cols=value_cols
        )
        series = values
        labels, centroids = self._tslearn_dtw_kmeans(series)
        trajectory_names = normalise_trajectory_names(
            self.n_trajectories, self.trajectory_names
        )
        assignments = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "trajectory_id": labels,
                "trajectory_name": [trajectory_names[idx] for idx in labels],
            }
        )
        return TrajectoryResult(
            n_trajectories=self.n_trajectories,
            trajectory_names=trajectory_names,
            assignments=assignments,
            centroids=centroids,
            posterior_probabilities=None,
            class_parameters=None,
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
        """Transform fitted DTW k-means trajectories.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Outcome columns used to build
                trajectory sequences.

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
        """Validate DTW k-means input assumptions.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Outcome columns used to build
                trajectory sequences.
        """
        value_cols = self._ensure_value_cols(value_cols)
        required = [id_col, time_col, *value_cols]
        self._ensure_required_columns(data, required)
        self._ensure_non_empty(data)
        self._ensure_time_numeric(data[time_col])
        if data[required].isna().any().any():
            raise ValueError(
                "DTW k-means requires complete data with no missing values in id, "
                "time, or value columns."
            )
        if data[time_col].nunique() < 2:
            raise ValueError("DTW k-means requires at least two distinct time points.")
        pivot = (
            data[required]
            .pivot_table(index=id_col, columns=time_col, values=value_cols)
            .sort_index(axis=1)
        )
        if pivot.isna().any().any():
            raise ValueError(
                "DTW k-means requires complete observations for every subject and "
                "time point."
            )

    @staticmethod
    def _pivot_longitudinal(
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Pivot long-format data to subject-time-feature tensor.

        Args:
            data: Long-format longitudinal data.
            id_col: Subject identifier column.
            time_col: Time/wave column.
            value_cols: Longitudinal value columns.

        Returns:
            tuple[np.ndarray, np.ndarray, list[int]]: Values tensor, time axis,
            and ordered subject IDs.
        """
        value_cols = list(value_cols)
        pivot = (
            data[[id_col, time_col, *value_cols]]
            .pivot_table(index=id_col, columns=time_col, values=value_cols)
            .sort_index(axis=1)
        )
        pivot = pivot.dropna(axis=0, how="any")
        subject_ids = pivot.index.to_list()
        times = pivot.columns.levels[1].to_numpy(dtype=float)
        values = pivot.to_numpy().reshape(len(subject_ids), len(value_cols), len(times))
        values = np.transpose(values, (0, 2, 1))
        return values, times, subject_ids

    def _tslearn_dtw_kmeans(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run tslearn DTW k-means on a pre-pivoted series tensor.

        Args:
            series: Time-series tensor with shape
                `(n_subjects, n_timepoints, n_features)`.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cluster labels and centroids.
        """
        model = TimeSeriesKMeans(
            n_clusters=self.n_trajectories,
            metric="dtw",
            random_state=self.random_state,
        )
        labels = model.fit_predict(series)
        centroids = model.cluster_centers_
        if centroids.shape[-1] == 1:
            centroids = centroids[:, :, 0]
        return labels.astype(int), centroids
