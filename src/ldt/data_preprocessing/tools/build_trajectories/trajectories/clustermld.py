from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.preprocessing import SplineTransformer

from ldt.utils.metadata import ComponentMetadata

from ..trajectory import (
    TrajectoryModel,
    TrajectoryResult,
    build_trajectory_dataset,
    normalise_trajectory_names,
)


@dataclass
class ClusterLeaf:
    """Leaf node representation for clusterMLD hierarchy construction.

    Attributes:
        id_in (list[Hashable]): Id in.
        xx (np.ndarray): Xx.
        xy (np.ndarray): Xy.
        y2 (np.ndarray): Y2.
        n_in (int): Number of in.
        ssr0 (np.ndarray): Ssr0.

    """

    id_in: list[Hashable]
    xx: np.ndarray
    xy: np.ndarray
    y2: np.ndarray
    n_in: int
    ssr0: np.ndarray


@dataclass
class HierarchyDiagnostics:
    """Diagnostic payload produced during hierarchy construction.

    Attributes:
        clusters_by_k (dict[int, list[ClusterLeaf]]): Clusters by k.
        gap_b_by_k (dict[int, float]): Gap b by k.
        ch_by_k (dict[int, float]): Ch by k.
        gap_dist_by_k (dict[int, float]): Gap dist by k.
        b_dist_by_k (dict[int, float]): B dist by k.
        w_dist_by_k (dict[int, float]): W dist by k.
        selected_k (int): Selected k.
        selection_metric (str): Selection metric.

    """

    clusters_by_k: dict[int, list[ClusterLeaf]]
    gap_b_by_k: dict[int, float]
    ch_by_k: dict[int, float]
    gap_dist_by_k: dict[int, float]
    b_dist_by_k: dict[int, float]
    w_dist_by_k: dict[int, float]
    selected_k: int
    selection_metric: str


@beartype
class ClusterMLD(TrajectoryModel):
    """Estimate trajectory classes via spline-based hierarchical clustering.

    `ClusterMLD` represents each subject trajectory with spline summaries, then
    performs hierarchical merges driven by longitudinal fit criteria. It can
    either use a fixed `n_trajectories` or select class count automatically via
    a model-selection metric.

    High-level algorithm (pseudocode):

    ```text
    1) Keep complete subject trajectories and collect (time, value) arrays.
    2) Build spline basis for the time axis and transform trajectories.
    3) Create one initial leaf per subject with spline fit statistics.
    4) Iteratively merge leaves using weighted merge-cost distances.
    5) Select final number of trajectories (fixed or metric-based).
    6) Emit subject assignments and hierarchy diagnostics.
    ```

    Args:
        n_trajectories (int | None): Fixed number of trajectory classes; if
            `None`, select automatically.
        random_state (int | None): Random seed placeholder for API consistency.
        spline_degree (int): Degree of spline basis functions.
        n_internal_knots (int): Number of internal knots on the time axis.
        dist_metric (str): Merge-distance variant (`W` or `UnW`).
        preprocess (bool): Whether to merge sparse leaves before hierarchy.
        weight_func (str): Outcome weighting strategy (`standardise`,
            `softmax`, or `equal`).
        selection_metric (str): Automatic class-count metric (`gapb` or `ch`).
        trajectory_names (Sequence[str] | None): Optional custom class names.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.build_trajectories.trajectories.clustermld import ClusterMLD

        model = ClusterMLD(n_trajectories=None, spline_degree=3, n_internal_knots=4)
        ```
    """

    metadata = ComponentMetadata(
        name="clusterMLD",
        full_name="ClusterMLD (Spline Hierarchical Clustering)",
        abstract_description=(
            "Hierarchical clustering of spline-based longitudinal summaries using "
            "merge-cost dissimilarity to form trajectory groups."
        ),
        tutorial_goal=(
            "Cluster subjects using spline-based summaries of longitudinal "
            "profiles when trajectories are noisy or irregular."
        ),
        tutorial_how_it_works=(
            "Builds spline-based longitudinal summaries and performs "
            "hierarchical clustering using merge-cost dissimilarity."
        ),
    )

    def __init__(
        self,
        n_trajectories: int | None = None,
        random_state: int | None = None,
        *,
        spline_degree: int = 3,
        n_internal_knots: int = 3,
        dist_metric: str = "W",
        preprocess: bool = True,
        weight_func: str = "standardise",
        selection_metric: str = "gapb",
        trajectory_names: Sequence[str] | None = None,
    ) -> None:
        """Initialise a clusterMLD trajectory model.

        Args:
            n_trajectories: Fixed number of clusters, or `None` for auto-select.
            random_state: Optional random seed placeholder.
            spline_degree: Spline basis degree.
            n_internal_knots: Number of internal spline knots.
            dist_metric: Merge distance metric (`W` or `UnW`).
            preprocess: Whether to merge sparse-subject leaves before hierarchy.
            weight_func: Outcome weighting strategy.
            selection_metric: Auto-selection metric (`gapb` or `ch`).
            trajectory_names: Optional custom trajectory names.
        """
        self.n_trajectories = n_trajectories
        self.random_state = random_state
        self.spline_degree = spline_degree
        self.n_internal_knots = n_internal_knots
        self.dist_metric = dist_metric
        self.preprocess = preprocess
        self.weight_func = weight_func
        self.selection_metric = selection_metric
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
        """Fit clusterMLD and return standardised trajectory results.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Outcome columns used for trajectory
                class construction.

        Returns:
            TrajectoryResult: Trajectory assignments plus hierarchy diagnostics.
        """
        value_cols = list(value_cols)
        complete_data, excluded_subject_ids = self._split_complete_subjects(
            data, id_col=id_col, time_col=time_col, value_cols=value_cols
        )
        times, values, ids = self._collect_long_data(
            complete_data, id_col=id_col, time_col=time_col, value_cols=value_cols
        )
        if len(ids) == 0:
            raise ValueError("No valid longitudinal records found for clustering.")
        spline = self._build_spline_transformer(times)
        spline.fit(times.reshape(-1, 1))
        basis = spline.transform(times.reshape(-1, 1))

        values = self._standardise_values(values)
        leaves, obs_no, p_var, y_dim = self._build_leaves(basis, values, ids)
        weights = self._compute_weights(basis, values, leaves, obs_no, p_var, y_dim)
        dropped_subject_ids: list[Hashable] = []
        if self.preprocess:
            leaves, weights = self._preprocess_leaves(
                basis,
                values,
                leaves,
                obs_no,
                p_var,
                weights,
            )
        else:
            kept: list[ClusterLeaf] = []
            for leaf in leaves:
                if leaf.n_in > p_var:
                    kept.append(leaf)
                else:
                    dropped_subject_ids.extend(leaf.id_in)
            leaves = kept
            if not leaves:
                raise ValueError(
                    "No subjects with sufficient observations to fit spline-based "
                    "clusterMLD when preprocess=False."
                )

        total_obs = int(np.sum([leaf.n_in for leaf in leaves]))
        xx_all = np.sum(np.stack([leaf.xx for leaf in leaves]), axis=0)
        xy_all = np.sum(np.stack([leaf.xy for leaf in leaves]), axis=0)
        y2_all = np.sum(np.stack([leaf.y2 for leaf in leaves]), axis=0)
        ssr_all = self._ssr_from_components(xx_all, xy_all, y2_all)
        e_sigma = np.sum(weights * (ssr_all / max(total_obs - p_var, 1)))

        hierarchy = self._build_hierarchy(
            leaves=leaves,
            weights=weights,
            p_var=p_var,
            xx_all=xx_all,
            xy_all=xy_all,
            y2_all=y2_all,
            ssr_all=ssr_all,
            total_obs=total_obs,
            e_sigma=float(e_sigma),
        )
        selected_k = self._choose_n_trajectories(hierarchy)
        clusters = hierarchy.clusters_by_k[selected_k]

        trajectory_names = normalise_trajectory_names(selected_k, self.trajectory_names)
        assignments = self._build_assignments(
            clusters, trajectory_names, excluded_subject_ids + dropped_subject_ids
        )
        diagnostics = self._build_hierarchy_diagnostics_frame(hierarchy)
        return TrajectoryResult(
            n_trajectories=selected_k,
            trajectory_names=trajectory_names,
            assignments=assignments,
            centroids=None,
            posterior_probabilities=None,
            class_parameters=diagnostics,
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
        """Transform fitted clusterMLD trajectories.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Outcome columns used for trajectory
                class construction.

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
        """Validate clusterMLD input assumptions and hyperparameter ranges.

        Args:
            data (pd.DataFrame): Input dataset.
            id_col (str): Subject identifier column name.
            time_col (str): Time or wave column name.
            value_cols (Iterable[str]): Outcome columns used for trajectory
                class construction.
        """
        value_cols = self._ensure_value_cols(value_cols)
        required = [id_col, time_col, *value_cols]
        self._ensure_required_columns(data, required)
        self._ensure_non_empty(data)
        self._ensure_time_numeric(data[time_col])
        cleaned = data[required].dropna()
        if cleaned.empty:
            raise ValueError(
                "clusterMLD requires at least one complete observation after "
                "dropping missing values."
            )
        if cleaned[time_col].nunique() < 2:
            raise ValueError("clusterMLD requires at least two distinct time points.")
        if self.dist_metric not in {"W", "UnW"}:
            raise ValueError("dist_metric must be 'W' or 'UnW'.")
        if self.weight_func not in {"standardise", "softmax", "equal"}:
            raise ValueError("weight_func must be one of: standardise, softmax, equal.")
        if self.selection_metric.lower() not in {"gapb", "ch"}:
            raise ValueError("selection_metric must be one of: gapb, ch.")
        if self.n_trajectories is not None and self.n_trajectories < 1:
            raise ValueError("n_trajectories must be at least 1 when provided.")
        if self.spline_degree < 1:
            raise ValueError("spline_degree must be >= 1.")
        if self.n_internal_knots < 1:
            raise ValueError("n_internal_knots must be >= 1.")

    @staticmethod
    def _collect_long_data(
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect sorted complete long-format arrays.

        Args:
            data: Long-format longitudinal dataframe.
            id_col: Subject identifier column.
            time_col: Time/wave column.
            value_cols: Longitudinal value columns.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Times, values, and
            subject IDs.
        """
        value_cols = list(value_cols)
        required = [id_col, time_col, *value_cols]
        cleaned = data[required].dropna()
        cleaned = cleaned.sort_values([id_col, time_col])
        times = cleaned[time_col].to_numpy(dtype=float)
        values = cleaned[value_cols].to_numpy(dtype=float)
        ids = cleaned[id_col].to_numpy()
        return times, values, ids

    @staticmethod
    def _split_complete_subjects(
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        value_cols: Sequence[str],
    ) -> tuple[pd.DataFrame, list[Hashable]]:
        """Split data into complete-subject rows and excluded subject IDs.

        Args:
            data: Long-format longitudinal dataframe.
            id_col: Subject identifier column.
            time_col: Time/wave column.
            value_cols: Longitudinal value columns.

        Returns:
            tuple[pd.DataFrame, list[Hashable]]: Complete-subject dataframe and
            excluded subject IDs.
        """
        required = [id_col, time_col, *value_cols]
        subset = data[required].copy()
        subset = subset[subset[id_col].notna()]
        if subset.empty:
            return subset, []

        row_missing = subset[[time_col, *value_cols]].isna().any(axis=1)
        subject_missing = row_missing.groupby(subset[id_col]).any()
        excluded_subject_ids = subject_missing[subject_missing].index.to_list()
        complete_subject_ids = subject_missing[~subject_missing].index.to_list()

        complete = subset[subset[id_col].isin(complete_subject_ids)].copy()
        complete = complete.dropna(subset=[time_col, *value_cols])
        return complete, excluded_subject_ids

    def _build_spline_transformer(self, times: np.ndarray) -> SplineTransformer:
        """Build a spline transformer for observed time values.

        Args:
            times: Numeric time-axis values.

        Returns:
            SplineTransformer: Configured spline transformer.
        """
        times = np.asarray(times, dtype=float)
        if np.min(times) == np.max(times):
            raise ValueError("Time values must contain at least two distinct points.")
        internal = np.quantile(times, np.linspace(0.25, 0.75, self.n_internal_knots))
        internal = np.unique(internal)
        knots = np.concatenate(([times.min()], internal, [times.max()]))
        knots = knots.reshape(-1, 1)
        return SplineTransformer(
            degree=self.spline_degree,
            knots=knots,
            include_bias=True,
        )

    @staticmethod
    def _standardise_values(values: np.ndarray) -> np.ndarray:
        """Standardise values to zero mean and unit variance.

        Args:
            values: Raw value matrix.

        Returns:
            np.ndarray: Standardised values.
        """
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        std = np.where(std == 0, 1.0, std)
        return (values - mean) / std

    @staticmethod
    def _build_leaves(
        basis: np.ndarray,
        values: np.ndarray,
        ids: np.ndarray,
    ) -> tuple[list[ClusterLeaf], np.ndarray, int, int]:
        """Build initial one-subject cluster leaves.

        Args:
            basis: Spline basis matrix.
            values: Standardised value matrix.
            ids: Subject IDs aligned with rows in `basis`/`values`.

        Returns:
            tuple[list[ClusterLeaf], np.ndarray, int, int]: Leaves, observation
            counts per subject, number of spline variables, and outcome
            dimension.
        """
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        p_var = basis.shape[1]
        y_dim = values.shape[1]
        obs_no = np.zeros(len(unique_ids), dtype=int)
        leaves: list[ClusterLeaf] = []

        for idx, subject_id in enumerate(unique_ids):
            mask = inverse == idx
            x_in = basis[mask]
            y_in = values[mask]
            obs_no[idx] = x_in.shape[0]
            xx = x_in.T @ x_in
            xy = x_in.T @ y_in
            y2 = np.sum(y_in**2, axis=0)
            inv_xx = np.linalg.pinv(xx)
            ssr0 = y2 - np.diag(xy.T @ inv_xx @ xy)
            leaves.append(
                ClusterLeaf(
                    id_in=[subject_id],
                    xx=xx,
                    xy=xy,
                    y2=y2,
                    n_in=x_in.shape[0],
                    ssr0=ssr0,
                )
            )
        return leaves, obs_no, p_var, y_dim

    def _compute_weights(
        self,
        basis: np.ndarray,
        values: np.ndarray,
        leaves: list[ClusterLeaf],
        obs_no: np.ndarray,
        p_var: int,
        y_dim: int,
    ) -> np.ndarray:
        """Compute outcome weights for merge-cost calculations.

        Args:
            basis: Spline basis matrix.
            values: Standardised value matrix.
            leaves: Current cluster leaves.
            obs_no: Observation counts per subject.
            p_var: Number of spline variables.
            y_dim: Number of outcome dimensions.

        Returns:
            np.ndarray: Outcome weight vector.
        """
        if np.max(obs_no) <= p_var:
            return np.full(y_dim, 1.0 / y_dim)

        xx = basis.T @ basis
        xy = basis.T @ values
        y2 = np.sum(values**2, axis=0)
        ssr_all = self._ssr_from_components(xx, xy, y2)
        ssr0_sum = np.sum(np.stack([leaf.ssr0 for leaf in leaves]), axis=0)
        ssr0_sum = np.where(ssr0_sum == 0, 1e-8, ssr0_sum)
        weights = ssr_all / ssr0_sum - 1.0
        return self._apply_weight_func(weights, y_dim)

    def _apply_weight_func(self, weights: np.ndarray, y_dim: int) -> np.ndarray:
        """Apply configured weighting strategy.

        Args:
            weights: Raw weight vector.
            y_dim: Number of outcome dimensions.

        Returns:
            np.ndarray: Normalised weight vector.
        """
        if self.weight_func == "standardise":
            total = float(np.sum(weights))
            if np.isclose(total, 0.0):
                return np.full(y_dim, 1.0 / y_dim)
            return weights / total
        if self.weight_func == "softmax":
            scaled = (weights - weights.mean()) / (weights.std() + 1e-8)
            exp_w = np.exp(scaled)
            return exp_w / np.sum(exp_w)
        if self.weight_func == "equal":
            return np.full(y_dim, 1.0 / y_dim)
        raise ValueError("weight_func must be 'standardise', 'softmax', or 'equal'")

    def _preprocess_leaves(
        self,
        basis: np.ndarray,
        values: np.ndarray,
        leaves: list[ClusterLeaf],
        obs_no: np.ndarray,
        p_var: int,
        weights: np.ndarray,
    ) -> tuple[list[ClusterLeaf], np.ndarray]:
        """Pre-merge sparse leaves with insufficient observations.

        Args:
            basis: Spline basis matrix.
            values: Standardised value matrix.
            leaves: Current cluster leaves.
            obs_no: Observation counts per subject.
            p_var: Number of spline variables.
            weights: Current outcome weight vector.

        Returns:
            tuple[list[ClusterLeaf], np.ndarray]: Updated leaves and weights.
        """
        if np.min(obs_no) > p_var:
            return leaves, weights

        leaves = list(leaves)
        while True:
            lengths = np.array([leaf.n_in for leaf in leaves])
            low_idx = np.where(lengths <= p_var)[0]
            if low_idx.size == 0:
                break

            best: tuple[int, int, float] | None = None
            for i in low_idx:
                for j in range(len(leaves)):
                    if i == j:
                        continue
                    n_merge = leaves[i].n_in + leaves[j].n_in
                    if n_merge <= p_var:
                        continue
                    ssr_merge = self._merge_ssr(leaves[i], leaves[j])
                    dist = float(np.sum(ssr_merge * weights) / n_merge)
                    if best is None or dist < best[2]:
                        best = (i, j, dist)

            if best is None:
                break

            i, j = best[0], best[1]
            merged = self._merge_leaf(leaves[i], leaves[j])
            leaves = [leaf for k, leaf in enumerate(leaves) if k not in (i, j)]
            leaves.append(merged)
            weights = self._compute_preprocess_weights(leaves, p_var, values.shape[1])
        return leaves, weights

    def _compute_preprocess_weights(
        self,
        leaves: list[ClusterLeaf],
        p_var: int,
        y_dim: int,
    ) -> np.ndarray:
        """Recompute outcome weights after preprocessing merges.

        Args:
            leaves: Current cluster leaves.
            p_var: Number of spline variables.
            y_dim: Number of outcome dimensions.

        Returns:
            np.ndarray: Updated outcome weights.
        """
        enough = [leaf for leaf in leaves if leaf.n_in > p_var]
        if not enough:
            return np.full(y_dim, 1.0 / y_dim)

        xx = np.sum(np.stack([leaf.xx for leaf in enough]), axis=0)
        xy = np.sum(np.stack([leaf.xy for leaf in enough]), axis=0)
        y2 = np.sum(np.stack([leaf.y2 for leaf in enough]), axis=0)
        ssr_enough = self._ssr_from_components(xx, xy, y2)
        ssr0_sum = np.sum(np.stack([leaf.ssr0 for leaf in enough]), axis=0)
        ssr0_sum = np.where(ssr0_sum == 0, 1e-8, ssr0_sum)
        raw_weights = ssr_enough / ssr0_sum - 1.0
        return self._apply_weight_func(raw_weights, y_dim)

    def _merge_clusters(
        self,
        leaves: list[ClusterLeaf],
        weights: np.ndarray,
        p_var: int,
    ) -> list[ClusterLeaf]:
        """Greedily merge clusters until target count is reached.

        Args:
            leaves: Current cluster leaves.
            weights: Outcome weight vector.
            p_var: Number of spline variables.

        Returns:
            list[ClusterLeaf]: Merged clusters.
        """
        leaves = list(leaves)
        while len(leaves) > self.n_trajectories:
            dist_tab = self._distance_matrix(leaves, weights, p_var)
            i, j = np.unravel_index(np.argmin(dist_tab), dist_tab.shape)
            merged = self._merge_leaf(leaves[i], leaves[j])
            leaves = [leaf for k, leaf in enumerate(leaves) if k not in (i, j)]
            leaves.append(merged)
        return leaves

    def _distance_matrix(
        self,
        leaves: list[ClusterLeaf],
        weights: np.ndarray,
        p_var: int,
    ) -> np.ndarray:
        """Compute pairwise merge-cost distance matrix.

        Args:
            leaves: Current cluster leaves.
            weights: Outcome weight vector.
            p_var: Number of spline variables.

        Returns:
            np.ndarray: Upper-triangular distance matrix.
        """
        n_leaf = len(leaves)
        dist_tab = np.full((n_leaf, n_leaf), np.inf)
        for i in range(n_leaf - 1):
            for j in range(i + 1, n_leaf):
                ssr_merge = self._merge_ssr(leaves[i], leaves[j])
                if self.dist_metric == "UnW":
                    dist = np.sum(
                        (ssr_merge - leaves[i].ssr0 - leaves[j].ssr0) * weights
                    )
                elif self.dist_metric == "W":
                    denom = leaves[i].ssr0 + leaves[j].ssr0
                    denom = np.where(denom == 0, 1e-8, denom)
                    ratio = (ssr_merge - leaves[i].ssr0 - leaves[j].ssr0) / denom
                    scale = leaves[i].n_in + leaves[j].n_in - 2 * p_var
                    dist = np.sum(ratio * scale * weights)
                else:
                    raise ValueError("dist_metric must be 'W' or 'UnW'")
                dist_tab[i, j] = dist
        return dist_tab

    def _build_hierarchy(
        self,
        *,
        leaves: list[ClusterLeaf],
        weights: np.ndarray,
        p_var: int,
        xx_all: np.ndarray,
        xy_all: np.ndarray,
        y2_all: np.ndarray,
        ssr_all: np.ndarray,
        total_obs: int,
        e_sigma: float,
    ) -> HierarchyDiagnostics:
        """Build full agglomerative hierarchy and diagnostics.

        Args:
            leaves: Initial cluster leaves.
            weights: Outcome weight vector.
            p_var: Number of spline variables.
            xx_all: Global design cross-product matrix.
            xy_all: Global design-response cross-product matrix.
            y2_all: Global response sum of squares.
            ssr_all: Global residual sum-of-squares vector.
            total_obs: Total number of observations.
            e_sigma: Estimated residual variance scale.

        Returns:
            HierarchyDiagnostics: Hierarchy states and selection metrics.
        """
        current = list(leaves)
        clusters_by_k: dict[int, list[ClusterLeaf]] = {}
        gap_dist_by_k: dict[int, float] = {}
        b_dist_by_k: dict[int, float] = {}
        w_dist_by_k: dict[int, float] = {}

        while True:
            k = len(current)
            clusters_by_k[k] = [self._copy_leaf(leaf) for leaf in current]
            gap_dist_by_k[k] = self._gap_distance(
                clusters=current,
                weights=weights,
                xx_all=xx_all,
                xy_all=xy_all,
                y2_all=y2_all,
                ssr_all=ssr_all,
            )
            b_dist_by_k[k] = self._between_distance(
                clusters=current,
                weights=weights,
                xx_all=xx_all,
                xy_all=xy_all,
            )
            w_dist_by_k[k] = self._within_distance(clusters=current, weights=weights)
            if k == 1:
                break
            dist_tab = self._distance_matrix(current, weights, p_var)
            i, j = np.unravel_index(np.argmin(dist_tab), dist_tab.shape)
            merged = self._merge_leaf(current[i], current[j])
            current = [leaf for idx, leaf in enumerate(current) if idx not in (i, j)]
            current.append(merged)

        gap_b_by_k: dict[int, float] = {}
        ch_by_k: dict[int, float] = {}
        for k in clusters_by_k:
            gap_b_by_k[k] = float(gap_dist_by_k[k] - k * p_var * e_sigma)
            if k <= 1:
                ch_by_k[k] = 0.0
            else:
                denom_w = max(total_obs - k * p_var, 1)
                w_term = w_dist_by_k[k] / denom_w
                if np.isclose(w_term, 0.0):
                    ch_by_k[k] = 0.0
                else:
                    ch_by_k[k] = float((b_dist_by_k[k] / (k - 1)) / w_term)

        return HierarchyDiagnostics(
            clusters_by_k=clusters_by_k,
            gap_b_by_k=gap_b_by_k,
            ch_by_k=ch_by_k,
            gap_dist_by_k=gap_dist_by_k,
            b_dist_by_k=b_dist_by_k,
            w_dist_by_k=w_dist_by_k,
            selected_k=-1,
            selection_metric=self.selection_metric.lower(),
        )

    def _choose_n_trajectories(self, hierarchy: HierarchyDiagnostics) -> int:
        """Choose number of trajectories from hierarchy diagnostics.

        Args:
            hierarchy: Hierarchy diagnostics payload.

        Returns:
            int: Selected number of trajectories.
        """
        max_k = max(hierarchy.clusters_by_k)
        if self.n_trajectories is not None:
            if self.n_trajectories > max_k:
                raise ValueError(
                    "n_trajectories exceeds available hierarchy size after "
                    f"preprocessing ({max_k})."
                )
            hierarchy.selected_k = self.n_trajectories
            hierarchy.selection_metric = "fixed"
            return self.n_trajectories

        metric = self.selection_metric.lower()
        k_values = sorted(hierarchy.clusters_by_k)
        if metric == "ch":
            candidates = [k for k in k_values if k >= 2]
            if not candidates:
                selected = 1
            else:
                selected = max(candidates, key=lambda k: hierarchy.ch_by_k[k])
            hierarchy.selected_k = selected
            hierarchy.selection_metric = "ch"
            return selected

        candidates = [k for k in k_values if 2 <= k <= (max_k - 1)]
        for k in candidates:
            if (
                hierarchy.gap_b_by_k[k] > hierarchy.gap_b_by_k[k - 1]
                and hierarchy.gap_b_by_k[k] > hierarchy.gap_b_by_k[k + 1]
            ):
                hierarchy.selected_k = k
                hierarchy.selection_metric = "gapb"
                return k
        selected = max(k_values, key=lambda k: hierarchy.gap_b_by_k[k])
        hierarchy.selected_k = selected
        hierarchy.selection_metric = "gapb"
        return selected

    @staticmethod
    def _within_distance(clusters: list[ClusterLeaf], weights: np.ndarray) -> float:
        """Compute within-cluster weighted distance.

        Args:
            clusters: Current hierarchy clusters.
            weights: Outcome weights.

        Returns:
            float: Within-cluster distance.
        """
        return float(np.sum([np.sum(leaf.ssr0 * weights) for leaf in clusters]))

    @staticmethod
    def _between_distance(
        *,
        clusters: list[ClusterLeaf],
        weights: np.ndarray,
        xx_all: np.ndarray,
        xy_all: np.ndarray,
    ) -> float:
        """Compute between-cluster weighted distance.

        Args:
            clusters: Current hierarchy clusters.
            weights: Outcome weights.
            xx_all: Global design cross-product matrix.
            xy_all: Global design-response cross-product matrix.

        Returns:
            float: Between-cluster distance.
        """
        beta_all = np.linalg.pinv(xx_all) @ xy_all
        total = 0.0
        for leaf in clusters:
            beta_leaf = np.linalg.pinv(leaf.xx) @ leaf.xy
            diff = beta_all - beta_leaf
            total += float(np.sum(np.diag(diff.T @ leaf.xx @ diff) * weights))
        return total

    @staticmethod
    def _gap_distance(
        *,
        clusters: list[ClusterLeaf],
        weights: np.ndarray,
        xx_all: np.ndarray,
        xy_all: np.ndarray,
        y2_all: np.ndarray,
        ssr_all: np.ndarray,
    ) -> float:
        """Compute gap-distance metric for hierarchy diagnostics.

        Args:
            clusters: Current hierarchy clusters.
            weights: Outcome weights.
            xx_all: Global design cross-product matrix.
            xy_all: Global design-response cross-product matrix.
            y2_all: Global response sum of squares.
            ssr_all: Global residual sum of squares.

        Returns:
            float: Gap-distance value.
        """
        total = 0.0
        for leaf in clusters:
            xx_s = xx_all - leaf.xx
            xy_s = xy_all - leaf.xy
            y2_s = y2_all - leaf.y2
            ssr_s = ClusterMLD._ssr_from_components(xx_s, xy_s, y2_s)
            total += float(np.sum((ssr_all - ssr_s - leaf.ssr0) * weights))
        return total

    @staticmethod
    def _ssr_from_components(
        xx: np.ndarray, xy: np.ndarray, y2: np.ndarray
    ) -> np.ndarray:
        """Compute residual sum of squares from matrix components.

        Args:
            xx: Design cross-product matrix.
            xy: Design-response cross-product matrix.
            y2: Response sum of squares.

        Returns:
            np.ndarray: Residual sum-of-squares vector.
        """
        inv_xx = np.linalg.pinv(xx)
        return y2 - np.diag(xy.T @ inv_xx @ xy)

    @staticmethod
    def _copy_leaf(leaf: ClusterLeaf) -> ClusterLeaf:
        """Deep-copy a `ClusterLeaf` instance.

        Args:
            leaf: Source leaf.

        Returns:
            ClusterLeaf: Copied leaf.
        """
        return ClusterLeaf(
            id_in=list(leaf.id_in),
            xx=leaf.xx.copy(),
            xy=leaf.xy.copy(),
            y2=leaf.y2.copy(),
            n_in=leaf.n_in,
            ssr0=leaf.ssr0.copy(),
        )

    @staticmethod
    def _merge_leaf(left: ClusterLeaf, right: ClusterLeaf) -> ClusterLeaf:
        """Merge two leaves into a single combined leaf.

        Args:
            left: Left leaf.
            right: Right leaf.

        Returns:
            ClusterLeaf: Merged leaf.
        """
        xx = left.xx + right.xx
        xy = left.xy + right.xy
        y2 = left.y2 + right.y2
        inv_xx = np.linalg.pinv(xx)
        ssr0 = y2 - np.diag(xy.T @ inv_xx @ xy)
        return ClusterLeaf(
            id_in=[*left.id_in, *right.id_in],
            xx=xx,
            xy=xy,
            y2=y2,
            n_in=left.n_in + right.n_in,
            ssr0=ssr0,
        )

    @staticmethod
    def _merge_ssr(left: ClusterLeaf, right: ClusterLeaf) -> np.ndarray:
        """Compute merged SSR without constructing full merged leaf object.

        Args:
            left: Left leaf.
            right: Right leaf.

        Returns:
            np.ndarray: SSR vector of merged leaf.
        """
        xx = left.xx + right.xx
        xy = left.xy + right.xy
        y2 = left.y2 + right.y2
        inv_xx = np.linalg.pinv(xx)
        return y2 - np.diag(xy.T @ inv_xx @ xy)

    def _build_assignments(
        self,
        clusters: list[ClusterLeaf],
        trajectory_names: tuple[str, ...],
        excluded_subject_ids: Sequence[Hashable],
    ) -> pd.DataFrame:
        """Build subject-level trajectory assignment dataframe.

        Args:
            clusters: Final hierarchy clusters.
            trajectory_names: Ordered trajectory names.
            excluded_subject_ids: Subjects excluded before clustering.

        Returns:
            pd.DataFrame: Assignment dataframe with IDs and trajectory labels.
        """
        subject_ids: list[Hashable] = []
        labels: list[int] = []
        for label, cluster in enumerate(clusters):
            for subject_id in cluster.id_in:
                subject_ids.append(subject_id)
                labels.append(label)
        assignments = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "trajectory_id": labels,
                "trajectory_name": [trajectory_names[label] for label in labels],
            }
        )
        if excluded_subject_ids:
            na_rows = pd.DataFrame(
                {
                    "subject_id": list(excluded_subject_ids),
                    "trajectory_id": [np.nan] * len(excluded_subject_ids),
                    "trajectory_name": [np.nan] * len(excluded_subject_ids),
                }
            )
            assignments = pd.concat([assignments, na_rows], ignore_index=True)
        return assignments

    @staticmethod
    def _build_hierarchy_diagnostics_frame(
        hierarchy: HierarchyDiagnostics,
    ) -> pd.DataFrame:
        """Build a diagnostics dataframe from hierarchy metrics.

        Args:
            hierarchy: Hierarchy diagnostics payload.

        Returns:
            pd.DataFrame: Diagnostics table keyed by cluster count.
        """
        rows: list[dict[str, Any]] = []
        for k in sorted(hierarchy.clusters_by_k):
            rows.append(
                {
                    "n_clusters": int(k),
                    "gap_b": float(hierarchy.gap_b_by_k[k]),
                    "ch_index": float(hierarchy.ch_by_k[k]),
                    "gap_distance": float(hierarchy.gap_dist_by_k[k]),
                    "between_distance": float(hierarchy.b_dist_by_k[k]),
                    "within_distance": float(hierarchy.w_dist_by_k[k]),
                    "selected_k": int(hierarchy.selected_k),
                    "selection_metric": hierarchy.selection_metric,
                    "is_selected": bool(k == hierarchy.selected_k),
                }
            )
        return pd.DataFrame(rows)
