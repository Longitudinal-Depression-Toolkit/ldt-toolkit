from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.errors import InputValidationError


def prepare_combined_dataset_for_trajectory_viz(
    *,
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    trajectory_col: str,
    trajectory_name_col: str,
) -> pd.DataFrame:
    """Validate and normalise combined trajectory data before plotting.

    The returned frame keeps only required columns, coerces numeric outcome
    values, removes incomplete rows, and standardises trajectory labels for
    plotting functions.

    Args:
        data (pd.DataFrame): Combined dataset containing longitudinal values and
            trajectory assignments.
        id_col (str): Subject identifier column name.
        time_col (str): Time or wave column name.
        value_col (str): Outcome/value column name.
        trajectory_col (str): Trajectory label column name.
        trajectory_name_col (str): Human-readable trajectory name column.

    Returns:
        pd.DataFrame: Cleaned plotting-ready dataset.
    """

    _ensure_columns(
        data, [id_col, time_col, value_col, trajectory_col, trajectory_name_col]
    )

    prepared = data[
        [id_col, time_col, value_col, trajectory_col, trajectory_name_col]
    ].copy()

    prepared[value_col] = pd.to_numeric(prepared[value_col], errors="coerce")
    prepared = prepared.dropna(subset=[trajectory_col, time_col, value_col]).copy()
    if prepared.empty:
        raise InputValidationError(
            "No rows remain for trajectory visualisation after filtering missing required fields."
        )

    prepared[trajectory_name_col] = prepared[trajectory_name_col].astype(str)
    prepared[time_col] = _coerce_time_axis(prepared[time_col])
    return prepared


def build_mean_profiles_figure(
    *,
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    trajectory_col: str,
    trajectory_name_col: str,
    show_error_band: bool,
    band_metric: str,
    sample_per_trajectory: int,
    random_state: int,
    title: str,
) -> go.Figure:
    """Build the `mean_profiles` visualisation.

    This chart displays one mean curve per trajectory class across time.
    Optional uncertainty bands (`std` or `sem`) and sampled individual curves
    help communicate dispersion around each class mean.

    Args:
        data (pd.DataFrame): Plotting-ready dataset.
        id_col (str): Subject identifier column name.
        time_col (str): Time or wave column name.
        value_col (str): Outcome/value column name.
        trajectory_col (str): Trajectory ID column name.
        trajectory_name_col (str): Trajectory display-name column.
        show_error_band (bool): Whether to draw variability bands around means.
        band_metric (str): Variability metric for bands (`sem` or `std`).
        sample_per_trajectory (int): Number of individual subject curves to
            overlay per trajectory (`0` disables sampling overlays).
        random_state (int): Random seed used for sampling subject curves.
        title (str): Figure title.

    Returns:
        go.Figure: Plotly figure for mean trajectory profiles.
    """

    summary = (
        data.groupby([trajectory_col, trajectory_name_col, time_col], as_index=False)[
            value_col
        ]
        .agg(mean="mean", std="std", count="count")
        .copy()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["sem"] = summary["sem"].fillna(0.0)

    trajectory_order = _trajectory_order(summary, trajectory_col, trajectory_name_col)
    colors = px.colors.qualitative.Safe
    figure = go.Figure()

    for idx, (trajectory_id, trajectory_name) in enumerate(trajectory_order):
        color = colors[idx % len(colors)]
        group = summary[summary[trajectory_col] == trajectory_id].sort_values(
            by=time_col
        )
        x_values = group[time_col].to_list()
        y_values = group["mean"].to_numpy(dtype=float)

        if show_error_band:
            band_values = group[band_metric].to_numpy(dtype=float)
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values + band_values,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"group_{trajectory_id}",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values - band_values,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, alpha=0.18),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"group_{trajectory_id}",
                )
            )

        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=trajectory_name,
                line={"color": color, "width": 3},
                marker={"size": 7},
                legendgroup=f"group_{trajectory_id}",
            )
        )

        if sample_per_trajectory > 0:
            trajectory_rows = data[data[trajectory_col] == trajectory_id]
            unique_subjects = trajectory_rows[id_col].dropna().unique()
            sample_size = min(sample_per_trajectory, len(unique_subjects))
            if sample_size > 0:
                rng = np.random.default_rng(seed=random_state + idx)
                sampled_subjects = rng.choice(
                    unique_subjects,
                    size=sample_size,
                    replace=False,
                )
                sampled_curves = (
                    trajectory_rows[trajectory_rows[id_col].isin(sampled_subjects)]
                    .pivot_table(
                        index=id_col, columns=time_col, values=value_col, aggfunc="mean"
                    )
                    .sort_index(axis=1)
                )
                for _, row in sampled_curves.iterrows():
                    figure.add_trace(
                        go.Scatter(
                            x=sampled_curves.columns.to_list(),
                            y=row.to_numpy(dtype=float),
                            mode="lines",
                            line={"color": _hex_to_rgba(color, alpha=0.28), "width": 1},
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=f"group_{trajectory_id}",
                        )
                    )

    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=time_col,
        yaxis_title=value_col,
        legend_title="Trajectory",
        hovermode="x unified",
    )
    return figure


def build_class_spaghetti_figure(
    *,
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    trajectory_col: str,
    trajectory_name_col: str,
    sample_per_trajectory: int,
    show_mean_profile: bool,
    random_state: int,
    title: str,
) -> go.Figure:
    """Build the `class_spaghetti` visualisation.

    This chart overlays individual subject trajectories within each class
    (spaghetti plot), optionally adding one mean curve per class for easier
    class-level interpretation.

    Args:
        data (pd.DataFrame): Plotting-ready dataset.
        id_col (str): Subject identifier column name.
        time_col (str): Time or wave column name.
        value_col (str): Outcome/value column name.
        trajectory_col (str): Trajectory ID column name.
        trajectory_name_col (str): Trajectory display-name column.
        sample_per_trajectory (int): Number of subject curves to plot per class
            (`0` means all available subjects).
        show_mean_profile (bool): Whether to overlay class mean profiles.
        random_state (int): Random seed for subject sampling.
        title (str): Figure title.

    Returns:
        go.Figure: Plotly figure for class-wise spaghetti trajectories.
    """

    colors = px.colors.qualitative.Safe
    figure = go.Figure()
    trajectory_order = _trajectory_order(data, trajectory_col, trajectory_name_col)

    for idx, (trajectory_id, trajectory_name) in enumerate(trajectory_order):
        color = colors[idx % len(colors)]
        class_rows = data[data[trajectory_col] == trajectory_id]
        class_curves = (
            class_rows.pivot_table(
                index=id_col, columns=time_col, values=value_col, aggfunc="mean"
            )
            .sort_index(axis=1)
            .dropna(how="all")
        )
        if class_curves.empty:
            continue

        n_available = len(class_curves)
        n_to_plot = (
            n_available
            if sample_per_trajectory == 0
            else min(sample_per_trajectory, n_available)
        )
        if n_to_plot <= 0:
            continue

        if n_to_plot < n_available:
            rng = np.random.default_rng(seed=random_state + idx)
            selected_ids = rng.choice(
                class_curves.index.to_numpy(), size=n_to_plot, replace=False
            )
            class_curves = class_curves.loc[selected_ids]

        x_axis = class_curves.columns.to_list()
        for _, row in class_curves.iterrows():
            figure.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=row.to_numpy(dtype=float),
                    mode="lines",
                    line={"color": _hex_to_rgba(color, alpha=0.22), "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"group_{trajectory_id}",
                )
            )

        if show_mean_profile:
            mean_profile = (
                class_rows.groupby(time_col, as_index=False)[value_col]
                .mean()
                .sort_values(by=time_col)
            )
            figure.add_trace(
                go.Scatter(
                    x=mean_profile[time_col].to_list(),
                    y=mean_profile[value_col].to_numpy(dtype=float),
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                    marker={"size": 7},
                    name=f"{trajectory_name} (mean)",
                    legendgroup=f"group_{trajectory_id}",
                )
            )

    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=time_col,
        yaxis_title=value_col,
        legend_title="Trajectory",
        hovermode="x unified",
    )
    return figure


def build_trajectory_sizes_figure(
    *,
    data: pd.DataFrame,
    id_col: str,
    trajectory_col: str,
    trajectory_name_col: str,
    title: str,
) -> go.Figure:
    """Build the `trajectory_sizes` visualisation.

    This chart reports class membership counts at subject level, which is
    useful to assess class balance and detect tiny trajectory groups.

    Args:
        data (pd.DataFrame): Dataset containing subject-level trajectory labels.
        id_col (str): Subject identifier column name.
        trajectory_col (str): Trajectory ID column name.
        trajectory_name_col (str): Trajectory display-name column.
        title (str): Figure title.

    Returns:
        go.Figure: Plotly bar chart of trajectory class sizes.
    """

    subject_level = data[[id_col, trajectory_col, trajectory_name_col]].drop_duplicates(
        subset=[id_col], keep="first"
    )
    sizes = (
        subject_level.groupby([trajectory_col, trajectory_name_col], as_index=False)[
            id_col
        ]
        .nunique()
        .rename(columns={id_col: "n_subjects"})
    )
    order = _trajectory_order(sizes, trajectory_col, trajectory_name_col)
    order_lookup = {item[0]: idx for idx, item in enumerate(order)}
    sizes["_order"] = sizes[trajectory_col].map(order_lookup)
    sizes = sizes.sort_values(by="_order")
    colors = px.colors.qualitative.Safe

    figure = go.Figure(
        data=[
            go.Bar(
                x=sizes[trajectory_name_col].to_list(),
                y=sizes["n_subjects"].to_list(),
                marker_color=[colors[idx % len(colors)] for idx in range(len(sizes))],
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Trajectory",
        yaxis_title="Number of subjects",
    )
    return figure


def _ensure_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise InputValidationError(
            "Missing required columns in combined dataset: "
            + ", ".join(sorted(missing))
        )


def _coerce_time_axis(time_series: pd.Series) -> pd.Series:
    as_numeric = pd.to_numeric(time_series, errors="coerce")
    if as_numeric.notna().all():
        return as_numeric

    as_datetime = pd.to_datetime(time_series, errors="coerce")
    if as_datetime.notna().all():
        return as_datetime

    return time_series.astype(str)


def _trajectory_order(
    data: pd.DataFrame,
    trajectory_col: str,
    trajectory_name_col: str,
) -> list[tuple[object, str]]:
    pairs = data[[trajectory_col, trajectory_name_col]].drop_duplicates()
    numeric_ids = pd.to_numeric(pairs[trajectory_col], errors="coerce")
    if numeric_ids.notna().all():
        pairs = pairs.assign(_sort=numeric_ids).sort_values(by="_sort")
    else:
        pairs = pairs.assign(_sort=pairs[trajectory_name_col].astype(str)).sort_values(
            by="_sort"
        )
    return [
        (row[trajectory_col], str(row[trajectory_name_col]))
        for _, row in pairs.iterrows()
    ]


def _hex_to_rgba(hex_color: str, *, alpha: float) -> str:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(80, 80, 80, {alpha})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"
