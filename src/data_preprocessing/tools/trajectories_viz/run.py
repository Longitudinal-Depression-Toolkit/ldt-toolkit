from __future__ import annotations

import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_bool,
    as_choice,
    as_required_int,
    as_required_string,
    normalise_key,
    run_with_validation,
)
from src.utils.errors import InputValidationError

from .render import (
    build_class_spaghetti_figure,
    build_mean_profiles_figure,
    build_trajectory_sizes_figure,
    prepare_combined_dataset_for_trajectory_viz,
)


def run_trajectories_viz(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate interactive visualisations for trajectory assignments.

    This tool takes a combined dataset containing longitudinal values and
    assigned trajectory labels, then produces a Plotly HTML figure.

    Available visualisation techniques:

    | Technique | What it shows | Typical use |
    | --- | --- | --- |
    | `mean_profiles` | Mean trajectory per class across time, optionally with uncertainty bands and sampled individual curves. | Compare average trajectory shapes between classes. |
    | `class_spaghetti` | Individual subject curves per class, optionally overlaid with class mean profiles. | Inspect within-class heterogeneity and outliers. |
    | `trajectory_sizes` | Bar chart of number of subjects in each trajectory class. | Check class balance and tiny-class risk. |

    Args:
        technique (str): Visualisation technique key.
        params (Mapping[str, Any]): Figure configuration, input/output paths,
            and required column mappings for the selected technique.

    Returns:
        dict[str, Any]: Generated HTML path and execution metadata.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.trajectories_viz.run import run_trajectories_viz

        result = run_trajectories_viz(
            technique="mean_profiles",
            params={
                "input_path": "./outputs/combined_with_trajectories.csv",
                "output_html": "./outputs/trajectory_mean_profiles.html",
                "title": "Trajectory Mean Profiles",
                "id_col": "subject_id",
                "time_col": "wave",
                "value_col": "depression_score",
                "trajectory_col": "trajectory_id",
                "trajectory_name_col": "trajectory_name",
                "show_error_band": True,
                "band_metric": "sem",
                "sample_per_trajectory": 25,
                "random_state": 42,
            },
        )
        ```
    """

    return run_with_validation(
        lambda: _run_trajectories_viz(technique=technique, params=params)
    )


def _run_trajectories_viz(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="trajectories_viz",
        technique_id=technique,
        provided_params=dict(params),
    )

    viz_mode = normalise_key(technique)
    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_html = Path(as_required_string(resolved, "output_html")).expanduser()
    title = as_required_string(resolved, "title")
    open_browser = as_bool(
        resolved.get("open_browser", False), field_name="open_browser"
    )

    data = pd.read_csv(input_path)

    if viz_mode == "mean_profiles":
        id_col = as_required_string(resolved, "id_col")
        time_col = as_required_string(resolved, "time_col")
        value_col = as_required_string(resolved, "value_col")
        trajectory_col = as_required_string(resolved, "trajectory_col")
        trajectory_name_col = as_required_string(resolved, "trajectory_name_col")
        show_error_band = as_bool(
            resolved.get("show_error_band", True),
            field_name="show_error_band",
        )
        band_metric = as_choice(
            as_required_string(resolved, "band_metric"),
            choices=("sem", "std"),
            field_name="band_metric",
        )
        sample_per_trajectory = as_required_int(
            resolved,
            "sample_per_trajectory",
            minimum=0,
        )
        random_state = as_required_int(resolved, "random_state")

        viz_data = prepare_combined_dataset_for_trajectory_viz(
            data=data,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            trajectory_col=trajectory_col,
            trajectory_name_col=trajectory_name_col,
        )
        figure = build_mean_profiles_figure(
            data=viz_data,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            trajectory_col=trajectory_col,
            trajectory_name_col=trajectory_name_col,
            show_error_band=show_error_band,
            band_metric=band_metric,
            sample_per_trajectory=sample_per_trajectory,
            random_state=random_state,
            title=title,
        )
    elif viz_mode == "class_spaghetti":
        id_col = as_required_string(resolved, "id_col")
        time_col = as_required_string(resolved, "time_col")
        value_col = as_required_string(resolved, "value_col")
        trajectory_col = as_required_string(resolved, "trajectory_col")
        trajectory_name_col = as_required_string(resolved, "trajectory_name_col")
        sample_per_trajectory = as_required_int(
            resolved,
            "sample_per_trajectory",
            minimum=0,
        )
        show_mean_profile = as_bool(
            resolved.get("show_mean_profile", True),
            field_name="show_mean_profile",
        )
        random_state = as_required_int(resolved, "random_state")

        viz_data = prepare_combined_dataset_for_trajectory_viz(
            data=data,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            trajectory_col=trajectory_col,
            trajectory_name_col=trajectory_name_col,
        )
        figure = build_class_spaghetti_figure(
            data=viz_data,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            trajectory_col=trajectory_col,
            trajectory_name_col=trajectory_name_col,
            sample_per_trajectory=sample_per_trajectory,
            show_mean_profile=show_mean_profile,
            random_state=random_state,
            title=title,
        )
    elif viz_mode == "trajectory_sizes":
        id_col = as_required_string(resolved, "id_col")
        trajectory_col = as_required_string(resolved, "trajectory_col")
        trajectory_name_col = as_required_string(resolved, "trajectory_name_col")

        if id_col not in data.columns or trajectory_col not in data.columns:
            raise InputValidationError(
                f"Missing required columns: {id_col}, {trajectory_col}"
            )
        if trajectory_name_col not in data.columns:
            data = data.copy()
            data[trajectory_name_col] = data[trajectory_col].astype(str)

        viz_data = data[[id_col, trajectory_col, trajectory_name_col]].copy()
        figure = build_trajectory_sizes_figure(
            data=viz_data,
            id_col=id_col,
            trajectory_col=trajectory_col,
            trajectory_name_col=trajectory_name_col,
            title=title,
        )
    else:
        raise InputValidationError(
            f"Unsupported trajectories-viz technique: {technique}"
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    if open_browser:
        webbrowser.open(output_html.resolve().as_uri())

    return {
        "technique": viz_mode,
        "output_html": str(output_html.resolve()),
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "title": title,
        "opened_browser": bool(open_browser),
    }
