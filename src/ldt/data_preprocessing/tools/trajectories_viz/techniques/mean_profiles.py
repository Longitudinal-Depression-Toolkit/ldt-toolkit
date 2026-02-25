from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from beartype import beartype

from ..render import (
    build_mean_profiles_figure,
    prepare_combined_dataset_for_trajectory_viz,
)
from ._inputs import as_bool, as_choice, as_int, as_required_string


@beartype
def build_mean_profiles_technique(
    *,
    data: pd.DataFrame,
    title: str,
    params: Mapping[str, Any],
) -> Any:
    """Build the `mean_profiles` trajectory visualisation."""

    id_col = as_required_string(params, "id_col")
    time_col = as_required_string(params, "time_col")
    value_col = as_required_string(params, "value_col")
    trajectory_col = as_required_string(params, "trajectory_col")
    trajectory_name_col = as_required_string(params, "trajectory_name_col")
    show_error_band = as_bool(params, "show_error_band", default=True)
    band_metric = as_choice(params, "band_metric", choices=("sem", "std"))
    sample_per_trajectory = as_int(params, "sample_per_trajectory", minimum=0)
    random_state = as_int(params, "random_state")

    viz_data = prepare_combined_dataset_for_trajectory_viz(
        data=data,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
        trajectory_col=trajectory_col,
        trajectory_name_col=trajectory_name_col,
    )
    return build_mean_profiles_figure(
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
