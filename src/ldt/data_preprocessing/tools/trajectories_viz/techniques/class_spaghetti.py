from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from beartype import beartype

from ..render import (
    build_class_spaghetti_figure,
    prepare_combined_dataset_for_trajectory_viz,
)
from ._inputs import as_bool, as_int, as_required_string


@beartype
def build_class_spaghetti_technique(
    *,
    data: pd.DataFrame,
    title: str,
    params: Mapping[str, Any],
) -> Any:
    """Build the `class_spaghetti` trajectory visualisation."""

    id_col = as_required_string(params, "id_col")
    time_col = as_required_string(params, "time_col")
    value_col = as_required_string(params, "value_col")
    trajectory_col = as_required_string(params, "trajectory_col")
    trajectory_name_col = as_required_string(params, "trajectory_name_col")
    sample_per_trajectory = as_int(params, "sample_per_trajectory", minimum=0)
    show_mean_profile = as_bool(params, "show_mean_profile", default=True)
    random_state = as_int(params, "random_state")

    viz_data = prepare_combined_dataset_for_trajectory_viz(
        data=data,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
        trajectory_col=trajectory_col,
        trajectory_name_col=trajectory_name_col,
    )
    return build_class_spaghetti_figure(
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
