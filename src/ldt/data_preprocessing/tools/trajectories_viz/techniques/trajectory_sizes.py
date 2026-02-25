from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError

from ..render import build_trajectory_sizes_figure
from ._inputs import as_required_string


@beartype
def build_trajectory_sizes_technique(
    *,
    data: pd.DataFrame,
    title: str,
    params: Mapping[str, Any],
) -> Any:
    """Build the `trajectory_sizes` trajectory visualisation."""

    id_col = as_required_string(params, "id_col")
    trajectory_col = as_required_string(params, "trajectory_col")
    trajectory_name_col = as_required_string(params, "trajectory_name_col")

    if id_col not in data.columns or trajectory_col not in data.columns:
        raise InputValidationError(
            f"Missing required columns: {id_col}, {trajectory_col}"
        )

    if trajectory_name_col not in data.columns:
        data = data.copy()
        data[trajectory_name_col] = data[trajectory_col].astype(str)

    viz_data = data[[id_col, trajectory_col, trajectory_name_col]].copy()
    return build_trajectory_sizes_figure(
        data=viz_data,
        id_col=id_col,
        trajectory_col=trajectory_col,
        trajectory_name_col=trajectory_name_col,
        title=title,
    )
