from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from beartype import beartype

from .tool import PreprocessMCSByLEAP


@beartype
def preprocess_mcs_by_leap_profile() -> dict[str, str]:
    """Return static profile metadata for the incoming preset."""

    return {
        "preset": "preprocess_mcs_by_leap",
        "status": "incoming",
        "message": (
            "This preset will preprocess prepared longitudinal datasets into an "
            "ML-ready dataset that is cleaned enough for modelling and experimentation."
        ),
    }


@beartype
def run_preprocess_mcs_by_leap(*, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run the incoming preset wrapper for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `PreprocessMCSByLEAP` directly from `ldt.data_preprocessing`.

    Args:
        params (Mapping[str, Any]): Placeholder payload for future preset
            parameters.

    Returns:
        dict[str, Any]: Placeholder result payload from the incoming preset.
    """

    return PreprocessMCSByLEAP().fit_preprocess(**dict(params))
