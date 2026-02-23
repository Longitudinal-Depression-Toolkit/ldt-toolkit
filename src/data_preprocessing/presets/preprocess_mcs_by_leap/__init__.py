from src.utils.errors import InputValidationError


def preprocess_mcs_by_leap_profile() -> dict[str, str]:
    """Return static profile metadata for the incoming preset.

    Returns:
        dict[str, str]: Mapping of names to values.
    """

    return {
        "preset": "preprocess_mcs_by_leap",
        "status": "incoming",
        "message": "This preset is not available yet.",
    }


def run_preprocess_mcs_by_leap(*, params: dict[str, object]) -> dict[str, object]:
    """Raise an explicit availability error for this preset.

    Args:
        params (dict[str, object]): Parameter mapping provided by the caller.

    Returns:
        dict[str, object]: Dictionary containing tool results.
    """

    _ = params
    raise InputValidationError(
        "Preset `preprocess_mcs_by_leap` is incoming and not available yet."
    )


__all__ = [
    "preprocess_mcs_by_leap_profile",
    "run_preprocess_mcs_by_leap",
]
