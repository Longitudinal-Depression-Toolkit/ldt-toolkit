from .class_spaghetti import build_class_spaghetti_technique
from .mean_profiles import build_mean_profiles_technique
from .trajectory_sizes import build_trajectory_sizes_technique

TECHNIQUE_RUNNERS = {
    "class_spaghetti": build_class_spaghetti_technique,
    "mean_profiles": build_mean_profiles_technique,
    "trajectory_sizes": build_trajectory_sizes_technique,
}

__all__ = [
    "TECHNIQUE_RUNNERS",
    "build_class_spaghetti_technique",
    "build_mean_profiles_technique",
    "build_trajectory_sizes_technique",
]
