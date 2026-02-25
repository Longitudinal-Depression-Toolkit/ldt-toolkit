from __future__ import annotations

import warnings

# Suppress known third-party pkg_resources deprecation warnings in CLI output.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"stopit(\..*)?$",
)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"PipelineProfiler\._plot_pipeline_matrix",
)
