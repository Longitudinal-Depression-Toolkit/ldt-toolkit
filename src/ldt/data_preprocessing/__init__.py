from __future__ import annotations

import importlib
from typing import Any

from ldt.utils.templates.tools.data_preprocessing import (
    DataPreprocessingTool,
    ToolParameterDefinition,
)

from .catalog import (
    list_aggregate_long_to_cross_sectional_techniques,
    list_build_trajectories_techniques,
    list_clean_dataset_techniques,
    list_combine_dataset_with_trajectories_techniques,
    list_harmonise_categories_techniques,
    list_missing_imputation_techniques,
    list_pivot_long_to_wide_techniques,
    list_remove_columns_techniques,
    list_rename_feature_techniques,
    list_show_table_techniques,
    list_tool_techniques,
    list_trajectories_viz_techniques,
    resolve_technique_with_defaults,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AggregateLongToCrossSectional": (
        "ldt.data_preprocessing.tools.aggregate_long_to_cross_sectional.run",
        "AggregateLongToCrossSectional",
    ),
    "AggregateLongToCrossSectionalResult": (
        "ldt.data_preprocessing.tools.aggregate_long_to_cross_sectional.run",
        "AggregateLongToCrossSectionalResult",
    ),
    "BuildTrajectories": (
        "ldt.data_preprocessing.tools.build_trajectories.run",
        "BuildTrajectories",
    ),
    "CleanDataset": (
        "ldt.data_preprocessing.tools.clean_dataset.run",
        "CleanDataset",
    ),
    "CleanDatasetResult": (
        "ldt.data_preprocessing.tools.clean_dataset.run",
        "CleanDatasetResult",
    ),
    "CombineDatasetWithTrajectories": (
        "ldt.data_preprocessing.tools.combine_dataset_with_trajectories.run",
        "CombineDatasetWithTrajectories",
    ),
    "CombineDatasetWithTrajectoriesResult": (
        "ldt.data_preprocessing.tools.combine_dataset_with_trajectories.run",
        "CombineDatasetWithTrajectoriesResult",
    ),
    "HarmoniseCategories": (
        "ldt.data_preprocessing.tools.harmonise_categories.run",
        "HarmoniseCategories",
    ),
    "HarmoniseCategoriesResult": (
        "ldt.data_preprocessing.tools.harmonise_categories.run",
        "HarmoniseCategoriesResult",
    ),
    "MICEImputationResult": (
        "ldt.data_preprocessing.tools.missing_imputation.imputers",
        "MICEImputationResult",
    ),
    "MICEImputer": (
        "ldt.data_preprocessing.tools.missing_imputation.imputers",
        "MICEImputer",
    ),
    "MissingImputation": (
        "ldt.data_preprocessing.tools.missing_imputation.run",
        "MissingImputation",
    ),
    "MissingImputer": (
        "ldt.data_preprocessing.tools.missing_imputation.imputers",
        "MissingImputer",
    ),
    "PivotLongToWide": (
        "ldt.data_preprocessing.tools.pivot_long_to_wide.run",
        "PivotLongToWide",
    ),
    "PivotLongToWideResult": (
        "ldt.data_preprocessing.tools.pivot_long_to_wide.run",
        "PivotLongToWideResult",
    ),
    "PreprocessMCSByLEAP": (
        "ldt.data_preprocessing.presets.preprocess_mcs_by_leap.tool",
        "PreprocessMCSByLEAP",
    ),
    "RemoveColumns": (
        "ldt.data_preprocessing.tools.remove_columns.run",
        "RemoveColumns",
    ),
    "RemoveColumnsResult": (
        "ldt.data_preprocessing.tools.remove_columns.run",
        "RemoveColumnsResult",
    ),
    "RenameFeature": (
        "ldt.data_preprocessing.tools.rename_feature.run",
        "RenameFeature",
    ),
    "RenameFeatureResult": (
        "ldt.data_preprocessing.tools.rename_feature.run",
        "RenameFeatureResult",
    ),
    "ShowTable": (
        "ldt.data_preprocessing.tools.show_table.run",
        "ShowTable",
    ),
    "ShowTableResult": (
        "ldt.data_preprocessing.tools.show_table.run",
        "ShowTableResult",
    ),
    "TrajectoriesViz": (
        "ldt.data_preprocessing.tools.trajectories_viz.run",
        "TrajectoriesViz",
    ),
    "TrajectoriesVizResult": (
        "ldt.data_preprocessing.tools.trajectories_viz.run",
        "TrajectoriesVizResult",
    ),
    "TrajectoryModel": (
        "ldt.data_preprocessing.tools.build_trajectories.trajectory",
        "TrajectoryModel",
    ),
    "TrajectoryResult": (
        "ldt.data_preprocessing.tools.build_trajectories.trajectory",
        "TrajectoryResult",
    ),
    "build_trajectory_dataset": (
        "ldt.data_preprocessing.tools.build_trajectories.trajectory",
        "build_trajectory_dataset",
    ),
    "discover_trajectory_builders": (
        "ldt.data_preprocessing.tools.build_trajectories.discovery",
        "discover_trajectory_builders",
    ),
    "normalise_trajectory_names": (
        "ldt.data_preprocessing.tools.build_trajectories.trajectory",
        "normalise_trajectory_names",
    ),
    "parse_columns_csv": (
        "ldt.data_preprocessing.tools.remove_columns.run",
        "parse_columns_csv",
    ),
    "discover_missing_imputers": (
        "ldt.data_preprocessing.tools.missing_imputation.discovery",
        "discover_missing_imputers",
    ),
}

__all__ = [
    "AggregateLongToCrossSectional",
    "AggregateLongToCrossSectionalResult",
    "BuildTrajectories",
    "CleanDataset",
    "CleanDatasetResult",
    "CombineDatasetWithTrajectories",
    "CombineDatasetWithTrajectoriesResult",
    "DataPreprocessingTool",
    "HarmoniseCategories",
    "HarmoniseCategoriesResult",
    "MICEImputationResult",
    "MICEImputer",
    "MissingImputation",
    "MissingImputer",
    "PivotLongToWide",
    "PivotLongToWideResult",
    "PreprocessMCSByLEAP",
    "RemoveColumns",
    "RemoveColumnsResult",
    "RenameFeature",
    "RenameFeatureResult",
    "ShowTable",
    "ShowTableResult",
    "ToolParameterDefinition",
    "TrajectoriesViz",
    "TrajectoriesVizResult",
    "TrajectoryModel",
    "TrajectoryResult",
    "build_trajectory_dataset",
    "discover_trajectory_builders",
    "list_aggregate_long_to_cross_sectional_techniques",
    "list_build_trajectories_techniques",
    "list_clean_dataset_techniques",
    "list_combine_dataset_with_trajectories_techniques",
    "list_harmonise_categories_techniques",
    "list_missing_imputation_techniques",
    "list_pivot_long_to_wide_techniques",
    "list_remove_columns_techniques",
    "list_rename_feature_techniques",
    "list_show_table_techniques",
    "list_tool_techniques",
    "list_trajectories_viz_techniques",
    "normalise_trajectory_names",
    "parse_columns_csv",
    "resolve_technique_with_defaults",
    "discover_missing_imputers",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, symbol_name = target
    module = importlib.import_module(module_path)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
