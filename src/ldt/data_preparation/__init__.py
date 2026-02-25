from __future__ import annotations

import importlib
from typing import Any

from ldt.utils.templates.tools.data_preparation import (
    DataPreparationTool,
    ToolParameterDefinition,
)

from .catalog import (
    list_data_conversion_techniques,
    list_data_preparation_presets,
    list_synthetic_techniques,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ConversionFileResult": (
        "ldt.data_preparation.tools.data_conversion.converters._tabular_converter",
        "ConversionFileResult",
    ),
    "ConversionBatchResult": (
        "ldt.data_preparation.tools.data_conversion.converters._tabular_converter",
        "ConversionBatchResult",
    ),
    "CsvToParquet": (
        "ldt.data_preparation.tools.data_conversion.converters.csv_to_parquet",
        "CsvToParquet",
    ),
    "CsvToStata": (
        "ldt.data_preparation.tools.data_conversion.converters.csv_to_stata",
        "CsvToStata",
    ),
    "ParquetToCsv": (
        "ldt.data_preparation.tools.data_conversion.converters.parquet_to_csv",
        "ParquetToCsv",
    ),
    "ParquetToStata": (
        "ldt.data_preparation.tools.data_conversion.converters.parquet_to_stata",
        "ParquetToStata",
    ),
    "StataToCsv": (
        "ldt.data_preparation.tools.data_conversion.converters.stata_to_csv",
        "StataToCsv",
    ),
    "StataToParquet": (
        "ldt.data_preparation.tools.data_conversion.converters.stata_to_parquet",
        "StataToParquet",
    ),
    "TrendPatterns": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns",
        "TrendPatterns",
    ),
    "SyntheticWaveDataset": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns",
        "SyntheticWaveDataset",
    ),
    "TrajectoryPatternSpec": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns",
        "TrajectoryPatternSpec",
    ),
    "TrajectoryFeatureSpec": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns",
        "TrajectoryFeatureSpec",
    ),
    "EventShockRecovery": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.event_shock_recovery",
        "EventShockRecovery",
    ),
    "PiecewiseChangepoint": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.piecewise_changepoint",
        "PiecewiseChangepoint",
    ),
    "MissingDataScenarios": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.missing_data_scenarios",
        "MissingDataScenarios",
    ),
    "HarmonisationChallenge": (
        "ldt.data_preparation.tools.synthetic_data_generation.generators.harmonisation_challenge",
        "HarmonisationChallenge",
    ),
    "PrepareMCSByLEAP": (
        "ldt.data_preparation.presets.prepare_mcs_by_leap.tool",
        "PrepareMCSByLEAP",
    ),
}

__all__ = [
    "DataPreparationTool",
    "ToolParameterDefinition",
    "list_data_conversion_techniques",
    "list_data_preparation_presets",
    "list_synthetic_techniques",
    "ConversionFileResult",
    "ConversionBatchResult",
    "CsvToParquet",
    "CsvToStata",
    "ParquetToCsv",
    "ParquetToStata",
    "StataToCsv",
    "StataToParquet",
    "TrendPatterns",
    "SyntheticWaveDataset",
    "TrajectoryPatternSpec",
    "TrajectoryFeatureSpec",
    "EventShockRecovery",
    "PiecewiseChangepoint",
    "MissingDataScenarios",
    "HarmonisationChallenge",
    "PrepareMCSByLEAP",
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
